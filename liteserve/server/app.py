"""FastAPI server for LiteServe with SSE streaming support.

Provides HTTP endpoints for inference, health checks, and metrics.
Implements the full request lifecycle: receive -> route -> schedule ->
infer -> stream tokens back.

Key fix: GPU inference (blocking) runs in a ThreadPoolExecutor so the
async event loop stays responsive for SSE streaming and new requests.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from liteserve.config import LiteServeConfig, load_config
from liteserve.engine.inference import InferenceEngine
from liteserve.engine.kv_cache import PagedKVCache
from liteserve.engine.types import Priority, Request, RequestStatus
from liteserve.metrics.collector import MetricsCollector
from liteserve.models.loader import ModelLoader
from liteserve.router.router import Router
from liteserve.scheduler.scheduler import Scheduler

logger = logging.getLogger(__name__)


# ── Request / Response schemas ───────────────────────────────────────────

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = True
    priority: str = Field(default="normal", pattern="^(normal|high)$")
    model_preference: str = Field(default="auto")


class CompletionResponse(BaseModel):
    request_id: str
    text: str
    usage: dict
    model: str
    finish_reason: str


class HealthResponse(BaseModel):
    status: str
    models: list[dict]
    gpu: dict
    scheduler: dict


# ── Application state ────────────────────────────────────────────────────

class LiteServeApp:
    """Holds all application state and orchestrates the serving pipeline."""

    def __init__(self, config: LiteServeConfig):
        self.config = config
        self.model_loader = ModelLoader()  # auto-detects device
        self.scheduler: Optional[Scheduler] = None
        self.router: Optional[Router] = None
        self.engines: dict[str, InferenceEngine] = {}
        self.metrics = MetricsCollector(
            port=config.metrics.port,
            enabled=config.metrics.enabled,
        )
        self._generation_loop_task: Optional[asyncio.Task] = None
        # Thread pool for running blocking GPU inference off the event loop
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")

    async def startup(self) -> None:
        """Initialize all components on server startup."""
        logger.info("Starting LiteServe...")

        loop = asyncio.get_event_loop()

        # Load models (blocking — run in executor)
        for model_cfg in self.config.models:
            loaded = await loop.run_in_executor(
                self._executor,
                lambda cfg=model_cfg: self.model_loader.load_model(
                    name=cfg.name,
                    model_path=cfg.path,
                    quantization=cfg.quantization,
                    max_model_len=cfg.max_model_len,
                    is_default=cfg.default,
                ),
            )

            self.engines[model_cfg.name] = InferenceEngine(
                model=loaded,
                device=loaded.device,
            )

        # Scheduler uses a lightweight KV-cache just for block-based memory tracking.
        # Actual KV data lives in HuggingFace's native past_key_values per-request.
        first_model = next(iter(self.model_loader.models.values()))
        kv_cache = PagedKVCache(
            num_layers=4,
            num_heads=4,
            head_dim=8,
            block_size=16,
            num_blocks=1024,
            dtype=first_model.dtype,
            device="cpu",
        )
        self.scheduler = Scheduler(
            config=self.config.scheduler,
            kv_cache=kv_cache,
        )

        self.router = Router(
            config=self.config.router,
            model_loader=self.model_loader,
        )

        if self.config.metrics.enabled:
            try:
                self.metrics.start_server()
            except Exception as e:
                logger.warning("Could not start metrics server: %s", e)

        self._generation_loop_task = asyncio.create_task(self._generation_loop())

        logger.info(
            "LiteServe started: %d model(s) loaded on %s",
            len(self.model_loader.models),
            self.model_loader.device,
        )

    async def shutdown(self) -> None:
        """Clean up resources on server shutdown."""
        logger.info("Shutting down LiteServe...")
        if self._generation_loop_task:
            self._generation_loop_task.cancel()
            try:
                await self._generation_loop_task
            except asyncio.CancelledError:
                pass
        self._executor.shutdown(wait=False)
        logger.info("LiteServe shut down.")

    async def _generation_loop(self) -> None:
        """Background loop: form batches, run inference in thread pool.

        This is the continuous batching scheduler loop. It:
        1. Checks the scheduler for work
        2. Forms a batch
        3. Offloads the blocking inference step to a thread
        4. Yields back to the event loop so SSE streams can flush
        """
        loop = asyncio.get_event_loop()

        while True:
            try:
                if self.scheduler.queue_depth == 0:
                    await asyncio.sleep(0.005)
                    continue

                batch = self.scheduler.schedule_step()
                if batch.is_empty:
                    await asyncio.sleep(0.005)
                    continue

                self.metrics.update_batch_size(batch.size)
                self.metrics.update_queue_depth(self.scheduler.pending_count)
                self.metrics.update_requests_in_progress(self.scheduler.active_count)

                # Run blocking inference in thread pool
                await loop.run_in_executor(
                    self._executor,
                    self._run_inference_step,
                    list(batch.requests),
                )

                await asyncio.sleep(0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Generation loop error: %s", e, exc_info=True)
                await asyncio.sleep(0.1)

    def _run_inference_step(self, requests: list[Request]) -> None:
        """Execute one inference step for all active requests (runs in thread)."""
        for request in requests:
            if request.is_finished:
                continue

            engine = self.engines.get(request.assigned_model)
            if engine is None:
                engine = next(iter(self.engines.values()), None)
                if engine is None:
                    continue

            try:
                if request.status == RequestStatus.PREFILL:
                    engine.prefill(request)
                    self.metrics.record_token_generated()
                    if request.time_to_first_token:
                        self.metrics.record_ttft(request.time_to_first_token)
                elif request.status == RequestStatus.GENERATING:
                    engine.decode_step(request)
                    self.metrics.record_token_generated()
            except Exception as e:
                logger.error(
                    "Inference error for request %s: %s",
                    request.request_id[:8], e, exc_info=True,
                )
                request.status = RequestStatus.FAILED
                request.completion_time = time.time()

    async def handle_completion(self, req: CompletionRequest) -> tuple[Request, asyncio.Event]:
        """Handle a completion request end-to-end."""
        request = Request(
            prompt=req.prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            priority=Priority.HIGH if req.priority == "high" else Priority.NORMAL,
            model_preference=req.model_preference,
            stream=req.stream,
        )

        model = self.router.route(request)
        self.metrics.record_routing_decision(model.name)

        gpu_stats = self.metrics.get_gpu_stats()
        self.router.update_system_state(
            gpu_utilization=gpu_stats["gpu_utilization"],
            gpu_memory_utilization=gpu_stats["memory_utilization"],
            queue_depth=self.scheduler.queue_depth,
            active_requests=self.scheduler.active_count,
        )

        engine = self.engines[model.name]
        request.prompt_tokens = engine.tokenize(req.prompt)

        completion_event = asyncio.Event()
        _loop = asyncio.get_event_loop()

        def on_complete(r: Request) -> None:
            _loop.call_soon_threadsafe(completion_event.set)

        self.scheduler.add_request(request, on_complete=on_complete)

        return request, completion_event


# ── FastAPI app factory ──────────────────────────────────────────────────

def create_app(config: Optional[LiteServeConfig] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if config is None:
        config = load_config()

    app_state = LiteServeApp(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await app_state.startup()
        yield
        await app_state.shutdown()

    app = FastAPI(
        title="LiteServe",
        description="A lightweight LLM inference serving engine",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.liteserve = app_state

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        ls: LiteServeApp = app.state.liteserve
        try:
            request, completion_event = await ls.handle_completion(req)
        except Exception as e:
            logger.error("Error handling completion: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        if req.stream:
            return StreamingResponse(
                _stream_tokens(request, completion_event, ls),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Request-ID": request.request_id,
                },
            )
        else:
            await completion_event.wait()
            text = ls.engines[request.assigned_model].decode_tokens(
                request.generated_tokens
            )
            return CompletionResponse(
                request_id=request.request_id,
                text=text,
                usage={
                    "prompt_tokens": len(request.prompt_tokens),
                    "completion_tokens": request.num_generated,
                    "total_tokens": request.total_tokens,
                },
                model=request.assigned_model,
                finish_reason="stop" if request.status == RequestStatus.COMPLETE else "error",
            )

    @app.get("/v1/health")
    async def health():
        ls: LiteServeApp = app.state.liteserve
        gpu_stats = ls.metrics.get_gpu_stats()
        return HealthResponse(
            status="healthy",
            models=ls.model_loader.get_stats()["loaded_models"],
            gpu=gpu_stats,
            scheduler=ls.scheduler.get_stats() if ls.scheduler else {},
        )

    @app.get("/v1/metrics")
    async def metrics():
        ls: LiteServeApp = app.state.liteserve
        summary = ls.metrics.get_summary()
        if ls.scheduler:
            summary["scheduler"] = ls.scheduler.get_stats()
        if ls.router:
            summary["router"] = ls.router.get_stats()
        for name, engine in ls.engines.items():
            summary[f"engine_{name}"] = engine.get_stats()
        return summary

    @app.delete("/v1/requests/{request_id}")
    async def abort_request(request_id: str):
        ls: LiteServeApp = app.state.liteserve
        if ls.scheduler and ls.scheduler.abort_request(request_id):
            return {"status": "aborted", "request_id": request_id}
        raise HTTPException(status_code=404, detail="Request not found")

    @app.get("/v1/models")
    async def list_models():
        ls: LiteServeApp = app.state.liteserve
        return ls.model_loader.get_stats()

    return app


async def _stream_tokens(
    request: Request,
    completion_event: asyncio.Event,
    ls: LiteServeApp,
):
    """SSE token streaming generator."""
    last_token_count = 0
    engine = ls.engines.get(request.assigned_model)
    if engine is None:
        engine = next(iter(ls.engines.values()))

    while not request.is_finished:
        current_count = request.num_generated
        if current_count > last_token_count:
            for i in range(last_token_count, current_count):
                token_id = request.generated_tokens[i]
                token_text = engine.decode_token(token_id)
                data = json.dumps({
                    "token": token_text,
                    "token_id": token_id,
                    "finish_reason": None,
                })
                yield f"data: {data}\n\n"
            last_token_count = current_count
        await asyncio.sleep(0.01)

    current_count = request.num_generated
    if current_count > last_token_count:
        for i in range(last_token_count, current_count):
            token_id = request.generated_tokens[i]
            token_text = engine.decode_token(token_id)
            data = json.dumps({
                "token": token_text,
                "token_id": token_id,
                "finish_reason": None,
            })
            yield f"data: {data}\n\n"

    finish_reason = "stop" if request.status == RequestStatus.COMPLETE else "error"
    data = json.dumps({
        "token": "",
        "finish_reason": finish_reason,
        "usage": {
            "prompt_tokens": len(request.prompt_tokens),
            "completion_tokens": request.num_generated,
            "total_tokens": request.total_tokens,
        },
    })
    yield f"data: {data}\n\n"

    if request.total_latency:
        ls.metrics.record_request_completed(request.total_latency, request.num_generated)


def main():
    """Entry point for running the server."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="LiteServe - LLM Inference Engine")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--host", type=str, default=None, help="Server host")
    parser.add_argument("--port", type=int, default=None, help="Server port")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port

    app = create_app(config)
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        workers=config.server.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
