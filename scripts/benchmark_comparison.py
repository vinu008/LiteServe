#!/usr/bin/env python3
"""Benchmark comparison: sequential vs continuous batching.

Runs the inference engine directly (no HTTP overhead) to measure the
actual throughput multiplier from continuous batching.

Usage:
    python scripts/benchmark_comparison.py
    python scripts/benchmark_comparison.py --model mistralai/Mistral-7B-v0.1 --concurrency 1 4 8 16
"""

import argparse
import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.WARNING)

# Continuation-style prompts generate a real run of tokens across models
# (question prompts can make small chat models emit EOS immediately).
PROMPTS = [
    "Quantum computing works by",
    "The ocean at sunset looked",
    "The capital of France is a city that",
    "A CPU executes instructions by",
    "Regular exercise improves health because",
    "Recursion is a technique where a function",
    "Airplanes are able to fly because",
    "Photosynthesis is the process through which",
]


def run_sequential(engine, prompts, max_tokens):
    """Baseline: process requests one at a time."""
    from liteserve.engine.types import Request

    results = []
    total_tokens = 0
    start = time.perf_counter()

    for prompt in prompts:
        req = Request(prompt=prompt, max_new_tokens=max_tokens, temperature=0.0)
        req.prompt_tokens = engine.tokenize(prompt)

        tokens = list(engine.generate_sequential(req))
        total_tokens += len(tokens)
        results.append({
            "tokens": len(tokens),
            "ttft": req.time_to_first_token,
            "latency": req.total_latency,
        })

    elapsed = time.perf_counter() - start
    return {
        "mode": "sequential",
        "num_requests": len(prompts),
        "total_tokens": total_tokens,
        "elapsed_s": elapsed,
        "throughput": total_tokens / elapsed,
        "results": results,
    }


def run_batched(engine, scheduler_cls, kv_cache_cls, sched_config, prompts, max_tokens):
    """Continuous batching: process requests through the scheduler."""
    import torch
    from liteserve.engine.types import Request, RequestStatus

    kv_cache = kv_cache_cls(
        num_layers=4, num_heads=4, head_dim=8, block_size=16,
        num_blocks=1024, dtype=torch.float32, device="cpu",
    )
    scheduler = scheduler_cls(config=sched_config, kv_cache=kv_cache)

    requests = []
    for prompt in prompts:
        req = Request(prompt=prompt, max_new_tokens=max_tokens, temperature=0.0)
        req.prompt_tokens = engine.tokenize(prompt)
        req.assigned_model = engine.model.name
        scheduler.add_request(req)
        requests.append(req)

    total_tokens = 0
    start = time.perf_counter()

    while any(not r.is_finished for r in requests):
        batch = scheduler.schedule_step()
        if batch.is_empty:
            break

        # Prefill new requests individually, then decode the rest as one batch.
        decode = []
        for request in list(batch.requests):
            if request.is_finished:
                continue
            if request.status == RequestStatus.PREFILL:
                engine.prefill(request)
                total_tokens += 1
            elif request.status == RequestStatus.GENERATING:
                decode.append(request)

        if decode:
            engine.decode_batch(decode)
            total_tokens += len(decode)

    elapsed = time.perf_counter() - start
    results = []
    for req in requests:
        results.append({
            "tokens": req.num_generated,
            "ttft": req.time_to_first_token,
            "latency": req.total_latency,
        })

    return {
        "mode": "continuous_batching",
        "num_requests": len(prompts),
        "total_tokens": total_tokens,
        "elapsed_s": elapsed,
        "throughput": total_tokens / elapsed,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="LiteServe Benchmark Comparison")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--num-requests", type=int, default=8)
    args = parser.parse_args()

    print("=" * 70)
    print("LiteServe Benchmark: Sequential vs Continuous Batching")
    print("=" * 70)

    # Load model
    print(f"\nLoading model: {args.model}")
    from liteserve.models.loader import ModelLoader
    from liteserve.engine.inference import InferenceEngine
    from liteserve.scheduler.scheduler import Scheduler
    from liteserve.engine.kv_cache import PagedKVCache
    from liteserve.config import SchedulerConfig

    loader = ModelLoader()
    loaded = loader.load_model(name="bench-model", model_path=args.model, is_default=True)
    engine = InferenceEngine(model=loaded, device=loaded.device)
    print(f"  Device: {loaded.device}, Memory: {loaded.memory_mb:.0f} MB\n")

    prompts = (PROMPTS * ((args.num_requests // len(PROMPTS)) + 1))[:args.num_requests]

    # Run sequential baseline
    print(f"Running sequential baseline ({args.num_requests} requests, {args.max_tokens} tokens each)...")
    seq_result = run_sequential(engine, prompts, args.max_tokens)
    print(f"  Throughput: {seq_result['throughput']:.1f} tokens/sec")
    print(f"  Time: {seq_result['elapsed_s']:.2f}s")
    print(f"  Total tokens: {seq_result['total_tokens']}")

    # Need fresh engine (past_kv state is cleared per request in sequential, but let's be safe)
    engine2 = InferenceEngine(model=loaded, device=loaded.device)

    # Run continuous batching
    sched_config = SchedulerConfig(max_batch_size=args.num_requests)
    print(f"\nRunning continuous batching ({args.num_requests} requests, batch_size={args.num_requests})...")
    batch_result = run_batched(engine2, Scheduler, PagedKVCache, sched_config, prompts, args.max_tokens)
    print(f"  Throughput: {batch_result['throughput']:.1f} tokens/sec")
    print(f"  Time: {batch_result['elapsed_s']:.2f}s")
    print(f"  Total tokens: {batch_result['total_tokens']}")

    # Comparison
    speedup = batch_result["throughput"] / seq_result["throughput"] if seq_result["throughput"] > 0 else 0
    print(f"\n{'=' * 70}")
    print(f"RESULTS")
    print(f"{'=' * 70}")
    print(f"  Sequential throughput:   {seq_result['throughput']:>8.1f} tokens/sec")
    print(f"  Batched throughput:      {batch_result['throughput']:>8.1f} tokens/sec")
    print(f"  Speedup:                 {speedup:>8.2f}x")
    print(f"{'=' * 70}")

    # Per-request latency comparison
    seq_latencies = [r["latency"] for r in seq_result["results"] if r["latency"]]
    batch_latencies = [r["latency"] for r in batch_result["results"] if r["latency"]]

    if seq_latencies and batch_latencies:
        print(f"\n  Latency (seconds):")
        print(f"    Sequential  avg={statistics.mean(seq_latencies):.2f}  p50={sorted(seq_latencies)[len(seq_latencies)//2]:.2f}")
        print(f"    Batched     avg={statistics.mean(batch_latencies):.2f}  p50={sorted(batch_latencies)[len(batch_latencies)//2]:.2f}")

    seq_ttfts = [r["ttft"] for r in seq_result["results"] if r["ttft"]]
    batch_ttfts = [r["ttft"] for r in batch_result["results"] if r["ttft"]]

    if seq_ttfts and batch_ttfts:
        print(f"\n  TTFT (seconds):")
        print(f"    Sequential  avg={statistics.mean(seq_ttfts):.3f}")
        print(f"    Batched     avg={statistics.mean(batch_ttfts):.3f}")


if __name__ == "__main__":
    main()
