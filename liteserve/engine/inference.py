"""Inference engine for running transformer forward passes.

Supports both sequential (single-request) and batched inference.
Tracks per-request past_key_values for autoregressive generation
across multiple calls from the scheduler loop.
"""

from __future__ import annotations

import logging
import time
from typing import Iterator, Optional

import torch

from liteserve.engine.types import Batch, Request, RequestStatus
from liteserve.models.loader import LoadedModel

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Runs transformer forward passes for LLM generation.

    Manages per-request KV-cache (past_key_values) so the scheduler
    can call prefill() and decode_step() across separate loop iterations
    without losing state.
    """

    def __init__(
        self,
        model: LoadedModel,
        device: Optional[str] = None,
    ):
        self.model = model
        self.device = device or model.device

        # Per-request past_key_values storage (HuggingFace native cache)
        self._past_kv: dict[str, tuple] = {}

        # EOS token ID for stopping generation
        self.eos_token_id = model.tokenizer.eos_token_id
        if self.eos_token_id is None:
            self.eos_token_id = model.tokenizer.convert_tokens_to_ids("</s>")

    def tokenize(self, prompt: str) -> list[int]:
        """Tokenize a prompt string."""
        return self.model.tokenizer.encode(prompt, add_special_tokens=True)

    def decode_token(self, token_id: int) -> str:
        """Decode a single token ID to string."""
        return self.model.tokenizer.decode([token_id], skip_special_tokens=True)

    def decode_tokens(self, token_ids: list[int]) -> str:
        """Decode a list of token IDs to string."""
        return self.model.tokenizer.decode(token_ids, skip_special_tokens=True)

    @torch.inference_mode()
    def prefill(self, request: Request) -> int:
        """Run prefill phase: process the full prompt and return the first token.

        Stores past_key_values internally so subsequent decode_step() calls
        can continue generation.
        """
        request.status = RequestStatus.PREFILL

        input_ids = torch.tensor(
            [request.prompt_tokens], dtype=torch.long, device=self.device
        )

        outputs = self.model.model(input_ids=input_ids, use_cache=True)

        # Store past_key_values for this request
        self._past_kv[request.request_id] = outputs.past_key_values

        # Sample first token
        logits = outputs.logits[:, -1, :]
        next_token = self._sample(logits, request.temperature)

        request.generated_tokens.append(next_token)
        request.first_token_time = time.time()
        request.status = RequestStatus.GENERATING

        return next_token

    @torch.inference_mode()
    def decode_step(self, request: Request) -> int:
        """Run one decode step using stored past_key_values.

        Returns the generated token ID.
        """
        last_token = request.generated_tokens[-1]
        input_ids = torch.tensor(
            [[last_token]], dtype=torch.long, device=self.device
        )

        past_kv = self._past_kv.get(request.request_id)

        outputs = self.model.model(
            input_ids=input_ids,
            past_key_values=past_kv,
            use_cache=True,
        )

        # Update stored past_key_values
        self._past_kv[request.request_id] = outputs.past_key_values

        # Sample next token
        logits = outputs.logits[:, -1, :]
        next_token = self._sample(logits, request.temperature)

        request.generated_tokens.append(next_token)

        # Check for completion
        if next_token == self.eos_token_id or request.num_generated >= request.max_new_tokens:
            request.status = RequestStatus.COMPLETE
            request.completion_time = time.time()
            # Free cached KV for this request
            self._past_kv.pop(request.request_id, None)

        return next_token

    def free_request(self, request_id: str) -> None:
        """Free stored past_key_values for a request."""
        self._past_kv.pop(request_id, None)

    @torch.inference_mode()
    def generate_sequential(self, request: Request) -> Iterator[int]:
        """Generate tokens sequentially for a single request (baseline).

        This is the naive, non-batched implementation used for baseline
        benchmarking. Yields token IDs one at a time.
        """
        if not request.prompt_tokens:
            request.prompt_tokens = self.tokenize(request.prompt)

        input_ids = torch.tensor(
            [request.prompt_tokens], dtype=torch.long, device=self.device
        )

        # Prefill
        request.status = RequestStatus.PREFILL
        outputs = self.model.model(input_ids=input_ids, use_cache=True)
        past_key_values = outputs.past_key_values

        # First token
        logits = outputs.logits[:, -1, :]
        next_token = self._sample(logits, request.temperature)
        request.generated_tokens.append(next_token)
        request.first_token_time = time.time()
        request.status = RequestStatus.GENERATING

        yield next_token

        # Decode loop
        while not request.is_finished:
            if next_token == self.eos_token_id:
                request.status = RequestStatus.COMPLETE
                request.completion_time = time.time()
                break

            if request.num_generated >= request.max_new_tokens:
                request.status = RequestStatus.COMPLETE
                request.completion_time = time.time()
                break

            input_ids = torch.tensor(
                [[next_token]], dtype=torch.long, device=self.device
            )
            outputs = self.model.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values

            logits = outputs.logits[:, -1, :]
            next_token = self._sample(logits, request.temperature)
            request.generated_tokens.append(next_token)

            yield next_token

        if request.completion_time is None:
            request.completion_time = time.time()

    def _sample(self, logits: torch.Tensor, temperature: float = 0.7) -> int:
        """Sample a token from logits with temperature scaling."""
        if temperature <= 0:
            return logits.argmax(dim=-1).item()

        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze().item()

    def get_stats(self) -> dict:
        """Return engine statistics."""
        return {
            "active_kv_caches": len(self._past_kv),
            "model": self.model.name,
            "device": self.device,
        }
