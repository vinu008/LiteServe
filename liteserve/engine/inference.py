"""Inference engine for running transformer forward passes.

Supports single-request prefill and **batched** decode. The decode phase
dominates total compute over a long generation (one forward pass per
generated token), so batching all in-flight decode steps into a single
padded forward pass is where continuous batching earns its throughput.

KV state is kept per request as a list of per-layer ``(keys, values)``
tensors (each ``[1, kv_heads, seq_len, head_dim]``). On every decode step
the active requests' caches are left-padded to a common length, stacked
along the batch dimension, run through one forward pass, then split back
out. Left padding plus an attention mask and per-request ``position_ids``
makes the batched result bit-for-bit equivalent to decoding each request
on its own (verified greedily against the sequential path).
"""

from __future__ import annotations

import logging
import time
from typing import Iterator, Optional

import torch
from transformers.cache_utils import DynamicCache

from liteserve.engine.types import Request, RequestStatus
from liteserve.models.loader import LoadedModel

logger = logging.getLogger(__name__)

# Per-layer KV: a (keys, values) pair, each shaped [batch, kv_heads, seq, head_dim].
LayerKV = tuple[torch.Tensor, torch.Tensor]


class InferenceEngine:
    """Runs transformer forward passes for LLM generation.

    Manages per-request KV-cache so the scheduler can call prefill() and
    decode_batch() across separate loop iterations without losing state.
    """

    def __init__(
        self,
        model: LoadedModel,
        device: Optional[str] = None,
    ):
        self.model = model
        self.device = device or model.device

        # request_id -> list of per-layer (keys, values), each [1, kv_heads, L, head_dim]
        self._past_kv: dict[str, list[LayerKV]] = {}
        # request_id -> current cached sequence length (L)
        self._seq_len: dict[str, int] = {}

        # Count of batched decode passes (for stats / showing real batching)
        self._decode_passes = 0
        self._decode_tokens = 0

        # EOS token ID for stopping generation
        self.eos_token_id = model.tokenizer.eos_token_id
        if self.eos_token_id is None:
            self.eos_token_id = model.tokenizer.convert_tokens_to_ids("</s>")

    # ── Tokenization helpers ─────────────────────────────────────────────

    def tokenize(self, prompt: str) -> list[int]:
        """Tokenize a prompt string."""
        return self.model.tokenizer.encode(prompt, add_special_tokens=True)

    def decode_token(self, token_id: int) -> str:
        """Decode a single token ID to string."""
        return self.model.tokenizer.decode([token_id], skip_special_tokens=True)

    def decode_tokens(self, token_ids: list[int]) -> str:
        """Decode a list of token IDs to string."""
        return self.model.tokenizer.decode(token_ids, skip_special_tokens=True)

    # ── Cache (un)packing ────────────────────────────────────────────────

    @staticmethod
    def _extract_kv(cache: DynamicCache) -> list[LayerKV]:
        """Pull per-layer (keys, values) tensors out of a DynamicCache."""
        return [(layer.keys, layer.values) for layer in cache.layers]

    # ── Prefill ──────────────────────────────────────────────────────────

    @torch.inference_mode()
    def prefill(self, request: Request) -> int:
        """Run prefill: process the full prompt and emit the first token.

        Kept single-request: prefill is one forward pass per request, and
        the per-request KV it produces is what the batched decoder consumes.
        """
        request.status = RequestStatus.PREFILL

        input_ids = torch.tensor(
            [request.prompt_tokens], dtype=torch.long, device=self.device
        )

        outputs = self.model.model(input_ids=input_ids, use_cache=True)

        self._past_kv[request.request_id] = self._extract_kv(outputs.past_key_values)
        self._seq_len[request.request_id] = input_ids.shape[1]

        next_token = self._sample(outputs.logits[:, -1, :], request.temperature)
        request.generated_tokens.append(next_token)
        request.first_token_time = time.time()
        request.status = RequestStatus.GENERATING

        if self._is_done(request, next_token):
            self._complete(request)

        return next_token

    # ── Batched decode ───────────────────────────────────────────────────

    @torch.inference_mode()
    def decode_batch(self, requests: list[Request]) -> None:
        """Run one decode step for every request in ``requests`` at once.

        All requests share a single padded forward pass. Each request's KV
        is left-padded to the batch's max length so a shorter sequence's real
        tokens sit flush against the new token, with the attention mask hiding
        the padding.
        """
        active = [
            r
            for r in requests
            if not r.is_finished and r.request_id in self._past_kv
        ]
        if not active:
            return

        lens = [self._seq_len[r.request_id] for r in active]
        max_len = max(lens)
        num_layers = len(self._past_kv[active[0].request_id])

        # Build the left-padded, batch-stacked cache.
        layers: list[LayerKV] = []
        for li in range(num_layers):
            keys, vals = [], []
            for r, seq_len in zip(active, lens):
                k, v = self._past_kv[r.request_id][li]  # [1, h, seq_len, d]
                pad = max_len - seq_len
                if pad > 0:
                    pad_shape = (1, k.shape[1], pad, k.shape[3])
                    k = torch.cat([k.new_zeros(pad_shape), k], dim=2)
                    v = torch.cat([v.new_zeros(pad_shape), v], dim=2)
                keys.append(k)
                vals.append(v)
            layers.append((torch.cat(keys, dim=0), torch.cat(vals, dim=0)))

        batched_cache = DynamicCache(ddp_cache_data=layers)

        # input: each request's most recent token.
        input_ids = torch.tensor(
            [[r.generated_tokens[-1]] for r in active],
            dtype=torch.long,
            device=self.device,
        )
        # attention mask: hide the left padding for short sequences.
        attn = torch.zeros(len(active), max_len + 1, dtype=torch.long, device=self.device)
        for i, seq_len in enumerate(lens):
            attn[i, max_len - seq_len:] = 1
        # position of the new token = the request's real (unpadded) length.
        position_ids = torch.tensor([[seq_len] for seq_len in lens], dtype=torch.long, device=self.device)
        cache_position = torch.tensor([max_len], dtype=torch.long, device=self.device)

        outputs = self.model.model(
            input_ids=input_ids,
            past_key_values=batched_cache,
            attention_mask=attn,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
        )

        self._decode_passes += 1
        new_layers = outputs.past_key_values.layers

        for i, request in enumerate(active):
            next_token = self._sample(outputs.logits[i : i + 1, -1, :], request.temperature)
            request.generated_tokens.append(next_token)
            self._decode_tokens += 1

            real_len = lens[i] + 1
            # Split this request's KV back out, dropping the left padding.
            new_cache = []
            for li in range(num_layers):
                k = new_layers[li].keys[i : i + 1, :, -real_len:, :].contiguous()
                v = new_layers[li].values[i : i + 1, :, -real_len:, :].contiguous()
                new_cache.append((k, v))
            self._past_kv[request.request_id] = new_cache
            self._seq_len[request.request_id] = real_len

            if self._is_done(request, next_token):
                self._complete(request)

    @torch.inference_mode()
    def decode_step(self, request: Request) -> int:
        """Decode a single request (convenience wrapper over decode_batch)."""
        self.decode_batch([request])
        return request.generated_tokens[-1]

    # ── Lifecycle helpers ────────────────────────────────────────────────

    def _is_done(self, request: Request, last_token: int) -> bool:
        return last_token == self.eos_token_id or request.num_generated >= request.max_new_tokens

    def _complete(self, request: Request) -> None:
        request.status = RequestStatus.COMPLETE
        request.completion_time = time.time()
        self.free_request(request.request_id)

    def free_request(self, request_id: str) -> None:
        """Free stored KV-cache for a request."""
        self._past_kv.pop(request_id, None)
        self._seq_len.pop(request_id, None)

    # ── Sequential baseline (for benchmarking) ───────────────────────────

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

        request.status = RequestStatus.PREFILL
        outputs = self.model.model(input_ids=input_ids, use_cache=True)
        past_key_values = outputs.past_key_values

        next_token = self._sample(outputs.logits[:, -1, :], request.temperature)
        request.generated_tokens.append(next_token)
        request.first_token_time = time.time()
        request.status = RequestStatus.GENERATING

        yield next_token

        while not request.is_finished:
            if next_token == self.eos_token_id or request.num_generated >= request.max_new_tokens:
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

            next_token = self._sample(outputs.logits[:, -1, :], request.temperature)
            request.generated_tokens.append(next_token)

            yield next_token

        if request.completion_time is None:
            request.completion_time = time.time()

    def _sample(self, logits: torch.Tensor, temperature: float = 0.7) -> int:
        """Sample a token from a single logits row with temperature scaling."""
        if temperature <= 0:
            return logits.argmax(dim=-1).item()

        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze().item()

    def get_stats(self) -> dict:
        """Return engine statistics."""
        avg_batch = self._decode_tokens / self._decode_passes if self._decode_passes else 0.0
        return {
            "active_kv_caches": len(self._past_kv),
            "decode_passes": self._decode_passes,
            "decode_tokens": self._decode_tokens,
            "avg_decode_batch_size": round(avg_batch, 2),
            "model": self.model.name,
            "device": self.device,
        }
