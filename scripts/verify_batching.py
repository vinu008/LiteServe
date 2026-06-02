#!/usr/bin/env python3
"""Verify batched decode is (a) correct and (b) faster than sequential.

Correctness: greedy batched decode must produce identical tokens to decoding
each request on its own. Throughput: decoding N requests as one batch should
take far less time than N sequential decodes.

Usage:
    python scripts/verify_batching.py
    python scripts/verify_batching.py --model mistralai/Mistral-7B-v0.1 --requests 16
"""

import argparse
import logging
import time

logging.basicConfig(level=logging.WARNING)

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village near the",
    "Photosynthesis is the process by which",
    "To make a good cup of coffee, you first",
    "The three laws of motion were described by",
    "In the beginning of the universe, there was",
    "A neural network learns by adjusting its",
    "The most important meal of the day is",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=24)
    args = parser.parse_args()

    from liteserve.engine.inference import InferenceEngine
    from liteserve.engine.types import Request, RequestStatus
    from liteserve.models.loader import ModelLoader

    loader = ModelLoader()
    loaded = loader.load_model(name="verify", model_path=args.model, is_default=True)
    print(f"Model on {loaded.device}, {loaded.memory_mb:.0f} MB\n")

    prompts = (PROMPTS * (args.requests // len(PROMPTS) + 1))[: args.requests]

    def fresh_requests():
        engine = InferenceEngine(model=loaded, device=loaded.device)
        reqs = []
        for p in prompts:
            r = Request(prompt=p, max_new_tokens=args.tokens, temperature=0.0)  # greedy
            r.prompt_tokens = engine.tokenize(p)
            reqs.append(r)
        return engine, reqs

    # ── Sequential (per-request) reference ──
    engine, reqs = fresh_requests()
    t0 = time.perf_counter()
    for r in reqs:
        engine.prefill(r)
        while not r.is_finished:
            engine.decode_step(r)
    seq_time = time.perf_counter() - t0
    seq_tokens = [list(r.generated_tokens) for r in reqs]
    seq_total = sum(len(t) for t in seq_tokens)

    # ── Batched decode ──
    engine, reqs = fresh_requests()
    t0 = time.perf_counter()
    for r in reqs:
        engine.prefill(r)
    while any(not r.is_finished for r in reqs):
        engine.decode_batch([r for r in reqs if not r.is_finished])
    bat_time = time.perf_counter() - t0
    bat_tokens = [list(r.generated_tokens) for r in reqs]
    bat_total = sum(len(t) for t in bat_tokens)

    # ── Correctness ──
    mismatches = sum(1 for a, b in zip(seq_tokens, bat_tokens) if a != b)
    print("=" * 60)
    print(f"Correctness: {args.requests - mismatches}/{args.requests} requests "
          f"produced identical greedy tokens "
          f"({'PASS' if mismatches == 0 else 'FAIL'})")
    print("=" * 60)

    # ── Throughput ──
    seq_tps = seq_total / seq_time
    bat_tps = bat_total / bat_time
    print(f"  Sequential:  {seq_total:4d} tokens in {seq_time:6.2f}s  ->  {seq_tps:7.1f} tok/s")
    print(f"  Batched:     {bat_total:4d} tokens in {bat_time:6.2f}s  ->  {bat_tps:7.1f} tok/s")
    print(f"  Speedup:     {bat_tps / seq_tps:.2f}x")
    print("=" * 60)
    stats = engine.get_stats()
    print(f"  Decode passes: {stats['decode_passes']}, "
          f"avg batch size: {stats['avg_decode_batch_size']}")


if __name__ == "__main__":
    main()
