#!/usr/bin/env python3
"""End-to-end validation: loads a model and generates text without the server.

This proves the inference pipeline works. Run it first before starting the
full server.

Usage:
    python scripts/validate.py
    python scripts/validate.py --model mistralai/Mistral-7B-v0.1
    python scripts/validate.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max-tokens 50
"""

import argparse
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("validate")


def main():
    parser = argparse.ArgumentParser(description="Validate LiteServe inference pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model path",
    )
    parser.add_argument("--prompt", type=str, default="Explain what a neural network is in one paragraph.")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    print("=" * 60)
    print("LiteServe End-to-End Validation")
    print("=" * 60)

    # Step 1: Load model
    print(f"\n[1/4] Loading model: {args.model}")
    t0 = time.time()

    from liteserve.models.loader import ModelLoader
    loader = ModelLoader()
    loaded = loader.load_model(
        name="test-model",
        model_path=args.model,
        quantization=None,
        is_default=True,
    )

    load_time = time.time() - t0
    print(f"  Device: {loaded.device}")
    print(f"  Dtype: {loaded.dtype}")
    print(f"  Layers: {loaded.num_layers}, Heads: {loaded.num_heads}, KV Heads: {loaded.num_kv_heads}")
    print(f"  Head dim: {loaded.head_dim}, Vocab: {loaded.vocab_size}")
    print(f"  Memory: {loaded.memory_mb:.1f} MB")
    print(f"  Load time: {load_time:.1f}s")

    # Step 2: Create engine and tokenize
    print(f"\n[2/4] Creating inference engine")
    from liteserve.engine.inference import InferenceEngine
    from liteserve.engine.types import Request

    engine = InferenceEngine(model=loaded, device=loaded.device)

    request = Request(
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    request.prompt_tokens = engine.tokenize(args.prompt)
    print(f"  Prompt: \"{args.prompt}\"")
    print(f"  Prompt tokens: {len(request.prompt_tokens)}")

    # Step 3: Sequential generation (baseline)
    print(f"\n[3/4] Generating {args.max_tokens} tokens (sequential baseline)...")
    t0 = time.time()
    token_times = []

    for i, token_id in enumerate(engine.generate_sequential(request)):
        token_text = engine.decode_token(token_id)
        now = time.time()
        if i == 0:
            ttft = now - t0
            print(f"  TTFT: {ttft*1000:.1f} ms")
        token_times.append(now)
        # Print token without newline
        print(token_text, end="", flush=True)

    total_time = time.time() - t0
    num_tokens = request.num_generated
    print()  # newline after generated text

    # Step 4: Report metrics
    print(f"\n[4/4] Results")
    print(f"  {'─' * 40}")
    print(f"  Tokens generated: {num_tokens}")
    print(f"  Total time: {total_time:.2f}s")
    throughput = num_tokens / total_time if total_time > 0 else 0
    print(f"  Throughput: {throughput:.1f} tokens/sec")

    if request.time_to_first_token is not None:
        print(f"  TTFT: {request.time_to_first_token*1000:.1f} ms")
    if request.total_latency is not None:
        print(f"  Total latency: {request.total_latency*1000:.1f} ms")

    # Inter-token latency
    if len(token_times) > 1:
        itls = [(token_times[i] - token_times[i-1]) * 1000 for i in range(1, len(token_times))]
        avg_itl = sum(itls) / len(itls)
        p50 = sorted(itls)[len(itls) // 2]
        p95 = sorted(itls)[int(len(itls) * 0.95)]
        print(f"  ITL avg: {avg_itl:.1f} ms")
        print(f"  ITL p50: {p50:.1f} ms")
        print(f"  ITL p95: {p95:.1f} ms")

    print(f"  {'─' * 40}")
    print(f"\n  VALIDATION PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
