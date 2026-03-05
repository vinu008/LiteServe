#!/usr/bin/env python3
"""Evaluate perplexity across quantization levels.

Measures perplexity on WikiText-2 test set for FP16 vs INT8 vs INT4,
producing the accuracy/throughput tradeoff data needed for the write-up.

Usage:
    python scripts/evaluate_perplexity.py --model mistralai/Mistral-7B-v0.1
    python scripts/evaluate_perplexity.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import argparse
import json
import logging
import math
import time

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("perplexity")


def evaluate_perplexity(model, tokenizer, device, max_samples=50, max_length=512):
    """Evaluate perplexity on WikiText-2 test set."""
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Concatenate and chunk into blocks
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    encodings = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = encodings.input_ids[0]

    total_loss = 0.0
    total_tokens = 0
    num_chunks = 0
    stride = max_length

    for start in range(0, min(len(input_ids), max_samples * stride), stride):
        end = min(start + max_length, len(input_ids))
        chunk = input_ids[start:end].unsqueeze(0).to(device)

        if chunk.shape[1] < 2:
            continue

        with torch.inference_mode():
            outputs = model(input_ids=chunk, labels=chunk)
            loss = outputs.loss

        total_loss += loss.item() * (chunk.shape[1] - 1)
        total_tokens += chunk.shape[1] - 1
        num_chunks += 1

        if num_chunks >= max_samples:
            break

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(avg_loss)

    return {
        "perplexity": round(perplexity, 2),
        "avg_loss": round(avg_loss, 4),
        "num_chunks": num_chunks,
        "total_tokens": total_tokens,
    }


def benchmark_throughput(model, tokenizer, device, num_runs=5, prompt_len=128, gen_len=64):
    """Measure generation throughput."""
    prompt_ids = torch.randint(100, 10000, (1, prompt_len), device=device)

    # Warmup
    with torch.inference_mode():
        model.generate(prompt_ids, max_new_tokens=10, do_sample=False)

    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(prompt_ids, max_new_tokens=gen_len, do_sample=False)
        elapsed = time.perf_counter() - start
        tokens_generated = output.shape[1] - prompt_len
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    throughput = gen_len / avg_time

    return {
        "throughput_tokens_per_sec": round(throughput, 1),
        "avg_time_s": round(avg_time, 3),
        "gen_tokens": gen_len,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate perplexity across quantization levels")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--output", type=str, default="benchmarks/perplexity_results.json")
    args = parser.parse_args()

    from liteserve.models.loader import ModelLoader

    print("=" * 70)
    print("Perplexity Evaluation Across Quantization Levels")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Max samples: {args.max_samples}")

    results = []

    # Quantization configs to test
    configs = [
        ("FP16/FP32", None),
    ]

    # Only test INT8/INT4 on CUDA
    if torch.cuda.is_available():
        configs.append(("INT8", "int8"))
        configs.append(("INT4-NF4", "int4-bnb"))

    for label, quant in configs:
        print(f"\n{'─' * 70}")
        print(f"Evaluating: {label} (quantization={quant})")
        print(f"{'─' * 70}")

        loader = ModelLoader()
        try:
            loaded = loader.load_model(
                name=f"eval-{label}",
                model_path=args.model,
                quantization=quant,
                is_default=True,
            )
        except Exception as e:
            print(f"  SKIPPED: {e}")
            continue

        print(f"  Memory: {loaded.memory_mb:.0f} MB")
        print(f"  Device: {loaded.device}")

        # Perplexity
        print("  Computing perplexity...")
        try:
            ppl = evaluate_perplexity(
                loaded.model, loaded.tokenizer, loaded.device,
                max_samples=args.max_samples,
            )
            print(f"  Perplexity: {ppl['perplexity']}")
            print(f"  Avg loss: {ppl['avg_loss']}")
        except Exception as e:
            print(f"  Perplexity FAILED: {e}")
            ppl = {"perplexity": None, "avg_loss": None}

        # Throughput
        print("  Measuring throughput...")
        try:
            tp = benchmark_throughput(loaded.model, loaded.tokenizer, loaded.device)
            print(f"  Throughput: {tp['throughput_tokens_per_sec']} tokens/sec")
        except Exception as e:
            print(f"  Throughput FAILED: {e}")
            tp = {"throughput_tokens_per_sec": None}

        result = {
            "label": label,
            "quantization": quant,
            "memory_mb": round(loaded.memory_mb, 1),
            **ppl,
            **tp,
        }
        results.append(result)

        # Unload to free memory
        loader.unload_model(f"eval-{label}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Variant':<12} {'Memory MB':>10} {'Perplexity':>12} {'Throughput':>14} {'PPL Delta':>10}")
    print(f"{'─' * 58}")

    baseline_ppl = results[0]["perplexity"] if results and results[0]["perplexity"] else None
    for r in results:
        ppl = r["perplexity"] or "N/A"
        tp = r.get("throughput_tokens_per_sec") or "N/A"
        delta = ""
        if baseline_ppl and r["perplexity"] and r != results[0]:
            delta = f"+{((r['perplexity'] / baseline_ppl) - 1) * 100:.1f}%"
        print(f"{r['label']:<12} {r['memory_mb']:>10.0f} {str(ppl):>12} {str(tp):>12}/s {delta:>10}")

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
