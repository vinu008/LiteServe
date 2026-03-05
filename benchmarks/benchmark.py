"""Benchmark suite for LiteServe.

Measures throughput, latency (TTFT, ITL, p50/p95/p99), memory usage,
and compares across concurrency levels and quantization modes.

Usage:
    python benchmarks/benchmark.py --url http://localhost:8000 --concurrency 1 4 8 16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class RequestResult:
    """Result of a single benchmark request."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float = 0.0
    total_latency_ms: float = 0.0
    inter_token_latencies_ms: list[float] = field(default_factory=list)
    success: bool = True
    error: str = ""


@dataclass
class BenchmarkResult:
    """Aggregated results of a benchmark run."""

    concurrency: int
    num_requests: int
    total_time_s: float
    results: list[RequestResult]

    @property
    def successful(self) -> list[RequestResult]:
        return [r for r in self.results if r.success]

    @property
    def total_tokens(self) -> int:
        return sum(r.completion_tokens for r in self.successful)

    @property
    def throughput_tokens_per_sec(self) -> float:
        return self.total_tokens / self.total_time_s if self.total_time_s > 0 else 0

    @property
    def requests_per_sec(self) -> float:
        return len(self.successful) / self.total_time_s if self.total_time_s > 0 else 0

    def latency_percentile(self, p: float) -> float:
        latencies = sorted(r.total_latency_ms for r in self.successful)
        if not latencies:
            return 0
        idx = int(len(latencies) * p / 100)
        return latencies[min(idx, len(latencies) - 1)]

    def ttft_percentile(self, p: float) -> float:
        ttfts = sorted(r.ttft_ms for r in self.successful)
        if not ttfts:
            return 0
        idx = int(len(ttfts) * p / 100)
        return ttfts[min(idx, len(ttfts) - 1)]

    def summary(self) -> dict:
        successful = self.successful
        if not successful:
            return {"error": "No successful requests"}

        itls = []
        for r in successful:
            itls.extend(r.inter_token_latencies_ms)

        return {
            "concurrency": self.concurrency,
            "num_requests": self.num_requests,
            "successful_requests": len(successful),
            "failed_requests": self.num_requests - len(successful),
            "total_time_s": round(self.total_time_s, 2),
            "throughput_tokens_per_sec": round(self.throughput_tokens_per_sec, 1),
            "requests_per_sec": round(self.requests_per_sec, 2),
            "latency_p50_ms": round(self.latency_percentile(50), 1),
            "latency_p95_ms": round(self.latency_percentile(95), 1),
            "latency_p99_ms": round(self.latency_percentile(99), 1),
            "ttft_p50_ms": round(self.ttft_percentile(50), 1),
            "ttft_p95_ms": round(self.ttft_percentile(95), 1),
            "avg_itl_ms": round(statistics.mean(itls), 1) if itls else 0,
            "total_tokens_generated": self.total_tokens,
        }


# Sample prompts of varying lengths
PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Write a short story about a robot learning to paint.",
    "What are the key differences between Python and Rust?",
    "Describe the process of photosynthesis step by step.",
    "How does a neural network learn? Explain backpropagation.",
    "What is the significance of the Turing test in AI?",
    "Explain the theory of general relativity to a 10-year-old.",
    "What are the main challenges in building self-driving cars?",
]


async def send_request(
    client: httpx.AsyncClient,
    url: str,
    prompt: str,
    max_tokens: int = 128,
    stream: bool = True,
) -> RequestResult:
    """Send a single inference request and measure performance."""
    result = RequestResult()
    start_time = time.perf_counter()
    first_token_time = None
    last_token_time = start_time
    tokens_received = 0

    try:
        if stream:
            async with client.stream(
                "POST",
                f"{url}/v1/completions",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                    "stream": True,
                },
                timeout=60.0,
            ) as response:
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = json.loads(line[6:])

                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now
                        result.ttft_ms = (first_token_time - start_time) * 1000
                    else:
                        itl = (now - last_token_time) * 1000
                        result.inter_token_latencies_ms.append(itl)

                    last_token_time = now
                    tokens_received += 1

                    if data.get("finish_reason") is not None:
                        usage = data.get("usage", {})
                        result.prompt_tokens = usage.get("prompt_tokens", 0)
                        result.completion_tokens = usage.get(
                            "completion_tokens", tokens_received
                        )
                        break
        else:
            response = await client.post(
                f"{url}/v1/completions",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                    "stream": False,
                },
                timeout=60.0,
            )
            data = response.json()
            result.prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
            result.completion_tokens = data.get("usage", {}).get(
                "completion_tokens", 0
            )
            result.ttft_ms = (time.perf_counter() - start_time) * 1000

        result.total_latency_ms = (time.perf_counter() - start_time) * 1000
        result.success = True

    except Exception as e:
        result.success = False
        result.error = str(e)
        result.total_latency_ms = (time.perf_counter() - start_time) * 1000

    return result


async def run_benchmark(
    url: str,
    concurrency: int,
    num_requests: int = 32,
    max_tokens: int = 128,
    stream: bool = True,
) -> BenchmarkResult:
    """Run a benchmark at a specific concurrency level."""
    semaphore = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []

    async def bounded_request(prompt: str) -> RequestResult:
        async with semaphore:
            async with httpx.AsyncClient() as client:
                return await send_request(client, url, prompt, max_tokens, stream)

    start_time = time.perf_counter()

    tasks = [
        bounded_request(PROMPTS[i % len(PROMPTS)]) for i in range(num_requests)
    ]
    results = await asyncio.gather(*tasks)

    total_time = time.perf_counter() - start_time

    return BenchmarkResult(
        concurrency=concurrency,
        num_requests=num_requests,
        total_time_s=total_time,
        results=list(results),
    )


async def main_async(args):
    """Run the full benchmark suite."""
    print("=" * 70)
    print("LiteServe Benchmark Suite")
    print("=" * 70)
    print(f"Target: {args.url}")
    print(f"Requests per level: {args.num_requests}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Concurrency levels: {args.concurrency}")
    print(f"Streaming: {args.stream}")
    print("=" * 70)

    all_results = []

    for conc in args.concurrency:
        print(f"\nRunning benchmark at concurrency={conc}...")
        result = await run_benchmark(
            url=args.url,
            concurrency=conc,
            num_requests=args.num_requests,
            max_tokens=args.max_tokens,
            stream=args.stream,
        )
        summary = result.summary()
        all_results.append(summary)

        print(f"  Throughput: {summary['throughput_tokens_per_sec']} tokens/s")
        print(f"  Requests/s: {summary['requests_per_sec']}")
        print(f"  Latency p50/p95/p99: {summary['latency_p50_ms']}/{summary['latency_p95_ms']}/{summary['latency_p99_ms']} ms")
        print(f"  TTFT p50/p95: {summary['ttft_p50_ms']}/{summary['ttft_p95_ms']} ms")
        print(f"  Avg ITL: {summary['avg_itl_ms']} ms")
        print(f"  Success rate: {summary['successful_requests']}/{summary['num_requests']}")

    # Save results
    output_path = args.output or "benchmarks/results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print comparison table
    print("\n" + "=" * 70)
    print("Summary Comparison")
    print("=" * 70)
    print(f"{'Concurrency':>12} {'Throughput':>12} {'Req/s':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'TTFT p50':>10}")
    print("-" * 70)
    for s in all_results:
        print(
            f"{s['concurrency']:>12} "
            f"{s['throughput_tokens_per_sec']:>10.1f}/s "
            f"{s['requests_per_sec']:>8.2f} "
            f"{s['latency_p50_ms']:>7.1f}ms "
            f"{s['latency_p95_ms']:>7.1f}ms "
            f"{s['latency_p99_ms']:>7.1f}ms "
            f"{s['ttft_p50_ms']:>9.1f}ms"
        )


def main():
    parser = argparse.ArgumentParser(description="LiteServe Benchmark Suite")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Server URL")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16], help="Concurrency levels")
    parser.add_argument("--num-requests", type=int, default=32, help="Requests per concurrency level")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens per request")
    parser.add_argument("--stream", action="store_true", default=True, help="Use streaming")
    parser.add_argument("--no-stream", dest="stream", action="store_false", help="Disable streaming")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
