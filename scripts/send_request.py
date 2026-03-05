#!/usr/bin/env python3
"""Quick client to test the running LiteServe server.

Usage:
    python scripts/send_request.py "What is machine learning?"
    python scripts/send_request.py "Explain gravity" --max-tokens 100 --no-stream
"""

import argparse
import json
import sys
import time

import httpx


def stream_request(url, prompt, max_tokens, temperature):
    """Send a streaming request and print tokens as they arrive."""
    print(f"Prompt: {prompt}")
    print(f"Streaming response:\n")

    start = time.perf_counter()
    first_token_time = None
    tokens = 0

    with httpx.Client() as client:
        with client.stream(
            "POST",
            f"{url}/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            },
            timeout=60.0,
        ) as response:
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])

                if first_token_time is None:
                    first_token_time = time.perf_counter()

                token = data.get("token", "")
                print(token, end="", flush=True)
                tokens += 1

                if data.get("finish_reason") is not None:
                    usage = data.get("usage", {})
                    print(f"\n\n--- Done ---")
                    print(f"Finish reason: {data['finish_reason']}")
                    print(f"Prompt tokens: {usage.get('prompt_tokens', '?')}")
                    print(f"Completion tokens: {usage.get('completion_tokens', '?')}")
                    break

    elapsed = time.perf_counter() - start
    ttft = (first_token_time - start) if first_token_time else 0
    throughput = tokens / elapsed if elapsed > 0 else 0
    print(f"TTFT: {ttft*1000:.0f}ms | Total: {elapsed:.2f}s | Throughput: {throughput:.1f} tok/s")


def non_stream_request(url, prompt, max_tokens, temperature):
    """Send a non-streaming request."""
    print(f"Prompt: {prompt}")
    print(f"Waiting for response...\n")

    start = time.perf_counter()
    with httpx.Client() as client:
        response = client.post(
            f"{url}/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            },
            timeout=60.0,
        )

    elapsed = time.perf_counter() - start
    data = response.json()

    print(f"Response:\n{data.get('text', data)}")
    print(f"\nModel: {data.get('model', '?')}")
    print(f"Usage: {data.get('usage', {})}")
    print(f"Finish: {data.get('finish_reason', '?')}")
    print(f"Latency: {elapsed:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Send a request to LiteServe")
    parser.add_argument("prompt", type=str, nargs="?", default="Explain what a neural network is in two sentences.")
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-stream", action="store_true")
    args = parser.parse_args()

    # Check server is up
    try:
        r = httpx.get(f"{args.url}/v1/health", timeout=5)
        health = r.json()
        print(f"Server: {health.get('status', 'unknown')} | Models: {[m['name'] for m in health.get('models', [])]}\n")
    except Exception as e:
        print(f"ERROR: Server not reachable at {args.url}: {e}")
        print(f"Start it with: python -m liteserve.server.app --config configs/local.yaml")
        sys.exit(1)

    if args.no_stream:
        non_stream_request(args.url, args.prompt, args.max_tokens, args.temperature)
    else:
        stream_request(args.url, args.prompt, args.max_tokens, args.temperature)


if __name__ == "__main__":
    main()
