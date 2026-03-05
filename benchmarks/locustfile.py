"""Locust load testing for LiteServe.

Usage:
    locust -f benchmarks/locustfile.py --host http://localhost:8000
"""

import json
import random

from locust import HttpUser, between, task

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


class LiteServeUser(HttpUser):
    """Simulates a user sending inference requests."""

    wait_time = between(0.5, 2.0)

    @task(weight=8)
    def completion_streaming(self):
        """Send a streaming completion request."""
        prompt = random.choice(PROMPTS)
        with self.client.post(
            "/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": 128,
                "temperature": 0.7,
                "stream": True,
                "priority": "normal",
            },
            stream=True,
            catch_response=True,
            name="/v1/completions (stream)",
        ) as response:
            if response.status_code == 200:
                tokens = 0
                for line in response.iter_lines():
                    if line and line.startswith("data: "):
                        tokens += 1
                if tokens > 0:
                    response.success()
                else:
                    response.failure("No tokens received")
            else:
                response.failure(f"Status {response.status_code}")

    @task(weight=2)
    def completion_nonstreaming(self):
        """Send a non-streaming completion request."""
        prompt = random.choice(PROMPTS)
        with self.client.post(
            "/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": 64,
                "temperature": 0.7,
                "stream": False,
                "priority": "normal",
            },
            catch_response=True,
            name="/v1/completions (non-stream)",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "text" in data:
                    response.success()
                else:
                    response.failure("No text in response")
            else:
                response.failure(f"Status {response.status_code}")

    @task(weight=1)
    def health_check(self):
        """Check server health."""
        self.client.get("/v1/health", name="/v1/health")

    @task(weight=1)
    def get_metrics(self):
        """Fetch metrics."""
        self.client.get("/v1/metrics", name="/v1/metrics")
