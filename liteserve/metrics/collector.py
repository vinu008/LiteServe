"""Metrics collection and Prometheus exporter for LiteServe.

Tracks throughput, latency, GPU utilization, queue depth, KV-cache
efficiency, and routing decisions. Exposes metrics via a Prometheus
endpoint for scraping by Grafana.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Info,
        start_http_server,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and exposes LiteServe performance metrics.

    Metrics are exposed in Prometheus format and can be scraped by
    Prometheus and visualized in Grafana.
    """

    def __init__(self, port: int = 9090, enabled: bool = True):
        self.port = port
        self.enabled = enabled and PROMETHEUS_AVAILABLE

        if not PROMETHEUS_AVAILABLE and enabled:
            logger.warning(
                "prometheus_client not installed. Metrics collection disabled. "
                "Install with: pip install prometheus-client"
            )

        if self.enabled:
            self._init_metrics()

        # Internal tracking (always available even without Prometheus)
        self._start_time = time.time()
        self._total_tokens_generated = 0
        self._total_requests_completed = 0
        self._total_requests_failed = 0
        self._ttft_samples: list[float] = []
        self._itl_samples: list[float] = []

    def _init_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        # Throughput
        self.tokens_generated_total = Counter(
            "liteserve_tokens_generated_total",
            "Total number of tokens generated",
        )
        self.tokens_per_second = Gauge(
            "liteserve_tokens_per_second",
            "Current tokens per second throughput",
        )

        # Request metrics
        self.requests_total = Counter(
            "liteserve_requests_total",
            "Total number of requests received",
            ["status"],  # completed, failed, preempted
        )
        self.requests_in_progress = Gauge(
            "liteserve_requests_in_progress",
            "Number of requests currently being processed",
        )

        # Latency
        self.time_to_first_token = Histogram(
            "liteserve_time_to_first_token_seconds",
            "Time to first token (TTFT)",
            buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )
        self.inter_token_latency = Histogram(
            "liteserve_inter_token_latency_ms",
            "Inter-token latency (ITL) in milliseconds",
            buckets=[5, 10, 25, 50, 100, 250, 500],
        )
        self.request_latency = Histogram(
            "liteserve_request_latency_seconds",
            "End-to-end request latency",
            buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
        )

        # Batch metrics
        self.batch_size = Gauge(
            "liteserve_batch_size",
            "Current effective batch size",
        )

        # Queue metrics
        self.queue_depth = Gauge(
            "liteserve_queue_depth",
            "Number of requests in the pending queue",
        )

        # GPU metrics
        self.gpu_utilization = Gauge(
            "liteserve_gpu_utilization_percent",
            "GPU compute utilization percentage",
        )
        self.gpu_memory_used = Gauge(
            "liteserve_gpu_memory_used_gb",
            "GPU memory used in GB",
        )
        self.gpu_memory_total = Gauge(
            "liteserve_gpu_memory_total_gb",
            "GPU memory total in GB",
        )

        # KV-cache metrics
        self.kv_cache_utilization = Gauge(
            "liteserve_kv_cache_utilization",
            "KV-cache block utilization ratio",
        )
        self.kv_cache_blocks_used = Gauge(
            "liteserve_kv_cache_blocks_used",
            "Number of KV-cache blocks in use",
        )

        # Preemption
        self.preemptions_total = Counter(
            "liteserve_preemptions_total",
            "Total number of request preemptions",
        )

        # Router
        self.routing_decisions = Counter(
            "liteserve_routing_decisions_total",
            "Routing decisions by model variant",
            ["model"],
        )

        # System info
        self.system_info = Info(
            "liteserve",
            "LiteServe system information",
        )

    def start_server(self) -> None:
        """Start the Prometheus metrics HTTP server."""
        if self.enabled:
            start_http_server(self.port)
            logger.info("Prometheus metrics server started on port %d", self.port)

    def record_token_generated(self, count: int = 1) -> None:
        """Record that tokens were generated."""
        self._total_tokens_generated += count
        if self.enabled:
            self.tokens_generated_total.inc(count)

    def record_request_completed(self, latency: float, tokens: int) -> None:
        """Record a completed request."""
        self._total_requests_completed += 1
        if self.enabled:
            self.requests_total.labels(status="completed").inc()
            self.request_latency.observe(latency)

    def record_request_failed(self) -> None:
        """Record a failed request."""
        self._total_requests_failed += 1
        if self.enabled:
            self.requests_total.labels(status="failed").inc()

    def record_ttft(self, ttft: float) -> None:
        """Record time-to-first-token."""
        self._ttft_samples.append(ttft)
        if self.enabled:
            self.time_to_first_token.observe(ttft)

    def record_itl(self, itl_ms: float) -> None:
        """Record inter-token latency in ms."""
        self._itl_samples.append(itl_ms)
        if self.enabled:
            self.inter_token_latency.observe(itl_ms)

    def record_preemption(self) -> None:
        """Record a preemption event."""
        if self.enabled:
            self.preemptions_total.inc()

    def record_routing_decision(self, model_name: str) -> None:
        """Record a routing decision."""
        if self.enabled:
            self.routing_decisions.labels(model=model_name).inc()

    def update_batch_size(self, size: int) -> None:
        """Update current batch size gauge."""
        if self.enabled:
            self.batch_size.set(size)

    def update_queue_depth(self, depth: int) -> None:
        """Update current queue depth gauge."""
        if self.enabled:
            self.queue_depth.set(depth)

    def update_gpu_stats(
        self,
        utilization: Optional[float] = None,
        memory_used_gb: Optional[float] = None,
        memory_total_gb: Optional[float] = None,
    ) -> None:
        """Update GPU statistics."""
        if self.enabled:
            if utilization is not None:
                self.gpu_utilization.set(utilization)
            if memory_used_gb is not None:
                self.gpu_memory_used.set(memory_used_gb)
            if memory_total_gb is not None:
                self.gpu_memory_total.set(memory_total_gb)

    def update_kv_cache_stats(self, utilization: float, blocks_used: int) -> None:
        """Update KV-cache statistics."""
        if self.enabled:
            self.kv_cache_utilization.set(utilization)
            self.kv_cache_blocks_used.set(blocks_used)

    def update_requests_in_progress(self, count: int) -> None:
        """Update number of in-progress requests."""
        if self.enabled:
            self.requests_in_progress.set(count)

    def update_throughput(self, tokens_per_sec: float) -> None:
        """Update the throughput gauge."""
        if self.enabled:
            self.tokens_per_second.set(tokens_per_sec)

    def get_gpu_stats(self) -> dict:
        """Gather current GPU stats using pynvml or torch.cuda."""
        stats = {
            "gpu_utilization": 0.0,
            "memory_used_gb": 0.0,
            "memory_total_gb": 0.0,
            "memory_utilization": 0.0,
        }

        try:
            import torch

            if torch.cuda.is_available():
                mem_used = torch.cuda.memory_allocated() / (1024**3)
                mem_total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
                stats["memory_used_gb"] = mem_used
                stats["memory_total_gb"] = mem_total
                stats["memory_utilization"] = mem_used / mem_total if mem_total > 0 else 0.0
        except Exception:
            pass

        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            stats["gpu_utilization"] = util.gpu / 100.0
            pynvml.nvmlShutdown()
        except Exception:
            pass

        return stats

    def get_summary(self) -> dict:
        """Return a summary of all tracked metrics."""
        uptime = time.time() - self._start_time
        avg_throughput = self._total_tokens_generated / uptime if uptime > 0 else 0

        avg_ttft = (
            sum(self._ttft_samples) / len(self._ttft_samples)
            if self._ttft_samples
            else 0
        )
        avg_itl = (
            sum(self._itl_samples) / len(self._itl_samples)
            if self._itl_samples
            else 0
        )

        return {
            "uptime_seconds": uptime,
            "total_tokens_generated": self._total_tokens_generated,
            "total_requests_completed": self._total_requests_completed,
            "total_requests_failed": self._total_requests_failed,
            "avg_throughput_tokens_per_sec": avg_throughput,
            "avg_ttft_seconds": avg_ttft,
            "avg_itl_ms": avg_itl,
            "gpu": self.get_gpu_stats(),
        }
