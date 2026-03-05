"""Tests for the Metrics Collector."""

from liteserve.metrics.collector import MetricsCollector


class TestMetricsCollector:
    def test_init_without_prometheus(self):
        # Should not crash even without prometheus_client
        collector = MetricsCollector(enabled=False)
        assert not collector.enabled

    def test_record_tokens(self):
        collector = MetricsCollector(enabled=False)
        collector.record_token_generated(10)
        assert collector._total_tokens_generated == 10

        collector.record_token_generated(5)
        assert collector._total_tokens_generated == 15

    def test_record_request_completed(self):
        collector = MetricsCollector(enabled=False)
        collector.record_request_completed(latency=1.5, tokens=100)
        assert collector._total_requests_completed == 1

    def test_record_request_failed(self):
        collector = MetricsCollector(enabled=False)
        collector.record_request_failed()
        assert collector._total_requests_failed == 1

    def test_record_ttft(self):
        collector = MetricsCollector(enabled=False)
        collector.record_ttft(0.25)
        collector.record_ttft(0.35)
        assert len(collector._ttft_samples) == 2

    def test_record_itl(self):
        collector = MetricsCollector(enabled=False)
        collector.record_itl(15.0)
        assert len(collector._itl_samples) == 1

    def test_get_summary(self):
        collector = MetricsCollector(enabled=False)
        collector.record_token_generated(100)
        collector.record_request_completed(1.0, 50)
        collector.record_ttft(0.3)

        summary = collector.get_summary()
        assert summary["total_tokens_generated"] == 100
        assert summary["total_requests_completed"] == 1
        assert summary["avg_ttft_seconds"] == 0.3
        assert "gpu" in summary
