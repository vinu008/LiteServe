"""Tests for the Dynamic Batch Scheduler."""

import torch

from liteserve.config import SchedulerConfig
from liteserve.engine.kv_cache import PagedKVCache
from liteserve.engine.types import Priority, Request, RequestStatus
from liteserve.scheduler.scheduler import Scheduler


def make_scheduler(max_batch_size: int = 4, num_blocks: int = 32) -> Scheduler:
    """Create a scheduler with a small KV-cache for testing."""
    config = SchedulerConfig(
        max_batch_size=max_batch_size,
        max_waiting_time_ms=100,
        memory_budget_pct=0.9,
        preemption_policy="fcfs",
    )
    kv_cache = PagedKVCache(
        num_layers=2,
        num_heads=4,
        head_dim=8,
        block_size=4,
        num_blocks=num_blocks,
        dtype=torch.float32,
        device="cpu",
    )
    return Scheduler(config, kv_cache)


def make_request(prompt_len: int = 10, max_tokens: int = 20, priority: Priority = Priority.NORMAL) -> Request:
    """Create a test request with fake prompt tokens."""
    req = Request(prompt="test", max_new_tokens=max_tokens, priority=priority)
    req.prompt_tokens = list(range(prompt_len))
    return req


class TestScheduler:
    def test_add_request(self):
        scheduler = make_scheduler()
        req = make_request()
        scheduler.add_request(req)
        assert scheduler.pending_count == 1
        assert req.status == RequestStatus.PENDING

    def test_schedule_step_adds_to_batch(self):
        scheduler = make_scheduler()
        req = make_request()
        scheduler.add_request(req)

        batch = scheduler.schedule_step()
        assert batch.size == 1
        assert req.status == RequestStatus.PREFILL
        assert scheduler.pending_count == 0

    def test_max_batch_size_respected(self):
        scheduler = make_scheduler(max_batch_size=2)

        for _ in range(5):
            scheduler.add_request(make_request(prompt_len=4, max_tokens=4))

        batch = scheduler.schedule_step()
        assert batch.size <= 2

    def test_completed_requests_removed(self):
        scheduler = make_scheduler()
        req = make_request()
        scheduler.add_request(req)

        batch = scheduler.schedule_step()
        assert batch.size == 1

        # Simulate completion
        req.status = RequestStatus.COMPLETE
        batch = scheduler.schedule_step()

        # Completed request should be removed
        assert scheduler.total_completed == 1

    def test_priority_ordering(self):
        scheduler = make_scheduler()

        low = make_request(priority=Priority.NORMAL)
        high = make_request(priority=Priority.HIGH)

        scheduler.add_request(low)
        scheduler.add_request(high)

        # High priority should be at front of queue
        assert scheduler.pending_queue[0] is high

    def test_abort_pending_request(self):
        scheduler = make_scheduler()
        req = make_request()
        scheduler.add_request(req)

        result = scheduler.abort_request(req.request_id)
        assert result is True
        assert scheduler.pending_count == 0
        assert req.status == RequestStatus.FAILED

    def test_abort_active_request(self):
        scheduler = make_scheduler()
        req = make_request()
        scheduler.add_request(req)
        scheduler.schedule_step()  # Move to active batch

        result = scheduler.abort_request(req.request_id)
        assert result is True
        assert req.status == RequestStatus.FAILED

    def test_abort_nonexistent(self):
        scheduler = make_scheduler()
        result = scheduler.abort_request("nonexistent")
        assert result is False

    def test_stats(self):
        scheduler = make_scheduler()
        req = make_request()
        scheduler.add_request(req)

        stats = scheduler.get_stats()
        assert stats["pending_requests"] == 1
        assert stats["active_requests"] == 0

        scheduler.schedule_step()
        stats = scheduler.get_stats()
        assert stats["pending_requests"] == 0
        assert stats["active_requests"] == 1
        assert stats["total_scheduled"] == 1

    def test_completion_callback(self):
        scheduler = make_scheduler()
        req = make_request()
        callback_called = [False]

        def on_complete(r):
            callback_called[0] = True

        scheduler.add_request(req, on_complete=on_complete)
        scheduler.schedule_step()

        # Mark complete and run another step
        req.status = RequestStatus.COMPLETE
        scheduler.schedule_step()

        assert callback_called[0] is True

    def test_memory_limited_scheduling(self):
        # Very small cache: only 4 blocks
        scheduler = make_scheduler(max_batch_size=10, num_blocks=4)

        # Each request needs ~8 blocks (prompt 10 + max_tokens 20 = 30 tokens / 4 = 8 blocks)
        r1 = make_request(prompt_len=10, max_tokens=6)  # needs 4 blocks
        r2 = make_request(prompt_len=10, max_tokens=6)  # needs 4 blocks

        scheduler.add_request(r1)
        scheduler.add_request(r2)

        batch = scheduler.schedule_step()
        # Only one should fit (4 blocks available, each needs 4)
        assert batch.size == 1
        assert scheduler.pending_count == 1
