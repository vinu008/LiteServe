"""Dynamic Batch Scheduler with continuous batching.

Implements the core scheduling loop that groups pending requests into
batches, manages request lifecycles, and handles preemption when GPU
memory is under pressure.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Callable, Optional

from liteserve.config import SchedulerConfig
from liteserve.engine.kv_cache import PagedKVCache
from liteserve.engine.types import Batch, Priority, Request, RequestStatus

logger = logging.getLogger(__name__)


class Scheduler:
    """Continuous batch scheduler for LLM inference.

    The scheduler runs a tight loop on every generation step:
    1. Check pending queue for new requests
    2. Estimate memory required for each pending request
    3. Greedily add requests to the active batch while memory allows
    4. Preempt low-priority requests if needed for high-priority ones
    5. Return the formed batch for inference

    Args:
        config: Scheduler configuration.
        kv_cache: The paged KV-cache manager (used for memory budgeting).
    """

    def __init__(self, config: SchedulerConfig, kv_cache: PagedKVCache):
        self.config = config
        self.kv_cache = kv_cache

        # Request queues
        self.pending_queue: deque[Request] = deque()
        self.active_batch: Batch = Batch()
        self.preempted_queue: deque[Request] = deque()

        # Completion tracking
        self._completed_requests: dict[str, Request] = {}
        self._completion_callbacks: dict[str, Callable] = {}

        # Stats
        self.total_scheduled = 0
        self.total_completed = 0
        self.total_preempted = 0

        # Running flag
        self._running = False

    @property
    def pending_count(self) -> int:
        return len(self.pending_queue)

    @property
    def active_count(self) -> int:
        return self.active_batch.size

    @property
    def queue_depth(self) -> int:
        return self.pending_count + self.active_count

    def add_request(
        self,
        request: Request,
        on_complete: Optional[Callable] = None,
    ) -> None:
        """Add a new request to the pending queue.

        Args:
            request: The request to schedule.
            on_complete: Optional callback when the request completes.
        """
        request.status = RequestStatus.PENDING
        request.arrival_time = time.time()

        # High priority goes to front of queue
        if request.priority == Priority.HIGH:
            self.pending_queue.appendleft(request)
        else:
            self.pending_queue.append(request)

        if on_complete:
            self._completion_callbacks[request.request_id] = on_complete

        logger.debug(
            "Added request %s to pending queue (depth: %d)",
            request.request_id[:8],
            self.pending_count,
        )

    def schedule_step(self) -> Batch:
        """Form a batch for the next forward pass.

        This is the core scheduling algorithm:
        1. Try to swap in preempted requests first
        2. Add new pending requests that fit in memory
        3. Preempt if needed for high-priority requests
        4. Remove completed requests and free their memory

        Returns:
            The batch to execute.
        """
        # Step 1: Remove completed requests from active batch
        completed = self.active_batch.get_completed()
        for request in completed:
            self.active_batch.remove_request(request.request_id)
            self.kv_cache.free(request.request_id)
            self._completed_requests[request.request_id] = request
            self.total_completed += 1

            # Fire completion callback
            callback = self._completion_callbacks.pop(request.request_id, None)
            if callback:
                callback(request)

            logger.debug(
                "Request %s completed (%d tokens, %.2fs)",
                request.request_id[:8],
                request.num_generated,
                request.total_latency or 0,
            )

        # Step 2: Try to swap in preempted requests
        while self.preempted_queue:
            request = self.preempted_queue[0]
            if self.kv_cache.swap_in(request.request_id):
                self.preempted_queue.popleft()
                request.status = RequestStatus.GENERATING
                self.active_batch.add_request(request)
                logger.debug("Swapped in preempted request %s", request.request_id[:8])
            else:
                break  # No more GPU memory for swap-ins

        # Step 3: Add pending requests that fit in memory
        newly_added = []
        reserved_blocks = 0  # Track blocks reserved this step
        while self.pending_queue and self.active_batch.size < self.config.max_batch_size:
            request = self.pending_queue[0]

            # Estimate memory needed
            blocks_needed = self._estimate_blocks_needed(request)
            available_blocks = self.kv_cache.num_free_blocks - reserved_blocks
            if blocks_needed <= available_blocks:
                self.pending_queue.popleft()
                request.status = RequestStatus.PREFILL
                self.active_batch.add_request(request)
                newly_added.append(request)
                reserved_blocks += blocks_needed
                self.total_scheduled += 1
            else:
                # Try preemption for high-priority requests
                if (
                    request.priority == Priority.HIGH
                    and self.config.preemption_policy in ("fcfs", "priority")
                    and self.active_batch.size > 0
                ):
                    if self._try_preempt(blocks_needed):
                        continue  # Retry adding after preemption
                break  # No memory available

        # Step 4: Force-add requests that have been waiting too long
        now = time.time()
        max_wait = self.config.max_waiting_time_ms / 1000.0
        timed_out = [
            r
            for r in self.pending_queue
            if (now - r.arrival_time) > max_wait
        ]
        for request in timed_out:
            if self.active_batch.size >= self.config.max_batch_size:
                break
            blocks_needed = self._estimate_blocks_needed(request)
            if self.kv_cache.can_allocate(blocks_needed * self.kv_cache.block_size):
                self.pending_queue.remove(request)
                request.status = RequestStatus.PREFILL
                self.active_batch.add_request(request)
                self.total_scheduled += 1

        return self.active_batch

    def _estimate_blocks_needed(self, request: Request) -> int:
        """Estimate the number of KV-cache blocks a request will need."""
        # Conservative estimate: prompt length + expected generation length
        prompt_len = len(request.prompt_tokens)
        estimated_total = prompt_len + request.max_new_tokens
        blocks = (estimated_total + self.kv_cache.block_size - 1) // self.kv_cache.block_size
        return blocks

    def _try_preempt(self, blocks_needed: int) -> bool:
        """Try to preempt the lowest-priority active request to free memory.

        Args:
            blocks_needed: Number of blocks we need to free.

        Returns:
            True if preemption was successful.
        """
        if self.active_batch.is_empty:
            return False

        # Find the best candidate for preemption
        candidates = sorted(
            self.active_batch.requests,
            key=lambda r: (r.priority, -r.arrival_time),  # Lowest priority, oldest first
        )

        for candidate in candidates:
            if candidate.status == RequestStatus.PREFILL:
                continue  # Don't preempt requests in prefill

            # Swap out to CPU
            self.kv_cache.swap_out(candidate.request_id)
            self.active_batch.remove_request(candidate.request_id)
            candidate.status = RequestStatus.PREEMPTED
            self.preempted_queue.append(candidate)
            self.total_preempted += 1

            logger.info(
                "Preempted request %s (priority=%d, tokens=%d)",
                candidate.request_id[:8],
                candidate.priority,
                candidate.num_generated,
            )

            # Check if we freed enough
            if self.kv_cache.num_free_blocks >= blocks_needed:
                return True

        return self.kv_cache.num_free_blocks >= blocks_needed

    def abort_request(self, request_id: str) -> bool:
        """Abort a request and free its resources."""
        # Check active batch
        request = self.active_batch.remove_request(request_id)
        if request:
            self.kv_cache.free(request_id)
            request.status = RequestStatus.FAILED
            return True

        # Check pending queue
        for r in self.pending_queue:
            if r.request_id == request_id:
                self.pending_queue.remove(r)
                r.status = RequestStatus.FAILED
                return True

        # Check preempted queue
        for r in self.preempted_queue:
            if r.request_id == request_id:
                self.preempted_queue.remove(r)
                self.kv_cache.free(request_id)
                r.status = RequestStatus.FAILED
                return True

        return False

    def get_completed(self, request_id: str) -> Optional[Request]:
        """Get and remove a completed request."""
        return self._completed_requests.pop(request_id, None)

    def get_stats(self) -> dict:
        """Return scheduler statistics."""
        return {
            "pending_requests": self.pending_count,
            "active_requests": self.active_count,
            "preempted_requests": len(self.preempted_queue),
            "total_scheduled": self.total_scheduled,
            "total_completed": self.total_completed,
            "total_preempted": self.total_preempted,
            "batch_size": self.active_batch.size,
            "max_batch_size": self.config.max_batch_size,
        }
