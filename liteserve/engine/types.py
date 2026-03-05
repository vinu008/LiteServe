"""Core data types for the LiteServe inference engine."""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


class RequestStatus(enum.Enum):
    PENDING = "pending"
    PREFILL = "prefill"
    GENERATING = "generating"
    COMPLETE = "complete"
    PREEMPTED = "preempted"
    FAILED = "failed"


class Priority(enum.IntEnum):
    NORMAL = 0
    HIGH = 1


@dataclass
class Request:
    """Represents a single inference request through its lifecycle."""

    prompt: str
    prompt_tokens: list[int] = field(default_factory=list)
    max_new_tokens: int = 256
    temperature: float = 0.7
    priority: Priority = Priority.NORMAL
    model_preference: str = "auto"  # "auto", "fp16", "int8", "int4"
    stream: bool = True

    # Internal state
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: RequestStatus = RequestStatus.PENDING
    kv_block_ids: list[int] = field(default_factory=list)
    generated_tokens: list[int] = field(default_factory=list)
    arrival_time: float = field(default_factory=time.time)
    first_token_time: Optional[float] = None
    completion_time: Optional[float] = None
    assigned_model: Optional[str] = None

    @property
    def num_generated(self) -> int:
        return len(self.generated_tokens)

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_tokens) + len(self.generated_tokens)

    @property
    def is_finished(self) -> bool:
        return self.status in (RequestStatus.COMPLETE, RequestStatus.FAILED)

    @property
    def time_to_first_token(self) -> Optional[float]:
        if self.first_token_time is not None:
            return self.first_token_time - self.arrival_time
        return None

    @property
    def total_latency(self) -> Optional[float]:
        if self.completion_time is not None:
            return self.completion_time - self.arrival_time
        return None


@dataclass
class Batch:
    """A batch of requests to be processed together in a single forward pass."""

    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    requests: list[Request] = field(default_factory=list)

    @property
    def prefill_requests(self) -> list[Request]:
        return [r for r in self.requests if r.status == RequestStatus.PREFILL]

    @property
    def generate_requests(self) -> list[Request]:
        return [r for r in self.requests if r.status == RequestStatus.GENERATING]

    @property
    def size(self) -> int:
        return len(self.requests)

    @property
    def is_empty(self) -> bool:
        return len(self.requests) == 0

    def add_request(self, request: Request) -> None:
        self.requests.append(request)

    def remove_request(self, request_id: str) -> Optional[Request]:
        for i, r in enumerate(self.requests):
            if r.request_id == request_id:
                return self.requests.pop(i)
        return None

    def get_completed(self) -> list[Request]:
        return [r for r in self.requests if r.is_finished]
