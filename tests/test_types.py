"""Tests for core data types."""

import time

from liteserve.engine.types import Batch, Priority, Request, RequestStatus


class TestRequest:
    def test_default_values(self):
        req = Request(prompt="Hello")
        assert req.prompt == "Hello"
        assert req.max_new_tokens == 256
        assert req.temperature == 0.7
        assert req.priority == Priority.NORMAL
        assert req.status == RequestStatus.PENDING
        assert req.num_generated == 0
        assert not req.is_finished

    def test_generated_tokens(self):
        req = Request(prompt="Hello")
        req.generated_tokens = [1, 2, 3]
        assert req.num_generated == 3
        assert req.total_tokens == len(req.prompt_tokens) + 3

    def test_is_finished(self):
        req = Request(prompt="Hello")
        assert not req.is_finished

        req.status = RequestStatus.COMPLETE
        assert req.is_finished

        req.status = RequestStatus.FAILED
        assert req.is_finished

        req.status = RequestStatus.GENERATING
        assert not req.is_finished

    def test_ttft(self):
        req = Request(prompt="Hello")
        req.arrival_time = 100.0
        assert req.time_to_first_token is None

        req.first_token_time = 100.5
        assert req.time_to_first_token == 0.5

    def test_total_latency(self):
        req = Request(prompt="Hello")
        req.arrival_time = 100.0
        assert req.total_latency is None

        req.completion_time = 102.0
        assert req.total_latency == 2.0

    def test_unique_ids(self):
        r1 = Request(prompt="Hello")
        r2 = Request(prompt="Hello")
        assert r1.request_id != r2.request_id


class TestBatch:
    def test_empty_batch(self):
        batch = Batch()
        assert batch.is_empty
        assert batch.size == 0

    def test_add_request(self):
        batch = Batch()
        req = Request(prompt="Hello")
        batch.add_request(req)
        assert batch.size == 1
        assert not batch.is_empty

    def test_remove_request(self):
        batch = Batch()
        req = Request(prompt="Hello")
        batch.add_request(req)
        removed = batch.remove_request(req.request_id)
        assert removed is req
        assert batch.is_empty

    def test_remove_nonexistent(self):
        batch = Batch()
        result = batch.remove_request("nonexistent")
        assert result is None

    def test_prefill_and_generate_separation(self):
        batch = Batch()

        r1 = Request(prompt="Hello")
        r1.status = RequestStatus.PREFILL
        batch.add_request(r1)

        r2 = Request(prompt="World")
        r2.status = RequestStatus.GENERATING
        batch.add_request(r2)

        assert len(batch.prefill_requests) == 1
        assert len(batch.generate_requests) == 1
        assert batch.prefill_requests[0] is r1
        assert batch.generate_requests[0] is r2

    def test_get_completed(self):
        batch = Batch()

        r1 = Request(prompt="Hello")
        r1.status = RequestStatus.COMPLETE
        batch.add_request(r1)

        r2 = Request(prompt="World")
        r2.status = RequestStatus.GENERATING
        batch.add_request(r2)

        completed = batch.get_completed()
        assert len(completed) == 1
        assert completed[0] is r1
