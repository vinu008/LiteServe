"""Tests for the Multi-Model Router."""

from unittest.mock import MagicMock

import torch

from liteserve.config import RouterConfig
from liteserve.engine.types import Priority, Request
from liteserve.models.loader import LoadedModel, ModelLoader
from liteserve.router.router import Router


def make_mock_model(name: str, quantization=None) -> LoadedModel:
    """Create a mock LoadedModel."""
    return LoadedModel(
        name=name,
        model=MagicMock(),
        tokenizer=MagicMock(),
        quantization=quantization,
        max_model_len=2048,
        device="cpu",
        dtype=torch.float16,
    )


def make_router(strategy: str = "adaptive") -> tuple[Router, ModelLoader]:
    """Create a router with mock models."""
    loader = ModelLoader(device="cpu")

    # Register mock models
    fp16 = make_mock_model("llama-fp16", quantization=None)
    int8 = make_mock_model("llama-int8", quantization="int8")
    int4 = make_mock_model("llama-int4", quantization="int4-gptq")

    loader.models = {"llama-fp16": fp16, "llama-int8": int8, "llama-int4": int4}
    loader.default_model = "llama-fp16"

    config = RouterConfig(strategy=strategy)
    router = Router(config, loader)
    return router, loader


class TestRouter:
    def test_quality_first(self):
        router, _ = make_router(strategy="quality_first")
        req = Request(prompt="Hello")
        model = router.route(req)
        assert model.quantization is None  # Should pick FP16

    def test_throughput_first_under_load(self):
        router, _ = make_router(strategy="throughput_first")
        router.update_system_state(gpu_utilization=0.8, queue_depth=10)
        req = Request(prompt="Hello")
        model = router.route(req)
        assert model.quantization is not None  # Should pick quantized

    def test_throughput_first_idle(self):
        router, _ = make_router(strategy="throughput_first")
        router.update_system_state(gpu_utilization=0.1, queue_depth=0)
        req = Request(prompt="Hello")
        model = router.route(req)
        assert model.quantization is None  # Should pick FP16 when idle

    def test_priority_routing_high(self):
        router, _ = make_router(strategy="priority")
        req = Request(prompt="Hello", priority=Priority.HIGH)
        model = router.route(req)
        assert model.quantization is None  # High priority -> FP16

    def test_priority_routing_normal(self):
        router, _ = make_router(strategy="priority")
        req = Request(prompt="Hello", priority=Priority.NORMAL)
        model = router.route(req)
        assert model.quantization is not None  # Normal -> quantized

    def test_adaptive_high_load(self):
        router, _ = make_router(strategy="adaptive")
        router.update_system_state(gpu_utilization=0.9, queue_depth=25)
        req = Request(prompt="Hello")
        model = router.route(req)
        assert model.quantization is not None  # High load -> quantized

    def test_adaptive_low_load(self):
        router, _ = make_router(strategy="adaptive")
        router.update_system_state(gpu_utilization=0.2, queue_depth=2)
        req = Request(prompt="Hello")
        model = router.route(req)
        assert model.quantization is None  # Low load -> FP16

    def test_explicit_model_preference(self):
        router, _ = make_router()
        req = Request(prompt="Hello", model_preference="int8")
        model = router.route(req)
        assert model.quantization == "int8"

    def test_routing_stats(self):
        router, _ = make_router()
        router.update_system_state(gpu_utilization=0.2)
        req = Request(prompt="Hello")
        router.route(req)

        stats = router.get_stats()
        assert stats["strategy"] == "adaptive"
        assert sum(stats["routing_decisions"].values()) == 1

    def test_assigned_model_set(self):
        router, _ = make_router()
        router.update_system_state(gpu_utilization=0.2)
        req = Request(prompt="Hello")
        model = router.route(req)
        assert req.assigned_model == model.name
