"""Multi-Model Router with adaptive routing strategies.

Routes incoming requests to the optimal model variant (FP16, INT8, INT4)
based on current system state — GPU utilization, queue depth, and
request priority.
"""

from __future__ import annotations

import logging
from typing import Optional

from liteserve.config import RouterConfig
from liteserve.engine.types import Priority, Request
from liteserve.models.loader import LoadedModel, ModelLoader

logger = logging.getLogger(__name__)


class SystemState:
    """Snapshot of the current system state for routing decisions."""

    def __init__(
        self,
        gpu_utilization: float = 0.0,
        gpu_memory_utilization: float = 0.0,
        queue_depth: int = 0,
        active_requests: int = 0,
    ):
        self.gpu_utilization = gpu_utilization
        self.gpu_memory_utilization = gpu_memory_utilization
        self.queue_depth = queue_depth
        self.active_requests = active_requests


class Router:
    """Routes requests to the best model variant based on system state.

    Supports four routing strategies:
    - quality_first: Always use FP16 unless OOM
    - throughput_first: Prefer INT4/INT8, fall back to FP16 if idle
    - adaptive: Dynamically switch based on GPU util and queue depth
    - priority: High-priority -> FP16, normal -> INT4

    Args:
        config: Router configuration.
        model_loader: The model loader with available model variants.
    """

    def __init__(self, config: RouterConfig, model_loader: ModelLoader):
        self.config = config
        self.model_loader = model_loader
        self._system_state = SystemState()

        # Routing stats
        self.routing_decisions: dict[str, int] = {}

    def update_system_state(
        self,
        gpu_utilization: Optional[float] = None,
        gpu_memory_utilization: Optional[float] = None,
        queue_depth: Optional[int] = None,
        active_requests: Optional[int] = None,
    ) -> None:
        """Update the system state used for routing decisions."""
        if gpu_utilization is not None:
            self._system_state.gpu_utilization = gpu_utilization
        if gpu_memory_utilization is not None:
            self._system_state.gpu_memory_utilization = gpu_memory_utilization
        if queue_depth is not None:
            self._system_state.queue_depth = queue_depth
        if active_requests is not None:
            self._system_state.active_requests = active_requests

    def route(self, request: Request) -> LoadedModel:
        """Route a request to the best available model.

        Args:
            request: The incoming request.

        Returns:
            The selected LoadedModel instance.
        """
        # If user specified a model preference, try to honor it
        if request.model_preference != "auto":
            model = self._get_by_preference(request.model_preference)
            if model:
                self._record_decision(model.name)
                request.assigned_model = model.name
                return model

        # Apply routing strategy
        strategy = self.config.strategy
        if strategy == "quality_first":
            model = self._route_quality_first(request)
        elif strategy == "throughput_first":
            model = self._route_throughput_first(request)
        elif strategy == "priority":
            model = self._route_priority(request)
        else:  # adaptive (default)
            model = self._route_adaptive(request)

        self._record_decision(model.name)
        request.assigned_model = model.name

        logger.debug(
            "Routed request %s to model '%s' (strategy=%s, gpu_util=%.2f, queue=%d)",
            request.request_id[:8],
            model.name,
            strategy,
            self._system_state.gpu_utilization,
            self._system_state.queue_depth,
        )

        return model

    def _route_quality_first(self, request: Request) -> LoadedModel:
        """Always route to FP16 unless it's not available."""
        model = self.model_loader.get_model_by_quantization(None)
        if model:
            return model
        # Fall back to any available model
        return self._get_default()

    def _route_throughput_first(self, request: Request) -> LoadedModel:
        """Prefer INT4/INT8 for throughput; fall back to FP16 if idle."""
        state = self._system_state

        if state.queue_depth == 0 and state.gpu_utilization < 0.2:
            # System is idle, use full quality
            model = self.model_loader.get_model_by_quantization(None)
            if model:
                return model

        # Prefer most quantized model
        for quant in ("int4-gptq", "int4-bnb", "int8"):
            model = self.model_loader.get_model_by_quantization(quant)
            if model:
                return model

        return self._get_default()

    def _route_priority(self, request: Request) -> LoadedModel:
        """High-priority -> FP16, normal -> INT4."""
        if request.priority == Priority.HIGH:
            model = self.model_loader.get_model_by_quantization(None)
            if model:
                return model
        else:
            for quant in ("int4-gptq", "int4-bnb", "int8"):
                model = self.model_loader.get_model_by_quantization(quant)
                if model:
                    return model

        return self._get_default()

    def _route_adaptive(self, request: Request) -> LoadedModel:
        """Dynamically route based on GPU utilization and queue depth."""
        state = self._system_state

        if (
            state.gpu_utilization > self.config.gpu_util_high_threshold
            or state.queue_depth > self.config.queue_depth_threshold
        ):
            # High load: use fastest (most quantized) model
            for quant in ("int4-gptq", "int4-bnb", "int8"):
                model = self.model_loader.get_model_by_quantization(quant)
                if model:
                    return model
        elif request.priority == Priority.HIGH:
            # High priority: use best quality
            model = self.model_loader.get_model_by_quantization(None)
            if model:
                return model
        elif state.gpu_utilization < self.config.gpu_util_low_threshold:
            # Low load: use full quality
            model = self.model_loader.get_model_by_quantization(None)
            if model:
                return model
        else:
            # Moderate load: use INT8 as balanced default
            model = self.model_loader.get_model_by_quantization("int8")
            if model:
                return model

        return self._get_default()

    def _get_by_preference(self, preference: str) -> Optional[LoadedModel]:
        """Get a model matching user preference."""
        quant_map = {
            "fp16": None,
            "int8": "int8",
            "int4": "int4-gptq",
        }
        quant = quant_map.get(preference)
        if quant is not None or preference == "fp16":
            return self.model_loader.get_model_by_quantization(quant)

        # Try by name
        return self.model_loader.get_model(preference)

    def _get_default(self) -> LoadedModel:
        """Get the default model (fallback)."""
        model = self.model_loader.get_model()
        if model is None:
            raise RuntimeError("No models loaded")
        return model

    def _record_decision(self, model_name: str) -> None:
        """Record a routing decision for metrics."""
        self.routing_decisions[model_name] = self.routing_decisions.get(model_name, 0) + 1

    def get_stats(self) -> dict:
        """Return routing statistics."""
        return {
            "strategy": self.config.strategy,
            "routing_decisions": dict(self.routing_decisions),
            "system_state": {
                "gpu_utilization": self._system_state.gpu_utilization,
                "gpu_memory_utilization": self._system_state.gpu_memory_utilization,
                "queue_depth": self._system_state.queue_depth,
                "active_requests": self._system_state.active_requests,
            },
        }
