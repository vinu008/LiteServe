"""Configuration loader for LiteServe."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    name: str
    path: str
    quantization: Optional[str] = None  # null, "int8", "int4-gptq"
    max_model_len: int = 2048
    default: bool = False


@dataclass
class SchedulerConfig:
    max_batch_size: int = 32
    max_waiting_time_ms: int = 100
    memory_budget_pct: float = 0.9
    preemption_policy: str = "fcfs"
    scheduling_interval_ms: int = 1


@dataclass
class KVCacheConfig:
    block_size: int = 16
    swap_space_gb: float = 1.0


@dataclass
class RouterConfig:
    strategy: str = "adaptive"
    gpu_util_high_threshold: float = 0.85
    gpu_util_low_threshold: float = 0.4
    queue_depth_threshold: int = 20


@dataclass
class MetricsConfig:
    enabled: bool = True
    port: int = 9090


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "json"


@dataclass
class LiteServeConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    models: list[ModelConfig] = field(default_factory=list)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    kv_cache: KVCacheConfig = field(default_factory=KVCacheConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(config_path: Optional[str] = None) -> LiteServeConfig:
    """Load configuration from a YAML file."""
    if config_path is None:
        config_path = os.environ.get(
            "LITESERVE_CONFIG",
            str(Path(__file__).parent.parent / "configs" / "default.yaml"),
        )

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    config = LiteServeConfig()

    if "server" in raw:
        config.server = ServerConfig(**raw["server"])

    if "models" in raw:
        config.models = [ModelConfig(**m) for m in raw["models"]]

    if "scheduler" in raw:
        config.scheduler = SchedulerConfig(**raw["scheduler"])

    if "kv_cache" in raw:
        config.kv_cache = KVCacheConfig(**raw["kv_cache"])

    if "router" in raw:
        config.router = RouterConfig(**raw["router"])

    if "metrics" in raw:
        config.metrics = MetricsConfig(**raw["metrics"])

    if "logging" in raw:
        config.logging = LoggingConfig(**raw["logging"])

    return config
