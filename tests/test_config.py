"""Tests for configuration loading."""

import os
import tempfile

import yaml

from liteserve.config import load_config


class TestConfig:
    def test_load_default_config(self):
        config = load_config(
            os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
        )
        assert config.server.port == 8000
        assert config.scheduler.max_batch_size == 32
        assert config.kv_cache.block_size == 16
        assert config.router.strategy == "adaptive"
        assert config.metrics.enabled is True

    def test_load_custom_config(self):
        custom = {
            "server": {"host": "127.0.0.1", "port": 9000, "workers": 2},
            "models": [
                {
                    "name": "test-model",
                    "path": "/tmp/model",
                    "quantization": "int8",
                    "max_model_len": 1024,
                    "default": True,
                }
            ],
            "scheduler": {
                "max_batch_size": 16,
                "max_waiting_time_ms": 50,
                "memory_budget_pct": 0.8,
                "preemption_policy": "priority",
                "scheduling_interval_ms": 5,
            },
            "kv_cache": {"block_size": 32, "swap_space_gb": 2.0},
            "router": {
                "strategy": "throughput_first",
                "gpu_util_high_threshold": 0.9,
                "gpu_util_low_threshold": 0.3,
                "queue_depth_threshold": 10,
            },
            "metrics": {"enabled": False, "port": 8080},
            "logging": {"level": "DEBUG", "format": "text"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(custom, f)
            tmp_path = f.name

        try:
            config = load_config(tmp_path)
            assert config.server.port == 9000
            assert config.server.workers == 2
            assert len(config.models) == 1
            assert config.models[0].name == "test-model"
            assert config.models[0].quantization == "int8"
            assert config.scheduler.max_batch_size == 16
            assert config.scheduler.preemption_policy == "priority"
            assert config.kv_cache.block_size == 32
            assert config.router.strategy == "throughput_first"
            assert config.metrics.enabled is False
            assert config.logging.level == "DEBUG"
        finally:
            os.unlink(tmp_path)
