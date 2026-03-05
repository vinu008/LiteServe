"""Model loading and management with quantization support.

Handles loading LLM models (Llama 2, Mistral, etc.) with various
quantization backends: FP16, INT8 (bitsandbytes), INT4 (GPTQ).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


def _detect_device() -> str:
    """Detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class LoadedModel:
    """Represents a loaded model variant."""

    name: str
    model: Any  # The actual model object
    tokenizer: Any  # The tokenizer
    quantization: Optional[str]
    max_model_len: int
    device: str
    dtype: torch.dtype
    num_layers: int = 0
    num_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    vocab_size: int = 0
    load_time: float = 0.0
    memory_mb: float = 0.0


class ModelLoader:
    """Loads and manages LLM models with quantization support."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or _detect_device()
        self.models: dict[str, LoadedModel] = {}
        self.default_model: Optional[str] = None

    def load_model(
        self,
        name: str,
        model_path: str,
        quantization: Optional[str] = None,
        max_model_len: int = 2048,
        is_default: bool = False,
    ) -> LoadedModel:
        """Load a model with optional quantization.

        Args:
            name: Friendly name for this model variant.
            model_path: HuggingFace model path or local directory.
            quantization: None for FP16, "int8", or "int4-gptq".
            max_model_len: Maximum sequence length.
            is_default: Whether this is the default model.

        Returns:
            LoadedModel instance.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading model '%s' from '%s' (quantization=%s, device=%s)",
                     name, model_path, quantization, self.device)
        start_time = time.time()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Determine dtype — MPS doesn't support float16 for all ops
        if self.device == "mps":
            model_dtype = torch.float32
        else:
            model_dtype = torch.float16

        # Load model with quantization
        model_kwargs = self._get_model_kwargs(quantization)

        if self.device == "cuda" and quantization is not None:
            # Quantized models need device_map="auto" on CUDA
            model_kwargs["device_map"] = "auto"
        elif self.device == "cuda":
            model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=model_dtype,
            **model_kwargs,
        )

        # Move to device if not already placed by device_map
        if not hasattr(model, "hf_device_map"):
            model = model.to(self.device)

        model.eval()

        # Extract model config
        config = model.config
        num_layers = getattr(config, "num_hidden_layers", 32)
        num_heads = getattr(config, "num_attention_heads", 32)
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        hidden_size = getattr(config, "hidden_size", 4096)
        head_dim = getattr(config, "head_dim", hidden_size // num_heads)
        vocab_size = config.vocab_size

        load_time = time.time() - start_time
        memory_mb = self._get_model_memory(model)

        loaded = LoadedModel(
            name=name,
            model=model,
            tokenizer=tokenizer,
            quantization=quantization,
            max_model_len=max_model_len,
            device=self.device,
            dtype=model_dtype,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
            load_time=load_time,
            memory_mb=memory_mb,
        )

        self.models[name] = loaded
        if is_default or self.default_model is None:
            self.default_model = name

        logger.info(
            "Loaded model '%s': %d layers, %d heads, head_dim=%d, vocab=%d, "
            "%.1f MB, loaded in %.1fs",
            name, num_layers, num_heads, head_dim, vocab_size, memory_mb, load_time,
        )

        return loaded

    def _get_model_kwargs(self, quantization: Optional[str]) -> dict:
        """Get model loading kwargs based on quantization type."""
        if quantization is None:
            return {}
        elif quantization == "int8":
            try:
                from transformers import BitsAndBytesConfig
                return {
                    "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
                }
            except ImportError:
                logger.warning("bitsandbytes not available, falling back to FP16")
                return {}
        elif quantization == "int4-gptq":
            # GPTQ models are pre-quantized, just load normally
            return {}
        elif quantization == "int4-bnb":
            try:
                from transformers import BitsAndBytesConfig
                return {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                    ),
                }
            except ImportError:
                logger.warning("bitsandbytes not available, falling back to FP16")
                return {}
        else:
            logger.warning("Unknown quantization '%s', using FP16", quantization)
            return {}

    def _get_model_memory(self, model: Any) -> float:
        """Estimate model memory usage in MB."""
        total_bytes = 0
        for p in model.parameters():
            total_bytes += p.numel() * p.element_size()
        return total_bytes / (1024**2)

    def get_model(self, name: Optional[str] = None) -> Optional[LoadedModel]:
        """Get a loaded model by name, or the default model."""
        if name is None:
            name = self.default_model
        return self.models.get(name)

    def get_available_models(self) -> list[str]:
        """Return names of all loaded models."""
        return list(self.models.keys())

    def get_model_by_quantization(self, quantization: Optional[str]) -> Optional[LoadedModel]:
        """Find a loaded model matching the given quantization level."""
        for model in self.models.values():
            if model.quantization == quantization:
                return model
        return None

    def unload_model(self, name: str) -> None:
        """Unload a model and free memory."""
        if name in self.models:
            loaded = self.models.pop(name)
            del loaded.model
            del loaded.tokenizer
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("Unloaded model '%s'", name)
            if self.default_model == name:
                self.default_model = next(iter(self.models), None)

    def get_stats(self) -> dict:
        """Return statistics about loaded models."""
        return {
            "loaded_models": [
                {
                    "name": m.name,
                    "quantization": m.quantization,
                    "memory_mb": m.memory_mb,
                    "max_model_len": m.max_model_len,
                }
                for m in self.models.values()
            ],
            "default_model": self.default_model,
            "total_memory_mb": sum(m.memory_mb for m in self.models.values()),
        }
