"""Paged KV-Cache Manager implementing PagedAttention memory management.

This module manages GPU memory for KV-cache using a block-based allocation scheme
inspired by the vLLM paper. Instead of pre-allocating contiguous memory for the
maximum sequence length of every request, memory is divided into fixed-size blocks
that are allocated on demand.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class KVCacheOutOfMemoryError(Exception):
    """Raised when the KV-cache cannot allocate enough blocks."""


@dataclass
class BlockMetadata:
    """Metadata for a single KV-cache block."""

    block_id: int
    ref_count: int = 1  # For copy-on-write prefix sharing
    num_tokens_filled: int = 0  # How many token slots are used


class PagedKVCache:
    """Paged KV-Cache manager using block-based allocation.

    Each block stores K and V tensors for `block_size` token positions across
    all layers and heads. Blocks are allocated from a pre-allocated pool and
    tracked via block tables per request.

    Args:
        num_layers: Number of transformer layers.
        num_heads: Number of KV attention heads.
        head_dim: Dimension per attention head.
        block_size: Number of tokens per block.
        num_blocks: Total number of blocks in the pool.
        dtype: Data type for KV tensors.
        device: Device to allocate on.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int = 16,
        num_blocks: int = 256,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.dtype = dtype
        self.device = device

        # Pre-allocate the full KV-cache pool
        # Shape: [num_blocks, 2 (K+V), num_layers, block_size, num_heads, head_dim]
        self.kv_pool = torch.zeros(
            (num_blocks, 2, num_layers, block_size, num_heads, head_dim),
            dtype=dtype,
            device=device,
        )

        # Free block pool (set of available block IDs)
        self.free_blocks: set[int] = set(range(num_blocks))

        # Block tables: request_id -> list of block IDs (in sequence order)
        self.block_tables: dict[str, list[int]] = {}

        # Block metadata
        self.block_metadata: dict[int, BlockMetadata] = {}

        # CPU swap space for preempted requests
        self.swap_space: dict[str, torch.Tensor] = {}

        logger.info(
            "Initialized PagedKVCache: %d blocks × %d tokens/block = %d max tokens, "
            "%.2f GB GPU memory",
            num_blocks,
            block_size,
            num_blocks * block_size,
            self._pool_memory_gb(),
        )

    def _pool_memory_gb(self) -> float:
        """Calculate memory usage of the KV pool in GB."""
        bytes_per_element = 2 if self.dtype == torch.float16 else 4
        total_bytes = (
            self.num_blocks
            * 2
            * self.num_layers
            * self.block_size
            * self.num_heads
            * self.head_dim
            * bytes_per_element
        )
        return total_bytes / (1024**3)

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - len(self.free_blocks)

    @property
    def utilization(self) -> float:
        return self.num_used_blocks / self.num_blocks if self.num_blocks > 0 else 0.0

    def can_allocate(self, num_tokens: int) -> bool:
        """Check if enough blocks are available for the given number of tokens."""
        blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        return blocks_needed <= len(self.free_blocks)

    def allocate(self, request_id: str, num_tokens: int) -> list[int]:
        """Allocate blocks for a request.

        Args:
            request_id: Unique request identifier.
            num_tokens: Number of tokens to allocate space for.

        Returns:
            List of allocated block IDs.

        Raises:
            KVCacheOutOfMemoryError: If not enough blocks are available.
        """
        blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if blocks_needed > len(self.free_blocks):
            raise KVCacheOutOfMemoryError(
                f"Need {blocks_needed} blocks but only {len(self.free_blocks)} available"
            )

        allocated = []
        for _ in range(blocks_needed):
            block_id = self.free_blocks.pop()
            allocated.append(block_id)
            self.block_metadata[block_id] = BlockMetadata(block_id=block_id)

        if request_id not in self.block_tables:
            self.block_tables[request_id] = []
        self.block_tables[request_id].extend(allocated)

        logger.debug(
            "Allocated %d blocks for request %s (total: %d, free: %d)",
            blocks_needed,
            request_id[:8],
            len(self.block_tables[request_id]),
            len(self.free_blocks),
        )

        return allocated

    def extend(self, request_id: str, additional_tokens: int) -> list[int]:
        """Allocate additional blocks for an ongoing request.

        Only allocates if the current last block is full.
        """
        if request_id not in self.block_tables:
            return self.allocate(request_id, additional_tokens)

        current_blocks = self.block_tables[request_id]
        if not current_blocks:
            return self.allocate(request_id, additional_tokens)

        # Check if the last block has space
        last_block = current_blocks[-1]
        meta = self.block_metadata.get(last_block)
        if meta and meta.num_tokens_filled < self.block_size:
            # Last block still has space, no new allocation needed
            return []

        # Need a new block
        return self.allocate(request_id, self.block_size)

    def free(self, request_id: str) -> None:
        """Free all blocks belonging to a request."""
        if request_id not in self.block_tables:
            return

        blocks = self.block_tables.pop(request_id)
        for block_id in blocks:
            meta = self.block_metadata.get(block_id)
            if meta:
                meta.ref_count -= 1
                if meta.ref_count <= 0:
                    # Zero out the block and return to free pool
                    self.kv_pool[block_id].zero_()
                    self.free_blocks.add(block_id)
                    del self.block_metadata[block_id]

        # Clean up swap space if any
        self.swap_space.pop(request_id, None)

        logger.debug(
            "Freed %d blocks for request %s (free: %d)",
            len(blocks),
            request_id[:8],
            len(self.free_blocks),
        )

    def swap_out(self, request_id: str) -> None:
        """Swap a request's KV-cache from GPU to CPU (for preemption)."""
        if request_id not in self.block_tables:
            return

        blocks = self.block_tables[request_id]
        # Copy KV data to CPU
        block_data = torch.stack([self.kv_pool[bid] for bid in blocks])
        self.swap_space[request_id] = block_data.to("cpu")

        # Free GPU blocks
        for block_id in blocks:
            self.kv_pool[block_id].zero_()
            self.free_blocks.add(block_id)
            if block_id in self.block_metadata:
                del self.block_metadata[block_id]

        self.block_tables[request_id] = []

        logger.info("Swapped out %d blocks for request %s to CPU", len(blocks), request_id[:8])

    def swap_in(self, request_id: str) -> bool:
        """Swap a request's KV-cache from CPU back to GPU."""
        if request_id not in self.swap_space:
            return False

        block_data = self.swap_space[request_id]
        num_blocks_needed = block_data.shape[0]

        if num_blocks_needed > len(self.free_blocks):
            return False  # Not enough GPU memory

        new_blocks = []
        for i in range(num_blocks_needed):
            block_id = self.free_blocks.pop()
            self.kv_pool[block_id] = block_data[i].to(self.device)
            self.block_metadata[block_id] = BlockMetadata(
                block_id=block_id,
                num_tokens_filled=self.block_size,
            )
            new_blocks.append(block_id)

        self.block_tables[request_id] = new_blocks
        del self.swap_space[request_id]

        logger.info("Swapped in %d blocks for request %s to GPU", num_blocks_needed, request_id[:8])
        return True

    def get_kv_for_request(
        self, request_id: str, layer_idx: int
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Get K and V tensors for a specific request and layer.

        Gathers KV data from scattered blocks into contiguous tensors.

        Returns:
            Tuple of (key_tensor, value_tensor) each of shape
            [total_tokens, num_heads, head_dim], or None if request not found.
        """
        if request_id not in self.block_tables:
            return None

        blocks = self.block_tables[request_id]
        if not blocks:
            return None

        # Gather K and V from blocks
        k_parts = []
        v_parts = []
        for block_id in blocks:
            meta = self.block_metadata.get(block_id)
            tokens_in_block = meta.num_tokens_filled if meta else self.block_size

            k_parts.append(self.kv_pool[block_id, 0, layer_idx, :tokens_in_block])
            v_parts.append(self.kv_pool[block_id, 1, layer_idx, :tokens_in_block])

        keys = torch.cat(k_parts, dim=0)  # [total_tokens, num_heads, head_dim]
        values = torch.cat(v_parts, dim=0)

        return keys, values

    def store_kv(
        self,
        request_id: str,
        layer_idx: int,
        position: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Store a K/V pair for a single token position.

        Args:
            request_id: Request identifier.
            layer_idx: Transformer layer index.
            position: Token position in the sequence.
            key: Key tensor of shape [num_heads, head_dim].
            value: Value tensor of shape [num_heads, head_dim].
        """
        if request_id not in self.block_tables:
            return

        blocks = self.block_tables[request_id]
        block_idx = position // self.block_size
        offset_in_block = position % self.block_size

        # Allocate new block if needed
        if block_idx >= len(blocks):
            new_blocks = self.allocate(request_id, self.block_size)
            if not new_blocks:
                return
            blocks = self.block_tables[request_id]

        block_id = blocks[block_idx]
        self.kv_pool[block_id, 0, layer_idx, offset_in_block] = key
        self.kv_pool[block_id, 1, layer_idx, offset_in_block] = value

        meta = self.block_metadata.get(block_id)
        if meta:
            meta.num_tokens_filled = max(meta.num_tokens_filled, offset_in_block + 1)

    def get_stats(self) -> dict:
        """Return cache statistics for monitoring."""
        return {
            "total_blocks": self.num_blocks,
            "free_blocks": self.num_free_blocks,
            "used_blocks": self.num_used_blocks,
            "utilization": self.utilization,
            "active_requests": len(self.block_tables),
            "swapped_requests": len(self.swap_space),
            "pool_memory_gb": self._pool_memory_gb(),
        }
