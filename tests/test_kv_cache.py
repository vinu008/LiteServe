"""Tests for the Paged KV-Cache Manager."""

import pytest
import torch

from liteserve.engine.kv_cache import KVCacheOutOfMemoryError, PagedKVCache


@pytest.fixture
def kv_cache():
    """Create a small KV-cache for testing (CPU-based)."""
    return PagedKVCache(
        num_layers=2,
        num_heads=4,
        head_dim=8,
        block_size=4,
        num_blocks=16,
        dtype=torch.float32,
        device="cpu",
    )


class TestPagedKVCache:
    def test_initialization(self, kv_cache):
        assert kv_cache.num_blocks == 16
        assert kv_cache.num_free_blocks == 16
        assert kv_cache.num_used_blocks == 0
        assert kv_cache.utilization == 0.0

    def test_allocate(self, kv_cache):
        blocks = kv_cache.allocate("req1", 8)  # 8 tokens -> 2 blocks (block_size=4)
        assert len(blocks) == 2
        assert kv_cache.num_free_blocks == 14
        assert kv_cache.num_used_blocks == 2

    def test_allocate_partial_block(self, kv_cache):
        blocks = kv_cache.allocate("req1", 5)  # 5 tokens -> 2 blocks
        assert len(blocks) == 2

    def test_allocate_exact_block(self, kv_cache):
        blocks = kv_cache.allocate("req1", 4)  # 4 tokens -> 1 block
        assert len(blocks) == 1

    def test_free(self, kv_cache):
        kv_cache.allocate("req1", 8)
        assert kv_cache.num_free_blocks == 14

        kv_cache.free("req1")
        assert kv_cache.num_free_blocks == 16
        assert "req1" not in kv_cache.block_tables

    def test_free_nonexistent(self, kv_cache):
        # Should not raise
        kv_cache.free("nonexistent")

    def test_can_allocate(self, kv_cache):
        assert kv_cache.can_allocate(64)  # 64 tokens -> 16 blocks (exactly fits)
        assert not kv_cache.can_allocate(65)  # 65 tokens -> 17 blocks (too many)

    def test_oom_error(self, kv_cache):
        with pytest.raises(KVCacheOutOfMemoryError):
            kv_cache.allocate("req1", 100)  # 100 tokens -> 25 blocks > 16

    def test_multiple_requests(self, kv_cache):
        kv_cache.allocate("req1", 8)  # 2 blocks
        kv_cache.allocate("req2", 12)  # 3 blocks
        assert kv_cache.num_free_blocks == 11
        assert len(kv_cache.block_tables) == 2

        kv_cache.free("req1")
        assert kv_cache.num_free_blocks == 13

    def test_store_and_retrieve_kv(self, kv_cache):
        kv_cache.allocate("req1", 4)

        # Store a KV pair
        key = torch.randn(4, 8)  # [num_heads, head_dim]
        value = torch.randn(4, 8)
        kv_cache.store_kv("req1", layer_idx=0, position=0, key=key, value=value)

        # Retrieve
        result = kv_cache.get_kv_for_request("req1", layer_idx=0)
        assert result is not None
        k, v = result
        assert k.shape[1] == 4  # num_heads
        assert k.shape[2] == 8  # head_dim

    def test_swap_out_and_in(self, kv_cache):
        kv_cache.allocate("req1", 8)
        assert kv_cache.num_free_blocks == 14

        # Swap out to CPU
        kv_cache.swap_out("req1")
        assert kv_cache.num_free_blocks == 16
        assert "req1" in kv_cache.swap_space

        # Swap back in
        success = kv_cache.swap_in("req1")
        assert success
        assert kv_cache.num_free_blocks == 14
        assert "req1" not in kv_cache.swap_space

    def test_get_stats(self, kv_cache):
        kv_cache.allocate("req1", 8)
        stats = kv_cache.get_stats()
        assert stats["total_blocks"] == 16
        assert stats["used_blocks"] == 2
        assert stats["free_blocks"] == 14
        assert stats["active_requests"] == 1
        assert stats["swapped_requests"] == 0

    def test_utilization(self, kv_cache):
        assert kv_cache.utilization == 0.0
        kv_cache.allocate("req1", 32)  # 8 blocks
        assert kv_cache.utilization == 0.5
