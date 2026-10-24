from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation import decode


class _FakeHybridKVPool:
    pass


class _StatePool:
    def __init__(self, available: int):
        self.available = available

    def available_size(self) -> int:
        return self.available


def test_native_pd_mamba_preallocation_applies_backpressure():
    queue = decode.DecodePreallocQueue.__new__(decode.DecodePreallocQueue)
    queue.token_to_kv_pool = _FakeHybridKVPool()
    queue.req_to_token_pool = SimpleNamespace(
        mamba_pool=_StatePool(2),
        enable_mamba_extra_buffer=True,
        mamba_ping_pong_track_buffer_size=2,
    )
    request = SimpleNamespace(
        mamba_pool_idx=None,
        mamba_ping_pong_track_buffer=None,
    )

    with patch.object(decode, "HybridLinearKVPool", _FakeHybridKVPool):
        # One active state plus two ping-pong slots are atomic admission.
        assert not queue._has_mamba_prealloc_capacity(request)
        queue.req_to_token_pool.mamba_pool.available = 3
        assert queue._has_mamba_prealloc_capacity(request)

        # Existing allocations are not double-counted on a resumed request.
        request.mamba_pool_idx = 7
        queue.req_to_token_pool.mamba_pool.available = 2
        assert queue._has_mamba_prealloc_capacity(request)
        request.mamba_ping_pong_track_buffer = [8, 9]
        queue.req_to_token_pool.mamba_pool.available = 0
        assert queue._has_mamba_prealloc_capacity(request)


def test_non_hybrid_native_pd_admission_is_unchanged():
    queue = decode.DecodePreallocQueue.__new__(decode.DecodePreallocQueue)
    queue.token_to_kv_pool = object()
    request = SimpleNamespace()
    assert queue._has_mamba_prealloc_capacity(request)
