from types import SimpleNamespace

import torch
import pytest

from sglang.srt.disaggregation.agentic_native_memory_adapter import (
    NativeRequestMemoryAdapter,
)


class _ReqRows:
    def __init__(self, values):
        self.req_to_token = values
        self.mamba_pool = None


def _scheduler(values, page_size=4):
    return SimpleNamespace(
        req_to_token_pool=_ReqRows(values),
        token_to_kv_pool_allocator=SimpleNamespace(page_size=page_size),
    )


def test_d2p_snapshot_reads_decode_growth_from_live_request_row():
    # Imported lease indices are intentionally absent.  The live row includes
    # pages materialized by later Decode growth.
    row = torch.arange(100, 124, dtype=torch.int64).reshape(1, -1)
    adapter = NativeRequestMemoryAdapter(_scheduler(row))
    req = SimpleNamespace(
        req_pool_idx=0,
        kv_committed_len=20,
        origin_input_ids=list(range(8)),
        output_ids=list(range(13)),  # reusable prefix excludes final sample
        fill_ids=list(range(21)),
    )
    snapshot = adapter.source_snapshot(req, direction="d2p")
    assert snapshot.token_count == 20
    assert snapshot.token_indices.tolist() == list(range(100, 120))
    assert snapshot.page_indices == (25, 26, 27, 28, 29)


def test_p2d_snapshot_uses_post_dedup_live_mapping_not_old_reservation():
    # Radix dedup can replace imported pages with shared pages.  P->D must use
    # the current request mapping, not the immutable allocation tensor.
    row = torch.tensor(
        [[400, 401, 402, 403, 800, 801, 802, 803, 900, 901, 902, 903]],
        dtype=torch.int64,
    )
    adapter = NativeRequestMemoryAdapter(_scheduler(row))
    req = SimpleNamespace(
        req_pool_idx=0,
        kv_committed_len=12,
        fill_ids=list(range(12)),
    )
    snapshot = adapter.source_snapshot(req, direction="p2d")
    assert snapshot.page_indices == (100, 200, 225)
    assert snapshot.token_indices.tolist()[:4] == [400, 401, 402, 403]


class _StrictPool:
    """CPU physical ownership; catches leaks and duplicate frees explicitly."""

    def __init__(self, live=()):
        self.live = set(live)
        self.freed = []
        self.allocations = []
        self.next_index = 100

    def alloc(self, count):
        indices = torch.arange(self.next_index, self.next_index + count)
        self.next_index += count
        self.allocations.append(indices.tolist())
        self.live.update(indices.tolist())
        return indices

    def free(self, indices):
        values = indices.reshape(-1).tolist()
        assert len(set(values)) == len(values)
        assert set(values) <= self.live, f"duplicate/foreign free: {values}"
        self.live.difference_update(values)
        self.freed.extend(values)


def _hybrid_fixture(*, row_available=True):
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

    pool = HybridReqToTokenPool.__new__(HybridReqToTokenPool)
    pool.free_slots = [0] if row_available else []
    pool.req_to_token = torch.zeros((1, 16), dtype=torch.int32)
    pool.mamba_pool = _StrictPool([10, 11, 12])
    pool.enable_mamba_extra_buffer = True
    pool.mamba_ping_pong_track_buffer_size = 2
    pool.req_index_to_mamba_index_mapping = torch.zeros(1, dtype=torch.int32)
    pool.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.zeros(
        (1, 2), dtype=torch.int32
    )
    allocator = _StrictPool(range(4, 8))
    tree = SimpleNamespace(req_to_token_pool=pool, supports_mamba=lambda: True)
    scheduler = SimpleNamespace(
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=allocator,
        tree_cache=tree,
    )
    req = SimpleNamespace(
        req_pool_idx=None, mamba_pool_idx=None,
        mamba_ping_pong_track_buffer=None,
        origin_input_ids=[1, 2, 3, 4], output_ids=[],
        set_extend_input_len=lambda _: None,
    )
    lease = SimpleNamespace(
        prompt_tokens=4, device_indices=torch.arange(4, 8),
        state_indices=(torch.tensor([10, 11, 12]),),
    )
    return NativeRequestMemoryAdapter(scheduler), pool, allocator, req, lease


def test_decode_binding_uses_lease_state_in_real_hybrid_row_mapping():
    adapter, pool, _, req, lease = _hybrid_fixture()
    adapter.bind_decode_prompt(lease, req)
    assert pool.mamba_pool.allocations == []
    assert pool.req_index_to_mamba_index_mapping.tolist() == [10]
    assert pool.req_index_to_mamba_ping_pong_track_buffer_mapping.tolist() == [[11, 12]]
    assert pool.req_to_token[0, :4].tolist() == [4, 5, 6, 7]


@pytest.mark.parametrize("failure", ["row_full", "write"])
def test_decode_bind_failure_releases_exact_state_and_private_pages(failure):
    adapter, pool, allocator, req, lease = _hybrid_fixture(
        row_available=failure != "row_full"
    )
    if failure == "write":
        def fail_write(*_):
            raise RuntimeError("injected row write failure")
        pool.write = fail_write
    with pytest.raises((MemoryError, RuntimeError)):
        adapter.bind_decode_prompt(lease, req)
    adapter.release_decode_unadopted(lease, req, None)
    assert pool.mamba_pool.allocations == []
    assert not pool.mamba_pool.live
    assert not allocator.live
    assert req.req_pool_idx is None


@pytest.mark.parametrize("insert_result", ["unknown", "donated", "deduplicated"])
def test_prefill_checkpoint_ownership_survives_bind_failure(monkeypatch, insert_result):
    # Real adapter and native no-Req-slot cleanup.  The tiny tree models the
    # precise insert ownership contract, including the shared checkpoint case.
    from sglang.srt.disaggregation import agentic_mamba_prefill
    monkeypatch.setattr(agentic_mamba_prefill, "prefill_state_admission_enabled", lambda *_: True)
    state = _StrictPool([10, 11, 12, 13, 14])
    allocator = _StrictPool(range(4, 12))
    pool = SimpleNamespace(
        mamba_pool=state, enable_mamba_extra_buffer=True,
        mamba_ping_pong_track_buffer_size=2,
    )

    def insert(_):
        if insert_result == "unknown":
            raise RuntimeError("injected insert failure")
        if insert_result == "deduplicated":
            allocator.free(torch.arange(4, 8))
        return SimpleNamespace(prefix_len=4 if insert_result == "deduplicated" else 0,
                               mamba_exist=insert_result == "deduplicated")

    def fail_match(_):
        raise RuntimeError("injected post-insert match failure")

    def release_tree(*_, **__):
        if insert_result == "donated":
            allocator.free(torch.arange(4, 8))
            state.free(torch.tensor([10]))

    tree = SimpleNamespace(
        disable=False, insert=insert, match_prefix=fail_match,
        release_agentic_request_cache=release_tree,
        req_to_token_pool=pool, supports_mamba=lambda: True,
    )
    adapter = NativeRequestMemoryAdapter(SimpleNamespace(
        req_to_token_pool=pool, tree_cache=tree,
        token_to_kv_pool_allocator=allocator, server_args=SimpleNamespace(),
    ))
    req = SimpleNamespace(req_pool_idx=None, mamba_pool_idx=None,
                          origin_input_ids=[1, 2, 3, 4], output_ids=[], extra_key="generation", priority=0)
    lease = SimpleNamespace(
        parent_tokens=4, parent_allocated_tokens=4,
        parent_indices=torch.arange(4, 8), suffix_indices=torch.arange(8, 12),
        device_indices=torch.arange(4, 12), state_indices=(torch.arange(10, 15),),
    )
    with pytest.raises(RuntimeError, match="injected"):
        adapter.bind_prefill_parent(lease, req)
    if insert_result == "unknown":
        with pytest.raises(RuntimeError, match="quarantined"):
            adapter.release_prefill_unadopted(lease, req, None)
        assert state.live == set(range(10, 15))
        assert allocator.live == set(range(4, 12))
    else:
        adapter.release_prefill_unadopted(lease, req, None)
        assert not state.live
        assert not allocator.live


@pytest.mark.parametrize("consumed", [0, 4, 8])
def test_adopted_partial_suffix_cleanup_keeps_consumed_and_private_ownership_disjoint(consumed):
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

    pool = ReqToTokenPool(1, 16, "cpu", enable_memory_saver=False)
    allocator = _StrictPool(range(4, 12))
    req = SimpleNamespace(
        req_pool_idx=None, last_node=None, kv_committed_len=0,
        _agentic_workset_backed=True,
        _agentic_workset_suffix_indices=torch.arange(4 + consumed, 12),
        pop_committed_kv_cache=lambda: consumed,
        pop_overallocated_kv_cache=lambda: (consumed, consumed),
    )
    if consumed:
        pool.alloc([req])
        pool.write((0, slice(0, consumed)), torch.arange(4, 4 + consumed))
    req.kv_committed_len = consumed
    tree = SimpleNamespace(req_to_token_pool=pool, token_to_kv_pool_allocator=allocator,
                           supports_mamba=lambda: False)
    adapter = NativeRequestMemoryAdapter(SimpleNamespace(
        tree_cache=tree, req_to_token_pool=pool,
        token_to_kv_pool_allocator=allocator,
    ))
    adapter.release_bound(SimpleNamespace(prompt_tokens=8), req, None)
    assert not allocator.live
    assert len(allocator.freed) == 8
    assert req.req_pool_idx is None
