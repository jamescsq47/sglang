"""Fresh-workset ownership contract; CPU allocators, no scheduler integration."""
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.disaggregation.agentic_workset import AgenticPWorksetLeaseBroker
from sglang.srt.disaggregation.agentic_mamba_prefill import fork_prefill_checkpoint
from sglang.srt.mem_cache.common import release_kv_cache


class Pool:
    """Strict physical ownership oracle: fail on duplicate/foreign releases."""

    def __init__(self, size):
        self.size = size
        self.free_ids = list(range(size))
        self.calls = []

    def available_size(self):
        return len(self.free_ids)

    def alloc(self, size):
        self.calls.append(size)
        if size > len(self.free_ids):
            return None
        ids, self.free_ids = self.free_ids[:size], self.free_ids[size:]
        return torch.tensor(ids, dtype=torch.int64)

    def free(self, indices):
        ids = indices.tolist()
        assert len(set(ids)) == len(ids)
        assert not set(ids).intersection(self.free_ids)
        assert set(ids).issubset(range(self.size))
        self.free_ids.extend(ids)

    def copy_from(self, source, target):
        assert not set(target.tolist()).intersection(self.free_ids)


def request(tokens=9):
    return NS(origin_input_ids=list(range(tokens)), req_pool_idx=None,
              prefix_indices=torch.empty(0, dtype=torch.int64), mamba_pool_idx=None,
              mamba_ping_pong_track_buffer=None)


def broker(*, states=(), checkpoint=True):
    req_pool = NS(enable_mamba_extra_buffer=True, mamba_ping_pong_track_buffer_size=2)
    return AgenticPWorksetLeaseBroker(4, state_allocators=states,
        mamba_req_to_token_pool=req_pool if states else None,
        reserve_mamba_checkpoint=checkpoint)


@pytest.mark.parametrize("size", [1, 2, 8])
def test_complete_fresh_lease_chunks_without_bind_or_io(size):
    brokers = [broker() for _ in range(size)]
    allocators = [Pool(32) for _ in range(size)]
    for b, alloc in zip(brokers, allocators):
        assert b.request("fresh:0", 0, 9, owner="fresh")
        if size > 1:
            b.install_tp_plan(1, (("fresh:0", "fresh", 0, 9),))
        b.service(alloc)
        lease = b.get("fresh:0")
        assert lease.parent_tokens == lease.parent_allocated_tokens == 0
        assert not lease.parent_bound and lease.io_attempt is None
        assert lease.parent_indices.numel() == lease.parent_page_indices.size == 0
        assert lease.suffix_allocated_tokens == 12 and alloc.available_size() == 20
        assert not b.begin_bind("fresh:0", lease)
        assert not b.begin_io_attempt("fresh:0", lease, "fake-io")
        req = request()
        b.handoff_fresh_to_req("fresh:0", req, lease)
        b.handoff_fresh_to_req("fresh:0", req, lease)  # Exact replay only.
        assert lease.state == "handed" and not lease.parent_bound
        first = b.consume_suffix(lease, 4, final_prompt_chunk=False)
        second = b.consume_suffix(lease, 4, final_prompt_chunk=False)
        final = b.consume_suffix(lease, 1, final_prompt_chunk=True)
        assert torch.cat((first, second, final)).tolist() == list(range(9))
        assert alloc.calls == [12]  # Later chunks never allocate another page.
        assert b.get("fresh:0") is None and alloc.available_size() == 20
        alloc.free(lease.device_indices)  # Native Req owns final-page padding.
        assert alloc.available_size() == 32


@pytest.mark.parametrize("size", [1, 2, 8])
def test_partial_fresh_cancel_frees_only_broker_suffix(size):
    for _ in range(size):
        b, alloc = broker(), Pool(32)
        b.request("recompute:2", 0, 9, owner="recompute")
        if size > 1:
            b.install_tp_plan(1, (("recompute:2", "recompute", 0, 9),))
        b.service(alloc)
        lease, req = b.get("recompute:2"), request()
        b.handoff_fresh_to_req("recompute:2", req, lease)
        chunk = b.consume_suffix(lease, 4, final_prompt_chunk=False)
        assert b.unaccounted_tokens == 8  # Consumed native pages not counted twice.
        assert b.release_handed("recompute:2", lease, req=req)
        if size > 1:
            assert b.tp_retire_ready("recompute:2")
            assert b.commit_tp_retire("recompute:2")
        b.service(alloc)
        assert alloc.available_size() == 28
        assert not set(chunk.tolist()).intersection(alloc.free_ids)
        assert not b.release_handed("recompute:2", lease, req=req)
        alloc.free(chunk)
        assert alloc.available_size() == 32


@pytest.mark.parametrize("checkpoint", [False, True])
def test_fresh_mamba_runtime_and_two_rolling_checkpoints_without_parent(checkpoint):
    state, alloc = Pool(16), Pool(32)
    b = broker(states=(state,), checkpoint=checkpoint)
    b.request("fresh", 0, 9)
    b.service(alloc)
    lease, req = b.get("fresh"), request()
    assert state.calls == [5]  # Active + two tracking + two output; no parent.
    assert not lease.state_device_indices
    b.handoff_fresh_to_req("fresh", req, lease)
    assert req.mamba_pool_idx.item() == 0
    assert req.mamba_ping_pong_track_buffer.tolist() == [1, 2]
    assert req._agentic_mamba_prefill_checkpoint.tolist() == [3, 4]
    assert not lease.runtime_state_device_indices
    # Existing checkpoint consumer uses preallocated outputs, not fork/alloc.
    donated = fork_prefill_checkpoint(req, state, req.mamba_pool_idx.unsqueeze(0))
    assert donated.tolist() == [3]
    assert req._agentic_mamba_prefill_checkpoint.tolist() == [4]
    assert state.calls == [5]
    assert b.release_handed("fresh", lease, req=req)
    b.service(alloc)
    tree = NS(supports_mamba=lambda: True, req_to_token_pool=NS(mamba_pool=state))
    release_kv_cache(req, tree, is_insert=False)
    release_kv_cache(req, tree, is_insert=False)
    # A checkpoint already donated to Radix must survive Req cancellation.
    assert state.available_size() == 15 and 3 not in state.free_ids
    state.free(donated)
    assert state.available_size() == 16 and alloc.available_size() == 32


@pytest.mark.parametrize("failure_pool", [0, 1])
def test_fresh_mamba_allocation_failure_rolls_back_full_transaction(failure_pool):
    states = [Pool(16), Pool(16)]
    states[failure_pool] = Pool(4)  # Cannot fit the complete five runtime slots.
    alloc, b = Pool(32), broker(states=states)
    b.request("fresh", 0, 9)
    b.service(alloc)
    assert b.get("fresh") is None
    assert not b.drain_grant_events()
    assert alloc.available_size() == 32
    assert all(pool.available_size() == pool.size for pool in states)
    assert b.cancel_unstarted("fresh")


def test_cancel_before_handoff_returns_all_fresh_runtime_and_attention():
    state, alloc = Pool(16), Pool(32)
    b = broker(states=(state,))
    b.request("fresh", 0, 9)
    b.service(alloc)
    lease = b.get("fresh")
    assert b.cancel_unstarted("fresh")
    with pytest.raises(RuntimeError):
        b.handoff_fresh_to_req("fresh", request(), lease)
    b.service(alloc)
    assert state.available_size() == 16 and alloc.available_size() == 32


def test_fresh_runtime_materialization_failure_keeps_only_lease_owner(monkeypatch):
    state, alloc = Pool(16), Pool(32)
    b = broker(states=(state,))
    b.request("fresh", 0, 9)
    b.service(alloc)
    lease, req = b.get("fresh"), request()
    def fail(*args, **kwargs):
        raise RuntimeError("injected tracking tensor failure")
    monkeypatch.setattr(torch, "full", fail)
    with pytest.raises(RuntimeError, match="tracking tensor"):
        b.handoff_fresh_to_req("fresh", req, lease)
    assert lease.state == "active" and lease.runtime_state_req is None
    assert req.mamba_pool_idx is None and req.mamba_ping_pong_track_buffer is None
    assert getattr(req, "_agentic_mamba_prefill_checkpoint", None) is None
    assert b.cancel_unstarted("fresh")
    b.service(alloc)
    tree = NS(supports_mamba=lambda: True, req_to_token_pool=NS(mamba_pool=state))
    release_kv_cache(req, tree, is_insert=False)
    assert state.available_size() == 16 and alloc.available_size() == 32


@pytest.mark.parametrize("bad", ["prompt", "prefix", "runtime", "other_req"])
def test_invalid_fresh_handoff_does_not_mutate_ownership(bad):
    b, alloc = broker(), Pool(32)
    b.request("fresh", 0, 9)
    b.service(alloc)
    lease, req = b.get("fresh"), request()
    if bad == "prompt":
        req.origin_input_ids.append(99)
    elif bad == "prefix":
        req.prefix_indices = torch.tensor([10])
    elif bad == "runtime":
        req.mamba_pool_idx = torch.tensor(1)
    else:
        b.handoff_fresh_to_req("fresh", request(), lease)
    with pytest.raises(RuntimeError):
        b.handoff_fresh_to_req("fresh", req, lease)
    assert lease.state == ("handed" if bad == "other_req" else "active")
    assert alloc.available_size() == 20


def test_restore_still_uses_parent_checkpoint_and_requires_actual_bind():
    state, alloc = Pool(16), Pool(32)
    b = broker(states=(state,))
    b.request("restore", 4, 9)
    b.service(alloc)
    lease, req = b.get("restore"), request()
    assert state.calls == [1, 4]
    with pytest.raises(RuntimeError):
        b.handoff_fresh_to_req("restore", req, lease)
    assert b.begin_bind("restore", lease)
    b.commit_parent_bound("restore", lease, state_duplicate=True)
    b.attach_runtime_state_for_bind("restore", req, lease)
    b.handoff_to_req("restore", req, lease)
    assert req._agentic_mamba_prefill_checkpoint.numel() == 1
    assert lease.parent_bound


@pytest.mark.parametrize("parent,prompt", [(-1, 8), (0, 0), (8, 4)])
def test_invalid_shape_rejected(parent, prompt):
    with pytest.raises(ValueError):
        broker().request("bad", parent, prompt)
