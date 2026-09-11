from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.disaggregation.agentic_mamba_prefill import (
    reserve_prefill_state, fork_prefill_checkpoint, release_prefill_checkpoint,
    missing_runtime_slots,
    prefill_state_admission_enabled,
)
from sglang.srt.managers.scheduler import AgenticPWorksetLeaseBroker
from sglang.srt.mem_cache.common import release_kv_cache


class Pool:
    def __init__(self, size):
        self.free_slots = torch.arange(1, size + 1)
        self.copies = []

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, size):
        if size > self.available_size():
            return None
        out, self.free_slots = self.free_slots[:size], self.free_slots[size:]
        return out

    def free(self, indices):
        assert not set(indices.tolist()) & set(self.free_slots.tolist()), 'double free'
        self.free_slots = torch.cat((self.free_slots, indices))

    def copy_from(self, source, target):
        self.copies.append((source.clone(), target.clone()))

    def fork_from(self, source):
        target = self.alloc(len(source))
        if target is not None:
            self.copy_from(source, target)
        return target


def setup(size):
    pool = Pool(size)
    req_pool = NS(mamba_pool=pool, enable_mamba_extra_buffer=True,
                  mamba_ping_pong_track_buffer_size=2)
    cache = NS(req_to_token_pool=req_pool, supports_mamba=lambda: True,
               evict=lambda _: None)
    return pool, req_pool, cache


def req():
    return NS(mamba_pool_idx=None, mamba_ping_pong_track_buffer=None,
              req_pool_idx=None, mamba_next_track_idx=0)


@pytest.mark.parametrize('mode', ['prefill', 'decode', 'null'])
@pytest.mark.parametrize('mamba', [False, True])
def test_initialization_gate_uses_server_args_not_later_scheduler_fields(monkeypatch, mode, mamba):
    for flag in ('LIFECYCLE', 'MAMBA_PROMPT_CHECKPOINT', 'CUSTOM_STORAGE_ONLY', 'MAMBA_REQUEST_OWNED'):
        monkeypatch.setenv('SGLANG_AGENTIC_KV_' + flag, 'true')
    pool = NS(mamba_pool=object()) if mamba else NS()
    assert prefill_state_admission_enabled(NS(disaggregation_mode=mode), pool) == (mode == 'prefill' and mamba)
    monkeypatch.setenv('SGLANG_AGENTIC_KV_CUSTOM_STORAGE_ONLY', 'false')
    assert not prefill_state_admission_enabled(NS(disaggregation_mode=mode), pool)


@pytest.mark.parametrize('free', [0, 1, 2, 3])
def test_new_requires_complete_runtime_and_checkpoint(free):
    pool, req_pool, cache = setup(free)
    request = req()
    assert reserve_prefill_state(request, req_pool, cache) is None
    assert pool.available_size() == free
    assert request.mamba_pool_idx is None


def test_twenty_new_requests_do_not_overcommit_remaining_slots():
    pool, req_pool, cache = setup(39)
    selected = []
    for _ in range(20):
        request = req()
        reservation = reserve_prefill_state(request, req_pool, cache)
        if reservation is not None:
            reservation.finish(True)
            selected.append(request)
    assert len(selected) == 9 and pool.available_size() == 3
    assert all(missing_runtime_slots(r, req_pool) == 0 for r in selected)
    for request in selected:
        release_kv_cache(request, cache)
        release_kv_cache(request, cache)
    assert pool.available_size() == 39


def test_rejected_candidate_returns_only_new_slots():
    pool, req_pool, cache = setup(4)
    request = req()
    request.mamba_pool_idx = pool.alloc(1)[0]  # Existing COW state.
    reservation = reserve_prefill_state(request, req_pool, cache)
    assert pool.available_size() == 0
    reservation.finish(False)
    reservation.finish(False)
    assert request.mamba_pool_idx.item() == 1
    assert request.mamba_ping_pong_track_buffer is None
    assert request._agentic_mamba_prefill_checkpoint is None
    assert pool.available_size() == 3


def test_resident_request_progresses_when_pool_full_and_rollback_keeps_owner():
    pool, req_pool, cache = setup(4)
    request = req()
    reserve_prefill_state(request, req_pool, cache).finish(True)
    assert reserve_prefill_state(req(), req_pool, cache) is None
    already = reserve_prefill_state(request, req_pool, cache)
    assert already is not None
    already.finish(False)
    assert pool.available_size() == 0
    release_kv_cache(request, cache)
    assert pool.available_size() == 4


def test_output_checkpoint_survives_overlapped_allocation_and_chunk_replenish():
    pool, req_pool, cache = setup(8)
    prev, current = req(), req()
    reserve_prefill_state(prev, req_pool, cache).finish(True)
    reserve_prefill_state(current, req_pool, cache).finish(True)
    assert pool.available_size() == 0
    fork = fork_prefill_checkpoint(prev, pool, prev.mamba_pool_idx.reshape(1))
    assert fork is not None and pool.available_size() == 0
    assert prev._agentic_mamba_prefill_checkpoint is None
    assert reserve_prefill_state(prev, req_pool, cache) is None
    pool.free(fork)  # Old/deduplicated Radix checkpoint retired.
    reserve_prefill_state(prev, req_pool, cache).finish(True)
    assert pool.available_size() == 0
    assert len(pool.copies) == 1


def test_checkpoint_cleanup_is_idempotent_and_native_fork_unchanged():
    pool, req_pool, cache = setup(4)
    request = req()
    reserve_prefill_state(request, req_pool, cache).finish(True)
    release_prefill_checkpoint(request, pool)
    release_prefill_checkpoint(request, pool)
    assert pool.available_size() == 1
    native = fork_prefill_checkpoint(req(), pool, torch.tensor([1]))
    assert native is not None and pool.available_size() == 0


@pytest.mark.parametrize('capacity', [0, 1, 4, 5])
def test_broker_reserves_scratch_atomically_and_abort_returns_it(capacity):
    states, req_pool, _ = setup(capacity)
    attention = Pool(128)
    broker = AgenticPWorksetLeaseBroker(64, state_allocators=(states,),
        mamba_req_to_token_pool=req_pool, reserve_mamba_checkpoint=True)
    broker.request('a:0', parent_tokens=64, prompt_tokens=128)
    broker.service(attention)
    lease = broker.get('a:0')
    if capacity < 5:
        assert lease is None
        assert states.available_size() == capacity
        assert attention.available_size() == 128
        return
    assert states.available_size() == 0
    assert broker.begin_bind('a:0', lease)
    broker.commit_parent_bound('a:0', lease, state_duplicate=True)
    request = req()
    broker.attach_runtime_state_for_bind('a:0', request, lease)
    assert request.mamba_ping_pong_track_buffer.numel() == 2
    assert request._agentic_mamba_prefill_checkpoint.numel() == 1
    assert broker.abort_bind('a:0', lease, parent_bound=True)
    assert request._agentic_mamba_prefill_checkpoint is None
    broker.service(attention)
    assert states.available_size() == capacity


def test_evictable_only_capacity_is_rechecked_before_any_partial_alloc():
    pool, req_pool, cache = setup(4)
    old = pool.alloc(4)
    cache.evict = lambda _: pool.free(old)
    reservation = reserve_prefill_state(req(), req_pool, cache)
    assert reservation is not None and pool.available_size() == 0
    reservation.finish(False)
    assert pool.available_size() == 4


@pytest.mark.parametrize('capacity', [4, 5])
def test_first_chunk_needs_replacement_before_first_checkpoint_exists(capacity):
    pool, req_pool, cache = setup(capacity)
    request = req()
    reservation = reserve_prefill_state(request, req_pool, cache, checkpoint_slots=2)
    if capacity == 4:
        assert reservation is None and pool.available_size() == 4
        return
    reservation.finish(True)
    old = fork_prefill_checkpoint(request, pool, request.mamba_pool_idx.reshape(1))
    assert pool.available_size() == 0
    for _ in range(5):
        # Each continuation owns runtime and the next replacement already.
        reserve_prefill_state(request, req_pool, cache).finish(True)
        new = fork_prefill_checkpoint(request, pool, request.mamba_pool_idx.reshape(1))
        pool.free(old)
        request._agentic_mamba_prefill_checkpoint = pool.alloc(1)
        assert request._agentic_mamba_prefill_checkpoint is not None
        old = new
    release_kv_cache(request, cache)
    pool.free(old)
    assert pool.available_size() == 5
