"""Proposed regression gate; run only after the current native group finishes."""
from types import SimpleNamespace
import os

import pytest
import torch
import threading
import time

from sglang.srt.managers.scheduler import AgenticPWorksetLeaseBroker, Scheduler
from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration, SnapshotState
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode_kvcache_offload_manager import DecodeKVCacheOffloadManager


class Allocator:
    def __init__(self, available=64):
        self.available = available
        self.next = 0
        self.freed = []

    def available_size(self):
        return self.available

    def alloc(self, count):
        if count > self.available:
            return None
        result = torch.arange(self.next, self.next + count)
        self.next += count
        self.available -= count
        return result

    def free(self, indices):
        self.freed.append(indices.clone())
        self.available += len(indices)


def request(broker, sid="a:1", *, eligible=True):
    return broker.request(sid, 5, 10, owner=broker.direct_owner(sid),
                          capacity_refusal_eligible=eligible)


def test_shortage_is_full_page_rounded_workset_and_final():
    broker = AgenticPWorksetLeaseBroker(4, capacity_recompute=True)
    allocator = Allocator(15)
    assert request(broker)
    broker.service(allocator)
    refusal = broker.capacity_refusal("a:1")
    assert refusal.required_tokens == 16
    assert refusal.available_tokens == 15
    assert broker.get("a:1") is None
    allocator.available = 64
    assert not request(broker)
    broker.service(allocator)
    assert broker.get("a:1") is None


def test_unserviced_intent_is_not_capacity_evidence():
    broker = AgenticPWorksetLeaseBroker(4, capacity_recompute=True)
    assert request(broker)
    assert broker.get("a:1") is None
    assert broker.capacity_refusal("a:1") is None


def test_slow_tool_or_slow_owner_cannot_capacity_recompute():
    broker = AgenticPWorksetLeaseBroker(4, capacity_recompute=True)
    assert request(broker, eligible=False)
    assert broker.request("b:1", 5, 10, owner=broker.slow_owner("b:1", "req"))
    broker.service(Allocator(0))
    assert broker.capacity_refusal("a:1") is None
    assert broker.capacity_refusal("b:1") is None


def test_native_chunk_reserve_is_counted():
    broker = AgenticPWorksetLeaseBroker(4, capacity_recompute=True)
    assert request(broker)
    broker.service(Allocator(20), reserve_tokens=8)
    refusal = broker.capacity_refusal("a:1")
    assert refusal.available_tokens == 12
    assert refusal.required_tokens == 16


def test_unknown_allocator_failure_is_not_capacity_evidence():
    class UnknownAllocator(Allocator):
        def alloc(self, count):
            return None

    broker = AgenticPWorksetLeaseBroker(4, capacity_recompute=True)
    assert request(broker)
    broker.service(UnknownAllocator(64))
    assert broker.capacity_refusal("a:1") is None


def test_prior_grant_prevents_later_capacity_refusal():
    broker = AgenticPWorksetLeaseBroker(4, capacity_recompute=True)
    allocator = Allocator(16)
    assert request(broker)
    broker.service(allocator)
    lease = broker.get("a:1")
    assert lease is not None
    assert broker.request_release("a:1", lease)
    broker.service(allocator)
    allocator.available = 0
    assert request(broker)
    broker.service(allocator)
    assert broker.capacity_refusal("a:1") is None


def test_default_keeps_existing_ablation_policy():
    broker = AgenticPWorksetLeaseBroker(4)
    assert request(broker)
    allocator = Allocator(0)
    broker.service(allocator)
    assert broker.capacity_refusal("a:1") is None
    allocator.available = 16
    broker.service(allocator)
    assert broker.get("a:1") is not None


def test_cancelled_mirror_retains_pages_until_fence():
    broker = AgenticPWorksetLeaseBroker(4)
    allocator = Allocator(16)
    broker.request("a:1", 5, 10)
    broker.service(allocator)
    lease = broker.get("a:1")
    ready = [False]
    lease.page_mirror_event = SimpleNamespace(query=lambda: ready[0])
    assert broker.request_release("a:1", lease)
    broker.service(allocator)
    assert broker.get("a:1") is lease
    assert not allocator.freed
    ready[0] = True
    broker.service(allocator)
    assert broker.get("a:1") is None
    assert len(allocator.freed) == 1


@pytest.mark.parametrize('stage', ['empty', 'event', 'copy', 'record'])
def test_mirror_failure_never_exposes_page_ids(monkeypatch, caplog, stage):
    class Indices:
        is_cuda = True
        dtype = torch.int64
        shape = (16,)

        def __getitem__(self, _key):
            return self

    class GPUAllocator(Allocator):
        def alloc(self, count):
            self.available -= count
            return Indices()

        def free(self, indices):
            self.freed.append(indices)

    class Host:
        def copy_(self, _indices, non_blocking):
            assert non_blocking
            if stage == 'copy':
                raise RuntimeError('possibly submitted copy')

    class Event:
        def __init__(self):
            if stage == 'event':
                raise RuntimeError('event creation failed before copy')

        def record(self):
            raise RuntimeError('event record failed')

    def allocate(*_args, **_kwargs):
        if stage == 'empty':
            raise RuntimeError('pinned allocation failed')
        return Host()

    monkeypatch.setattr(torch, 'empty', allocate)
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    broker = AgenticPWorksetLeaseBroker(4, capacity_recompute=True)
    allocator = GPUAllocator()
    assert request(broker)
    broker.service(allocator)
    lease = broker.get('a:1')
    assert lease.page_mirror_failed
    assert not broker.prepare_parent_page_indices(lease)
    assert lease.parent_page_indices is None
    assert lease.page_mirror_event is None
    assert not broker.has_capacity_refusal('a:1')
    assert not request(broker)
    broker.service(allocator)
    if stage in ('empty', 'event'):
        assert broker.get('a:1') is None
        assert len(allocator.freed) == 1
    else:
        assert broker.get('a:1') is lease
        assert not allocator.freed
    assert 'direct_page_mirror_failure' in caplog.text


def test_late_mirror_does_not_populate_replacement():
    broker = AgenticPWorksetLeaseBroker(4)
    allocator = Allocator()
    request(broker)
    broker.service(allocator)
    old = broker.get('a:1')
    broker.request_release('a:1', old)
    broker.service(allocator)
    request(broker)
    broker.service(allocator)
    replacement = broker.get('a:1')
    assert replacement.lease_id != old.lease_id
    assert not broker.prepare_parent_page_indices(old)


@pytest.mark.parametrize('outcome', ['missing', 'contention', 'io_error', 'claimed', 'slow', 'success'])
def test_capacity_publication_retries_and_respects_ownership(outcome):
    broker = AgenticPWorksetLeaseBroker(4, capacity_recompute=True)
    generation = RequestGeneration('capacity-publication', 1)
    request(broker, generation.snapshot_id)
    broker.service(Allocator(0))
    state = {
        'claimed': SnapshotState.DIRECT_LOADING,
        'slow': SnapshotState.SLOW_FALLBACK,
    }.get(outcome, SnapshotState.DIRECT_READY)
    manifest = SimpleNamespace(state=state)
    calls = []

    def load(*_args, **_kwargs):
        if outcome == 'io_error':
            raise OSError('transient metadata read failure')
        return None if outcome == 'missing' else manifest

    def terminalize(current, **kwargs):
        calls.append(current)
        assert kwargs['reason'].startswith('direct_workset_capacity_refused ')
        return None if outcome == 'contention' else SimpleNamespace(state=SnapshotState.FAILED)

    result = Scheduler._agentic_publish_direct_capacity_refusal(
        SimpleNamespace(agentic_p_workset_broker=broker), generation,
        SimpleNamespace(load=load, fail_direct_offer=terminalize), 2,
    )
    assert result is (outcome in ('claimed', 'slow', 'success'))
    if outcome in ('claimed', 'slow', 'missing', 'io_error'):
        assert not calls


def test_tp_frozen_plan_does_not_issue_rank_local_refusal():
    broker = AgenticPWorksetLeaseBroker(4, capacity_recompute=True)
    request(broker)
    broker.prepare_tp_plan(0)
    broker.service(Allocator(0))
    assert not broker.has_capacity_refusal('a:1')


def test_matching_pending_intent_can_learn_fast_tool_eligibility():
    broker = AgenticPWorksetLeaseBroker(4, capacity_recompute=True)
    assert request(broker, eligible=False)
    assert request(broker, eligible=True)
    broker.service(Allocator(0))
    assert broker.capacity_refusal('a:1') is not None


@pytest.mark.parametrize('operation', ['prepare', 'release'])
def test_query_failure_quarantines_once(caplog, operation):
    broker = AgenticPWorksetLeaseBroker(4)
    allocator = Allocator()
    request(broker)
    broker.service(allocator)
    lease = broker.get('a:1')
    queries = []

    def query():
        queries.append(1)
        raise RuntimeError('CUDA event query failed')

    lease.page_mirror_submitted = True
    lease.page_mirror_event = SimpleNamespace(query=query)
    lease.parent_page_indices = None
    if operation == 'prepare':
        assert not broker.prepare_parent_page_indices(lease)
    broker.request_release('a:1', lease)
    broker.service(allocator)
    broker.service(allocator)
    assert queries == [1]
    assert broker.get('a:1') is lease
    assert not allocator.freed
    assert 'direct_page_mirror_failure' in caplog.text


def test_mirror_pending_becomes_ready_without_synchronize():
    broker = AgenticPWorksetLeaseBroker(4)
    allocator = Allocator()
    request(broker)
    broker.service(allocator)
    lease = broker.get('a:1')
    ready = [False]
    lease.parent_page_indices = None
    lease.page_mirror_host = torch.arange(lease.parent_allocated_tokens)
    lease.page_mirror_submitted = True
    lease.page_mirror_event = SimpleNamespace(query=lambda: ready[0])
    assert not broker.prepare_parent_page_indices(lease)
    ready[0] = True
    assert broker.prepare_parent_page_indices(lease)
    assert lease.parent_page_indices.tolist() == [0, 1]


def d_manager(state=SnapshotState.DIRECT_READY, *, sent=False):
    sid = 'capacity-d:1'
    manifest = SimpleNamespace(snapshot_id=sid, state=state, token_count=1024,
                               failure_reason='direct_workset_capacity_refused required=2048 available=1024')
    candidate = dict(req=object(), metadata=SimpleNamespace(current=SimpleNamespace(snapshot_id=sid)),
                     manifest=manifest, sent=sent, staging=False, claimed_at=None,
                     sender=SimpleNamespace(poll=lambda: KVPoll.Success if sent else KVPoll.WaitingForInput),
                     created_at=time.monotonic()-3, fallback_retry_at=0.0,
                     io_lock=threading.RLock(), fast_arrival_seen=True,
                     fast_arrival_seen_at=time.monotonic()-2)
    actions = []
    route_ok = [True]

    def route(*_args, **kwargs):
        actions.append(('route', kwargs['route']))
        return route_ok[0]

    manager = SimpleNamespace(
        tp_world_size=1, tp_rank=0, agentic_force_slow_path=False,
        agentic_fast_direct_failure_recompute=False, agentic_direct_capacity_recompute=True,
        agentic_fast_threshold=1.0, agentic_direct_setup_timeout=1.0,
        agentic_relay_worker=None, agentic_early_claim_store=object(), agentic_host_staging_client=object(),
        _agentic_candidate_items=lambda: ((sid, candidate),),
        _agentic_try_final_confirmation=lambda _candidate: False,
        _agentic_candidate_is_live_locked=lambda _sid, value: value is candidate,
        _agentic_try_early_claim=lambda *_args: 'arrived',
        _agentic_direct_manifest=lambda *_args, **_kwargs: manifest,
        _agentic_try_tool_confirmation=lambda *_args: True,
        _agentic_direct_kv_usage=lambda: 0.5,
        _agentic_release_early_claim=lambda *_args: actions.append(('claim_cleanup',)),
        _publish_agentic_route=route,
        _start_agentic_host_staging=lambda *_args: actions.append(('slow',)) or candidate.update(staging=True) or True,
        _cleanup_agentic_direct_sender=lambda *_args: actions.append(('sender_cleanup',)),
        _retire_candidate_for_release=lambda *_args: actions.append(('release',)),
    )
    return manager, candidate, actions, route_ok


def test_capacity_failed_routes_before_source_release_and_retries():
    manager, candidate, actions, route_ok = d_manager(SnapshotState.FAILED)
    route_ok[0] = False
    DecodeKVCacheOffloadManager._check_agentic_direct_progress(manager, progress_relay=False)
    assert actions == [('route', 'recompute')]
    assert candidate['fast_direct_recompute_terminalized']
    route_ok[0] = True
    candidate['fallback_retry_at'] = 0.0
    DecodeKVCacheOffloadManager._check_agentic_direct_progress(manager, progress_relay=False)
    assert actions[-4:] == [('route', 'recompute'), ('claim_cleanup',), ('sender_cleanup',), ('release',)]


@pytest.mark.parametrize('previous_claim', [False, True])
def test_setup_timeout_without_capacity_evidence_is_slow(previous_claim):
    manager, candidate, actions, _route_ok = d_manager()
    candidate['direct_abort_tool_confirmed'] = previous_claim
    DecodeKVCacheOffloadManager._check_agentic_direct_progress(manager, progress_relay=False)
    assert ('slow',) in actions
    assert ('route', 'recompute') not in actions
    assert ('release',) not in actions


def test_capacity_refusal_after_sent_quarantines_source(caplog):
    manager, _candidate, actions, _route_ok = d_manager(SnapshotState.FAILED, sent=True)
    DecodeKVCacheOffloadManager._check_agentic_direct_progress(manager, progress_relay=False)
    assert not actions
    assert 'capacity_refusal_after_send' in caplog.text


@pytest.mark.skipif(os.environ.get('PD_TEST_CUDA_MIRROR') != '1', reason='explicit GPU audit gate')
def test_real_cuda_parent_mirror_and_cancel_fence():
    class CUDAAllocator(Allocator):
        def alloc(self, count):
            indices = torch.arange(self.next, self.next + count, device='cuda:0')
            self.next += count
            self.available -= count
            return indices

        def free(self, indices):
            self.freed.append(indices)
            self.available += indices.numel()

    with torch.cuda.device(0):
        stream = torch.cuda.Stream()
        broker = AgenticPWorksetLeaseBroker(64, capacity_recompute=True)
        allocator = CUDAAllocator(65536)
        for i in range(8):
            sid = f'cuda-mirror:{i}'
            assert broker.request(sid, 7001, 9103, owner=broker.direct_owner(sid),
                                  capacity_refusal_eligible=True)
            with torch.cuda.stream(stream):
                torch.cuda._sleep(10_000_000)
                broker.service(allocator)
            lease = broker.get(sid)
            assert lease.page_mirror_submitted
            assert lease.parent_page_indices is None
            if i % 2 == 0:
                broker.request_release(sid, lease)
                with torch.cuda.stream(stream):
                    broker.service(allocator)
                stream.synchronize()  # Test-owned stream only, never production.
                with torch.cuda.stream(stream):
                    broker.service(allocator)
                assert broker.get(sid) is None
            else:
                stream.synchronize()
                assert broker.prepare_parent_page_indices(lease)
                expected = lease.parent_indices.cpu().numpy()[::64] // 64
                assert (lease.parent_page_indices == expected).all()
                assert str(lease.parent_page_indices.dtype) == 'int32'
                broker.request_release(sid, lease)
                with torch.cuda.stream(stream):
                    broker.service(allocator)
        assert len(allocator.freed) == 8
