"""CPU checks for Direct allocator descriptors and delayed workset grants."""

from collections import deque
from types import SimpleNamespace as NS
import threading
import time

import numpy as np
import pytest
import torch

from sglang.srt.disaggregation import agentic_hybrid_transfer as hybrid
from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration, SnapshotState
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.agentic_tp_events import EventKey
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import AgenticEarlyDirectReceive, AgenticPWorksetLeaseBroker, Scheduler


class Allocator:
    def __init__(self, size):
        self.free_indices = torch.arange(size, dtype=torch.int64)

    def available_size(self):
        return self.free_indices.numel()

    def alloc(self, count):
        if count > self.available_size():
            return None
        result = self.free_indices[:count]
        self.free_indices = self.free_indices[count:]
        return result

    def free(self, indices):
        self.free_indices = torch.cat((self.free_indices, indices))


def test_compact_descriptor_has_one_blocking_copy_and_owns_cpu_data(monkeypatch):
    parent = torch.tensor([8, 9, 10, 11, 24, 25, 26, 27])
    slot = torch.tensor([17])
    copies = []
    original = torch.Tensor.cpu

    def cpu(tensor, *args, **kwargs):
        copies.append(tensor.numel())
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    pages, slots = hybrid.prepare_workset_transfer_indices(parent, (slot,), 4)
    assert copies == [3]  # Two page starts and one Mamba slot, not eight tokens.
    parent.fill_(0)
    slot.fill_(0)
    assert pages.tolist() == [2, 6]
    assert pages.dtype == np.int32
    assert not pages.flags.writeable
    assert slots == ((17,),)
    with pytest.raises(ValueError):
        pages[0] = 0


def test_cached_hybrid_metadata_does_not_read_gpu_indices(monkeypatch):
    pages, slots = hybrid.prepare_workset_transfer_indices(
        torch.arange(8), (torch.tensor([9]),), 4
    )
    lease = NS(parent_page_indices=pages, state_device_indices=(torch.tensor([9]),),
               state_cpu_indices=slots)
    sent = []
    monkeypatch.setattr(torch.Tensor, "cpu", lambda *_a, **_k: pytest.fail("worker GPU read"))
    hybrid.submit_reverse_receive(NS(send_metadata=lambda *a, **k: sent.append((a, k))),
                                  lease, (hybrid.StateType.MAMBA,))
    assert sent[0][0][0].tolist() == [0, 1]
    assert sent[0][1] == {"aux_index": 0, "state_indices": [9]}
    # A cached descriptor is not authority to reuse a released/handed slot.
    lease.state_device_indices = ()
    with pytest.raises(RuntimeError, match="missing its Mamba slot"):
        hybrid.state_indices_for_workset(lease, (hybrid.StateType.MAMBA,))


def test_legacy_workset_keeps_state_mirror():
    lease = NS(state_device_indices=(torch.tensor([23]),))
    assert hybrid.state_indices_for_workset(lease, (hybrid.StateType.MAMBA,)) == [[23]]


@pytest.mark.parametrize("parent,slots,page_size", [
    (torch.arange(7), (torch.tensor([1]),), 4),
    (torch.arange(8), (torch.tensor([1, 2]),), 4),
    (torch.arange(8), (), 0),
])
def test_invalid_descriptor_is_not_published(parent, slots, page_size):
    with pytest.raises(ValueError):
        hybrid.prepare_workset_transfer_indices(parent, slots, page_size)


@pytest.mark.parametrize("tp,socket", [(False, True), (True, False), (True, True)])
def test_broker_publishes_descriptor_only_after_allocator_fence(monkeypatch, tp, socket):
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "tcp://test:1" if socket else "")
    allocator, state = Allocator(64), Allocator(8)
    broker = AgenticPWorksetLeaseBroker(4, state_allocators=(state,))
    broker.request("snapshot", 8, 12, owner=broker.direct_owner("snapshot"))
    if tp:
        broker.prepare_tp_control(1)
    original = hybrid.prepare_workset_transfer_indices
    copies = []

    def prepare(*args):
        assert broker.get("snapshot") is None
        assert broker.drain_grant_events() == ()
        result = original(*args)
        copies.append(result)
        return result

    monkeypatch.setattr(hybrid, "prepare_workset_transfer_indices", prepare)
    broker.service(allocator)
    lease = broker.get("snapshot")
    assert (lease.state_cpu_indices is not None) == (tp and socket)
    assert len(copies) == int(tp and socket)
    assert lease.parent_page_indices.tolist() == [0, 1]
    assert lease.intent_at <= lease.grant_at <= time.monotonic()
    assert broker._intent_requested_at == {}
    assert broker.drain_grant_events() == ("snapshot",)
    assert allocator.available_size() == 52
    assert state.available_size() == 6


def test_descriptor_failure_rolls_back_complete_unpublished_allocation(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "tcp://test:1")
    allocator, state = Allocator(64), Allocator(8)
    broker = AgenticPWorksetLeaseBroker(4, state_allocators=(state,))
    broker.request("snapshot", 8, 12)
    broker.prepare_tp_control(1)

    def fail(*args):
        raise RuntimeError("index fence failed")

    monkeypatch.setattr(hybrid, "prepare_workset_transfer_indices", fail)
    with pytest.raises(RuntimeError, match="index fence failed"):
        broker.service(allocator)
    assert broker.get("snapshot") is None
    assert broker.drain_grant_events() == ()
    assert allocator.available_size() == 64
    assert state.available_size() == 8
    assert sorted(allocator.free_indices.tolist()) == list(range(64))
    assert sorted(state.free_indices.tolist()) == list(range(8))


@pytest.mark.parametrize("tp_size,socket,expected_pending", [
    (1, True, 32), (2, False, 32), (2, True, 4),
])
def test_direct_burst_respects_cap_before_native_allocator_grants(
    monkeypatch, tp_size, socket, expected_pending
):
    monkeypatch.setenv("SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS", "0")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_DIRECT_IO_CAP", "4")
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "tcp://test:1" if socket else "")
    broker, allocator = AgenticPWorksetLeaseBroker(4), Allocator(512)
    arrivals = []
    manifests = {}
    for index in range(32):
        request = RequestGeneration(f"delayed-grant-{index}", 1)
        manifest = NS(request=request, state=SnapshotState.DIRECT_READY,
                      created_at=time.time(), token_count=8, direct_room=12)
        manifests[request.snapshot_id] = manifest
        arrivals.append((request, {"arrived_at": time.time(), "prompt_token_count": 12}, manifest))
    scheduler = NS(
        tp_size=tp_size, tp_rank=0, agentic_early_claim_store=object(),
        agentic_early_direct_admission_queue=deque(arrivals),
        agentic_early_direct_admission_ids=set(manifests),
        agentic_early_direct_receives={}, agentic_early_direct_terminal={},
        agentic_tp_direct_admission_active={}, agentic_p_workset_broker=broker,
        agentic_tp_direct_local_failed=set(),
        agentic_tp_direct_mailbox=NS(receipt=lambda *_a: None,
                                    _key=lambda sid: EventKey(sid, "direct-room:12"),
                                    publish_receipt=lambda *_a: None),
    )
    store = NS(load=lambda request, **_k: manifests[request.snapshot_id])
    lock = threading.RLock()
    scheduler.agentic_early_direct_poll_lock = lock
    Scheduler._agentic_admit_queued_direct_receives(scheduler, store, 1.0, lock)
    assert len(broker.direct_admission_snapshot_ids()) == expected_pending
    assert not broker._leases
    if expected_pending == 32:
        # TP1 and legacy TP preserve their original delayed-allocation behavior:
        # pending intents alone do not spend the local physical/grant budget,
        # either within the same pass or on the next worker pass.
        assert Scheduler._agentic_early_direct_slots_used(scheduler) == 0
        Scheduler._agentic_admit_queued_direct_receives(scheduler, store, 1.0, lock)
        assert len(broker._intents) == 32
        assert Scheduler._agentic_early_direct_slots_used(scheduler) == 0
        return
    assert Scheduler._agentic_early_direct_slots_used(scheduler) == 4
    broker.prepare_tp_control(1)
    broker.service(allocator)
    assert len(broker._leases) == 4
    # The selected four must consume their own slots, not deadlock at the cap.
    Scheduler._agentic_admit_queued_direct_receives(scheduler, store, 1.0, lock)
    assert len(scheduler.agentic_tp_direct_admission_active) == 4
    assert len(scheduler.agentic_early_direct_admission_queue) == 28
    assert Scheduler._agentic_early_direct_slots_used(scheduler) == 4
    completed = arrivals[0][0].snapshot_id
    scheduler.agentic_early_direct_receives[completed] = NS(
        completed_at=time.monotonic(), transport_poll=KVPoll.Success
    )
    assert Scheduler._agentic_early_direct_slots_used(scheduler) == 4  # Local completion alone is not all-rank.
    Scheduler._agentic_return_direct_transport_credit(
        scheduler, completed, scheduler.agentic_tp_direct_admission_active[completed])
    assert Scheduler._agentic_early_direct_slots_used(scheduler) == 3
    Scheduler._agentic_admit_queued_direct_receives(scheduler, store, 1.0, lock)
    assert len(broker._intents) == 1
    assert Scheduler._agentic_early_direct_slots_used(scheduler) == 4


def test_lagging_follower_executes_leader_grant_above_local_occupancy(monkeypatch):
    """Leader may free its lane while a follower still has four shard DMAs."""
    monkeypatch.setenv("SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS", "0")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_DIRECT_IO_CAP", "4")
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "tcp://test:1")
    broker, allocator = AgenticPWorksetLeaseBroker(4), Allocator(512)
    requests = [RequestGeneration(f"rank-skew-{index}", 1) for index in range(5)]
    for request in requests:
        broker.request(request.snapshot_id, 8, 12,
                       owner=broker.direct_owner(request.snapshot_id))
    # The fifth workset is part of rank0's committed native allocator plan,
    # not an independently selected follower request.
    broker.prepare_tp_control(1)
    broker.service(allocator)
    request = requests[-1]
    manifest = NS(request=request, state=SnapshotState.DIRECT_LOADING,
                  created_at=time.time(), token_count=8)
    receipts = {request.snapshot_id: 1}
    scheduler = NS(
        tp_size=8, tp_rank=1, agentic_early_claim_store=object(),
        agentic_early_direct_admission_queue=deque([
            (request, {"arrived_at": time.time(), "prompt_token_count": 12}, manifest)
        ]),
        agentic_early_direct_admission_ids={request.snapshot_id},
        agentic_early_direct_receives={old.snapshot_id: NS(
            completed_at=None, transport_poll=KVPoll.Transferring
        ) for old in requests[:-1]},
        agentic_early_direct_terminal={}, agentic_tp_direct_admission_active={},
        agentic_tp_direct_local_failed=set(), agentic_tp_direct_local_admitted=set(),
        agentic_p_workset_broker=broker,
        agentic_tp_direct_mailbox=NS(receipt=lambda sid: receipts.get(sid)),
    )
    started = []
    scheduler._agentic_tp_start_direct_shard = lambda req, **_kw: started.append(req)
    assert Scheduler._agentic_early_direct_slots_used(scheduler) == 5
    store = NS(load=lambda *_a, **_k: manifest)
    Scheduler._agentic_admit_queued_direct_receives(scheduler, store, 1.0, threading.RLock())
    assert request.snapshot_id in scheduler.agentic_tp_direct_admission_active
    Scheduler._agentic_progress_tp_direct_grants(scheduler, store)
    assert started == [request]
    assert not scheduler.agentic_early_direct_admission_queue


def test_cancelled_intent_does_not_retain_timing_or_direct_slot():
    broker = AgenticPWorksetLeaseBroker(4)
    owner = broker.direct_owner("snapshot")
    broker.request("snapshot", 8, 12, owner=owner)
    assert broker.direct_admission_snapshot_ids() == {"snapshot"}
    assert broker.cancel_unstarted("snapshot", owner=owner)
    assert not broker.direct_admission_snapshot_ids()
    assert broker._intent_requested_at == {}


def test_native_admit_and_later_tp_clear_do_not_resurrect_completed_lane(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "tcp://test:1")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_DIRECT_IO_CAP", "1")
    monkeypatch.setenv("SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS", "0")
    broker, allocator = AgenticPWorksetLeaseBroker(4), Allocator(64)
    request = RequestGeneration("admitted-awaiting-group-five", 1)
    sid = request.snapshot_id
    broker.request(sid, 8, 12, owner=broker.direct_owner(sid))
    broker.prepare_tp_control(1)
    broker.service(allocator)
    lease = broker.get(sid)
    assert broker.begin_io_attempt(sid, lease, "attempt")
    broker.mark_io_inflight(sid, lease, "attempt")
    assert broker.mark_io_quiesced(sid, lease, "attempt")
    assert broker.begin_bind(sid, lease)
    broker.commit_parent_bound(sid, lease)
    manifest = NS(request=request, token_count=8, direct_room=42)
    entry = AgenticEarlyDirectReceive(
        request, manifest, "attempt", None, lease.parent_indices,
        time.monotonic(), time.time(), workset_lease=lease,
        completed_at=time.monotonic(), transport_poll=KVPoll.Success,
    )
    owner = Scheduler.__new__(Scheduler)
    owner.tp_size, owner.tp_rank = 8, 0
    owner.disaggregation_mode = DisaggregationMode.PREFILL
    owner.agentic_p_workset_broker = broker
    owner.agentic_early_direct_receives = {sid: entry}
    owner.agentic_tp_direct_admission_active = {sid: (request, time.time(), None, 12, lease)}
    owner.agentic_tp_direct_local_admitted = set()
    owner.agentic_tp_direct_local_failed = set()
    owner.agentic_early_direct_terminal = {}
    owner.agentic_tp_direct_group_status = {}
    owner.agentic_tp_direct_mailbox = NS(
        _key=lambda sid: EventKey(sid, "direct-room:42"),
        receipt=lambda *_a: None,
        publish_receipt=lambda *_a: None, clear_local=lambda *_a: None,
        clear_local_rollback=lambda *_a: None, clear_group=lambda *_a: None,
        clear_group_rollback=lambda *_a: None,
    )
    class AdmissionLock:
        def __init__(self):
            self.lock = threading.RLock()
        def __enter__(self):
            self.lock.acquire()
        def __exit__(self, *_args):
            # Inspect the exact publication boundary, not merely final state.
            if sid not in owner.agentic_early_direct_receives and sid in owner.agentic_tp_direct_admission_active:
                assert sid in owner.agentic_tp_direct_local_admitted
            self.lock.release()
    owner.agentic_early_direct_poll_lock = AdmissionLock()
    # Native bind is reachable only after the existing all-rank group>=3 ACK.
    owner._agentic_return_direct_transport_credit(sid, owner.agentic_tp_direct_admission_active[sid])
    assert owner._agentic_early_direct_slots_used() == 0
    req = NS(rid="native-child", origin_input_ids=list(range(12)))
    assert owner._agentic_admit_early_direct_bind(req, request, entry, tp_size=8, marker_store=None) is False
    assert entry.workset_lease is None and lease.state == "handed"
    assert req._agentic_p_workset_lease is lease
    assert sid in owner.agentic_tp_direct_admission_active  # Followers can still report 4.
    assert owner._agentic_early_direct_slots_used() == 0

    new_request = RequestGeneration("new-arrival-during-group-five", 1)
    new_manifest = NS(request=new_request, state=SnapshotState.DIRECT_READY,
                      token_count=8, created_at=time.time(), direct_room=12)
    owner.agentic_early_claim_store = object()
    owner.agentic_early_direct_admission_queue = deque([
        (new_request, {"arrived_at": time.time(), "prompt_token_count": 12}, new_manifest)
    ])
    owner.agentic_early_direct_admission_ids = {new_request.snapshot_id}
    owner._agentic_admit_queued_direct_receives(
        NS(load=lambda *_a, **_kw: new_manifest), 1.0, owner.agentic_early_direct_poll_lock)
    assert new_request.snapshot_id in broker._intents
    assert owner._agentic_early_direct_slots_used() == 1
    # Execute the real native clear after group5, without touching Req pages.
    plan = broker.prepare_tp_control(2)[0]
    command = {Scheduler._AGENTIC_TP_CONTROL_KEY: True, "workset_plan_epoch": 2,
               "workset_allocation_plan": plan, "direct_commands": [
                   {"snapshot": sid, "action": "clear"}], "host_commands": []}
    assert owner._agentic_tp_consume_admission_control([command]) == []
    assert sid not in owner.agentic_tp_direct_admission_active
    assert sid not in owner.agentic_tp_direct_local_admitted
    assert owner._agentic_early_direct_slots_used() == 1
    assert req._agentic_p_workset_lease is lease and lease.state == "handed"
    assert allocator.available_size() == 52


@pytest.mark.parametrize("case", ["intent", "unstarted_grant", "pending_claim", "unfenced_cancel", "failed_grant"])
def test_ghost_slot_fix_keeps_pending_and_unfenced_attempts_counted(monkeypatch, case):
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "tcp://test:1")
    broker, allocator = AgenticPWorksetLeaseBroker(4), Allocator(64)
    request = RequestGeneration(case, 1)
    sid = request.snapshot_id
    broker.request(sid, 8, 12, owner=broker.direct_owner(sid))
    lease = None
    if case != "intent":
        broker.prepare_tp_control(1)
        broker.service(allocator)
        lease = broker.get(sid)
    if case in {"pending_claim", "unfenced_cancel"}:
        assert broker.begin_io_attempt(sid, lease, "attempt")
    receives = {}
    if case == "unfenced_cancel":
        broker.mark_io_inflight(sid, lease, "attempt")
        broker.request_release(sid, lease, io_attempt="attempt")
        receives[sid] = NS(completed_at=None, transport_poll=KVPoll.WaitingForInput,
                           abort_requested=True)
        assert lease.state == "release_pending"
    owner = NS(tp_size=8, agentic_p_workset_broker=broker,
               agentic_early_direct_receives=receives,
               agentic_tp_direct_admission_active={} if lease is None else {sid: (request, 0, None, 12, lease)},
               agentic_tp_direct_local_admitted=set(),
               agentic_tp_direct_local_failed={sid} if case in {"unfenced_cancel", "failed_grant"} else set())
    assert Scheduler._agentic_early_direct_slots_used(owner) == 1
    assert allocator.available_size() == (64 if lease is None else 52)


@pytest.mark.parametrize("tp_size,socket,expected", [(1, True, 1), (8, False, 1), (8, True, 0)])
def test_admitted_cleanup_exemption_is_socket_tp_only(monkeypatch, tp_size, socket, expected):
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "tcp://test:1" if socket else "")
    owner = NS(tp_size=tp_size, agentic_early_direct_receives={},
               agentic_tp_direct_admission_active={"done": object()},
               agentic_tp_direct_local_admitted={"done"},
               agentic_p_workset_broker=AgenticPWorksetLeaseBroker(4),
               agentic_tp_direct_mailbox=NS(_key=lambda sid: EventKey(sid, "direct-room:12")),
               agentic_tp_direct_transport_returned={"done": EventKey("done", "direct-room:12")})
    assert Scheduler._agentic_early_direct_slots_used(owner) == expected
