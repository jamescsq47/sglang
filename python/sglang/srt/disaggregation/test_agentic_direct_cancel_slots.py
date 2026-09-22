"""Cancelled unposted work keeps HBM ownership, not a transport admission."""

from collections import deque
from types import SimpleNamespace as NS
import time

import pytest

from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration, SnapshotState
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.test_agentic_direct_descriptors import Allocator
from sglang.srt.disaggregation.test_agentic_direct_retirement import scheduler, workset
from sglang.srt.managers.scheduler import AgenticPWorksetLeaseBroker


@pytest.fixture(autouse=True)
def socket_tp(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "tcp://test")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_DIRECT_IO_CAP", "1")
    monkeypatch.setenv("SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS", "0")


def test_stale_cached_offer_returns_admission_but_keeps_exact_hbm_until_retirement():
    old, fresh = RequestGeneration("expired-no-grant", 1), RequestGeneration("fresh", 1)
    broker, allocator, lease = workset(old)
    owner = scheduler(old, broker, None)
    owner.agentic_tp_direct_admission_active = {}  # No receipt ever authorized I/O.
    owner.agentic_early_claim_store = object()
    now = time.time()
    cached = NS(state=SnapshotState.DIRECT_READY, created_at=now - 2, token_count=8, direct_room=12)
    current = NS(state=SnapshotState.DIRECT_READY, created_at=now - 1, token_count=8, direct_room=12)
    owner.agentic_tp_direct_mailbox = NS(receipt=lambda _sid: None,
                                       publish_receipt=lambda *_a: None)
    owner.agentic_early_direct_admission_queue = deque([
        (old, {"arrived_at": now - 1.1, "prompt_token_count": 12}, cached),
        (fresh, {"arrived_at": now - .1, "prompt_token_count": 12}, current),
    ])
    owner.agentic_early_direct_admission_ids = {old.snapshot_id, fresh.snapshot_id}
    store = NS(load=lambda request, **_kw: (
        NS(state=SnapshotState.SLOW_FALLBACK) if request == old else current
    ))
    assert owner._agentic_early_direct_slots_used() == 1
    owner._agentic_admit_queued_direct_receives(store, 1, owner.agentic_early_direct_poll_lock)
    assert list(owner.agentic_early_direct_admission_ids) == [fresh.snapshot_id]
    assert owner.agentic_tp_direct_admission_active == {}
    assert owner._agentic_early_direct_slots_used() == 0
    assert lease.state == "active" and lease.io_attempt is None
    assert broker.get(old.snapshot_id) is lease and allocator.available_size() == 52
    assert broker.owner_has_unretired_work(old.snapshot_id, owner=lease.owner)
    assert not broker.begin_io_attempt(old.snapshot_id, lease, "late-claim")

    # The next normal worker pass can select fresh work. No deadline/cap or
    # allocator behavior changes, and selection itself allocates no pages.
    owner._agentic_admit_queued_direct_receives(store, 1, owner.agentic_early_direct_poll_lock)
    assert fresh.snapshot_id in broker._intents
    assert owner._agentic_early_direct_slots_used() == 1
    assert allocator.available_size() == 52
    plan, retiring, _ = broker.prepare_tp_control(2)
    assert {entry[0] for entry in plan} == {old.snapshot_id, fresh.snapshot_id}
    assert retiring == (old.snapshot_id,)
    broker.service(allocator)
    assert broker.get(old.snapshot_id) is lease  # Local slot release is not TP free.
    assert allocator.available_size() == 40
    assert broker.commit_tp_retire(old.snapshot_id)
    broker.service(allocator)
    assert allocator.available_size() == 52
    assert broker.get(fresh.snapshot_id).state == "active"


@pytest.mark.parametrize("tp_size", [2, 8])
def test_cancel_after_freeze_never_authorizes_io_on_late_plan_materialization(tp_size):
    request = RequestGeneration("late-frozen-plan", 1)
    sid = request.snapshot_id
    brokers = [AgenticPWorksetLeaseBroker(4) for _ in range(tp_size)]
    allocators = [Allocator(64) for _ in brokers]
    brokers[0].request(sid, 8, 12, owner=brokers[0].direct_owner(sid))
    plan = brokers[0].prepare_tp_plan(1)
    assert brokers[0].direct_admission_snapshot_ids() == {sid}
    assert brokers[0].cancel_unstarted(sid, owner=brokers[0].direct_owner(sid))
    assert brokers[0].direct_admission_snapshot_ids() == set()
    brokers[0].install_tp_plan(1, plan)  # Same-epoch replay cannot clear cancellation.
    for broker in brokers[1:]:
        broker.install_tp_plan(1, plan)
    for broker, allocator in zip(brokers, allocators):
        broker.service(allocator)
        assert allocator.available_size() == 52
    assert not brokers[0].begin_io_attempt(sid, brokers[0].get(sid), "late")
    assert brokers[0].direct_admission_snapshot_ids() == set()
    # Followers do not invent a cancellation before the authoritative native
    # broadcast. Only leader capacity accounting admits new logical work.
    assert all(b.direct_admission_snapshot_ids() == {sid} for b in brokers[1:])
    next_plan, retiring, _ = brokers[0].prepare_tp_control(2)
    for broker in brokers[1:]:
        broker.install_tp_plan(2, next_plan, retiring_ids=retiring)
    for broker, allocator in zip(brokers, allocators):
        assert broker.direct_admission_snapshot_ids() == set()
        assert not broker.begin_io_attempt(sid, broker.get(sid), "late")
        broker.service(allocator)
        assert allocator.available_size() == 52
    assert all(b.tp_retire_ready(sid) for b in brokers)
    for broker, allocator in zip(brokers, allocators):
        assert broker.commit_tp_retire(sid)
        broker.service(allocator)
        assert allocator.available_size() == 64


@pytest.mark.parametrize("retirement", ["cancel", "release", "native"])
@pytest.mark.parametrize("reserved", [False, True])
def test_cancellation_slot_proof_uses_same_locked_gate_as_begin_io(retirement, reserved):
    request = RequestGeneration("cancel-race", 1)
    sid = request.snapshot_id
    broker, allocator, lease = workset(request)
    owner = scheduler(request, broker, None)
    owner.agentic_tp_direct_admission_active = {}
    if reserved:
        assert broker.begin_io_attempt(sid, lease, "pending-claim")
    if retirement == "cancel":
        broker.cancel_unstarted(sid, owner=lease.owner)
    elif retirement == "release":
        broker.request_release(sid, lease, io_attempt="pending-claim" if reserved else None)
    else:
        broker.prepare_tp_retire(sid)
    assert owner._agentic_early_direct_slots_used() == int(reserved)
    assert not broker.begin_io_attempt(sid, lease, "late")
    assert broker.owner_has_unretired_work(sid, owner=lease.owner)
    assert allocator.available_size() == 52
    if reserved:
        assert lease.state == "io_reserved" and lease.io_attempt == "pending-claim"
        # A late successful/failed RPC must first settle the exact attempt.
        assert not broker.cancel_io_attempt(sid, lease, "old-attempt")
        assert owner._agentic_early_direct_slots_used() == 1
        assert broker.cancel_io_attempt(sid, lease, "pending-claim")
        assert owner._agentic_early_direct_slots_used() == 0


@pytest.mark.parametrize("receiver", [False, True])
def test_posted_unfenced_and_only_locally_rolled_back_grants_keep_slots(receiver):
    request = RequestGeneration("posted", 1)
    sid = request.snapshot_id
    broker, allocator, lease = workset(request)
    assert broker.begin_io_attempt(sid, lease, "physical")
    broker.mark_io_inflight(sid, lease, "physical")
    broker.prepare_tp_retire(sid)
    owner = scheduler(request, broker, lease)
    owner.agentic_tp_direct_local_failed.add(sid)
    owner.agentic_tp_direct_local_rolled_back = {sid}
    if receiver:
        owner.agentic_early_direct_receives[sid] = NS(
            completed_at=None, transport_poll=KVPoll.Transferring, abort_requested=True
        )
    assert lease.state == "release_pending"
    assert owner._agentic_early_direct_slots_used() == 1
    assert not broker.cancel_io_attempt(sid, lease, "physical")
    assert not broker.commit_tp_retire(sid)
    assert allocator.available_size() == 52
    if not receiver:
        assert broker.mark_io_quiesced(sid, lease, "physical")
        # This rank being quiet is not proof that every peer DMA completed.
        assert owner._agentic_early_direct_slots_used() == 1


@pytest.mark.parametrize("kind", ["intent", "active", "reserved"])
def test_normal_intents_and_failed_flags_do_not_bypass_original_cap(kind):
    request = RequestGeneration(kind, 1)
    sid = request.snapshot_id
    broker, allocator = AgenticPWorksetLeaseBroker(4), Allocator(64)
    broker.request(sid, 8, 12, owner=broker.direct_owner(sid))
    lease = None
    if kind != "intent":
        broker.prepare_tp_control(1)
        broker.service(allocator)
        lease = broker.get(sid)
    if kind == "reserved":
        broker.begin_io_attempt(sid, lease, "pending")
    owner = scheduler(request, broker, lease)
    owner.agentic_tp_direct_admission_active = {}
    owner.agentic_tp_direct_local_failed.add(sid)  # A flag is not physical proof.
    assert owner._agentic_early_direct_slots_used() == 1
    assert not broker.cancel_unstarted(sid, owner="different-owner")
    assert owner._agentic_early_direct_slots_used() == 1


@pytest.mark.parametrize("tp_size,socket", [(1, True), (8, False)])
def test_tp1_and_legacy_grant_accounting_is_unchanged(monkeypatch, tp_size, socket):
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "tcp://test" if socket else "")
    request = RequestGeneration("legacy", 1)
    broker, _, lease = workset(request, tp_size)
    owner = scheduler(request, broker, lease, tp_size)
    broker.cancel_unstarted(request.snapshot_id, owner=lease.owner)
    assert owner._agentic_early_direct_slots_used() == 1


@pytest.mark.parametrize("state", ["binding", "handed", "consumed"])
def test_readonly_cancelled_slot_accounting_never_frees_bound_or_native_owned_pages(state):
    request = RequestGeneration("native-owned", 1)
    sid = request.snapshot_id
    broker, allocator, lease = workset(request)
    assert broker.begin_bind(sid, lease)
    broker.commit_parent_bound(sid, lease)
    req = NS(origin_input_ids=list(range(12)))
    if state in {"handed", "consumed"}:
        broker.handoff_to_req(sid, req, lease)
    if state == "consumed":
        broker.consume_suffix(lease, 4, final_prompt_chunk=True)
    owner = scheduler(request, broker, None)
    owner.agentic_tp_direct_admission_active = {}
    for _ in range(2):
        assert owner._agentic_early_direct_slots_used() == 0
        assert allocator.available_size() == 52
        assert lease.state == state
