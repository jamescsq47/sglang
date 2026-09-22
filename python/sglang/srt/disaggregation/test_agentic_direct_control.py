"""Direct control futures against real broker CAS semantics, without CUDA."""

from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.disaggregation import agentic_direct_control as control
from sglang.srt.disaggregation import agentic_lifecycle_control as lifecycle
from sglang.srt.disaggregation.agentic_control_store import BrokerRawStore, MemoryControlStore
from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    MooncakeSnapshotStore, RequestGeneration, SnapshotManifest,
    SnapshotNotReadyError, SnapshotState,
)
from sglang.srt.disaggregation.agentic_tp_events import ControlUnavailable, EventKey
from sglang.srt.managers.scheduler import AgenticEarlyDirectReceive, AgenticPWorksetLeaseBroker, Scheduler


class CheckedFuture(Future):
    def result(self, *args, **kwargs):
        assert self.done(), "Direct must never block on a control reply"
        return super().result(*args, **kwargs)


class DeferredClient:
    def __init__(self):
        self.records = MemoryControlStore(lambda *_a: None)
        self.queue = []
        self.history = []

    def submit(self, service, method, *args):
        assert service == "records"
        future = CheckedFuture()
        self.queue.append((future, method, args))
        self.history.append((method, args))
        return future

    def execute(self, *, ack=True):
        future, method, args = self.queue.pop(0)
        try:
            value = getattr(self.records, method)(*args)
        except Exception as exc:
            future.set_exception(exc)
            return future, None
        if ack:
            future.set_result(value)
        return future, value


class Records:
    def __init__(self, client, namespace):
        self.client, self.namespace = client, namespace

    def check(self):
        pass

    def get(self, key):
        return self.client.records._data.get(self.namespace, {}).get(str(key))

    def call(self, method, key, *args):
        return getattr(self.client.records, method)(self.namespace, str(key), *args)


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "tcp://test")
    monkeypatch.setenv("SGLANG_PD_P_READY_DIR", "direct-test")
    client = DeferredClient()
    fences = Records(client, "fences")
    monkeypatch.setattr(lifecycle, "lifecycle_records", lambda: fences)
    raw = BrokerRawStore.__new__(BrokerRawStore)
    raw.records = Records(client, "snapshots")
    store = MooncakeSnapshotStore(raw)
    request = RequestGeneration("async-direct", 1)
    manifest = SnapshotManifest(request, (), 8, 128, SnapshotState.DIRECT_READY,
                                token_digest="digest", direct_bootstrap_addr="test:1",
                                direct_room=12, tp_size=8)
    raw.put(manifest.manifest_key, manifest.to_bytes())
    return NS(client=client, fences=fences, raw=raw, store=store,
              request=request, manifest=manifest, claim="direct-early-tp:p:async-direct:1")


def operation(setup, function, *args):
    return control.DirectControlCall(
        (setup.request.snapshot_id, setup.manifest.direct_room, 3, setup.claim),
        function, *args,
    )


def receive_entry(manifest, *, lease_id=3, **kwargs):
    return AgenticEarlyDirectReceive(
        request=manifest.request, manifest=manifest, claim_id=manifest.claim_id,
        receiver=None, device_indices=None, started_at=1.0, arrived_at=1.0,
        workset_lease=NS(lease_id=lease_id), **kwargs,
    )


def finish(setup, call, *, cancel=False):
    for _ in range(20):
        ready, result = call.poll(cancel=cancel)
        if ready:
            return result
        assert setup.client.queue, "unexpected unresolvable mirror wait"
        setup.client.execute()
    pytest.fail("control call did not finish")


def claim(setup):
    return finish(setup, operation(setup, control.claim_direct,
                                  setup.store, setup.request, setup.claim, 12))


def test_claim_completion_and_bound_keep_original_fences(setup):
    claimed = claim(setup)
    assert claimed.state is SnapshotState.DIRECT_LOADING
    assert setup.fences.get(setup.store._local_claim_path(setup.request)) == setup.claim
    assert setup.raw.get(setup.request.claim_key) == f"direct:{setup.claim}".encode()
    received = finish(setup, operation(setup, control.complete_direct,
                                      setup.store, claimed, setup.claim))
    assert received.state is SnapshotState.P_RECEIVED
    terminal = finish(setup, operation(setup, control.commit_bound,
                                      setup.store, received, setup.claim))
    assert terminal.state is SnapshotState.CONSUMED
    assert setup.store.load(setup.request, require_ready=False).state is SnapshotState.CONSUMED
    assert setup.fences.get(setup.store._local_claim_path(setup.request)) is None
    assert not setup.raw.get(setup.request.claim_key)
    assert [method for method, _ in setup.client.history] == [
        "claim", "put", "upsert_if_owner", "upsert_if_owner",
        "upsert_if_owner", "remove", "remove",
    ]


def test_follower_joins_same_claim_without_sleep_or_another_manifest_write(setup):
    claimed = claim(setup)
    count = len(setup.client.history)
    assert claim(setup) == claimed
    assert [method for method, _ in setup.client.history[count:]] == ["claim"]


def test_fallback_owner_rejects_claim_without_mutating_it(setup):
    path = setup.store._local_claim_path(setup.request)
    setup.fences.call("claim", path, "fallback:d")
    with pytest.raises(SnapshotNotReadyError):
        claim(setup)
    assert setup.fences.get(path) == "fallback:d"
    assert setup.store.load(setup.request, require_ready=False).state is SnapshotState.DIRECT_READY
    assert not setup.raw.get(setup.request.claim_key)


@pytest.mark.parametrize("pending_stage", [0, 1, 2])
def test_cancel_drains_late_claim_reply_without_submitting_following_transition(setup, pending_stage):
    call = operation(setup, control.claim_direct, setup.store, setup.request, setup.claim, 12)
    for _ in range(pending_stage):
        assert call.poll()[0] is False
        setup.client.execute()
    assert call.poll()[0] is False
    future, result = setup.client.execute(ack=False)
    assert call.poll(cancel=True) == (False, None)
    assert not call.settled
    future.set_result(result)
    assert call.poll(cancel=True)[0]
    assert call.settled and not setup.client.queue
    current = setup.store.load(setup.request, require_ready=False)
    if pending_stage < 2:
        assert current.state is SnapshotState.DIRECT_READY
        # Only after the scheduler's all-rank rollback barrier may rank0
        # remove the acquired pre-manifest fence.
        cleanup = operation(setup, control.release_unstarted_claim,
                            setup.store, setup.request, setup.claim)
        finish(setup, cleanup)
        assert setup.fences.get(setup.store._local_claim_path(setup.request)) is None
    else:
        assert current.state is SnapshotState.DIRECT_LOADING
        assert current.claim_id == setup.claim  # Original group abort owns release.


def test_unknown_claim_outcome_cannot_retire_attempt(setup):
    call = operation(setup, control.claim_direct, setup.store, setup.request, setup.claim, 12)
    call.poll()
    future, _, _ = setup.client.queue.pop()
    future.set_exception(ControlUnavailable("reply lost"))
    with pytest.raises(ControlUnavailable):
        call.poll(cancel=True)
    assert not call.settled


def test_bound_cancel_after_consumed_cas_finishes_only_terminal_cleanup(setup):
    claimed = claim(setup)
    received = finish(setup, operation(setup, control.complete_direct, setup.store, claimed, setup.claim))
    call = operation(setup, control.commit_bound, setup.store, received, setup.claim)
    call.poll()
    future, value = setup.client.execute(ack=False)
    assert setup.store.load(setup.request, require_ready=False).state is SnapshotState.CONSUMED
    assert call.poll(cancel=True) == (False, None)
    future.set_result(value)
    terminal = finish(setup, call, cancel=True)
    assert terminal.state is SnapshotState.CONSUMED
    assert not setup.raw.get(setup.request.claim_key)
    assert setup.fences.get(setup.store._local_claim_path(setup.request)) is None


def test_authoritative_owner_change_rejects_pending_completion_cas(setup):
    claimed = claim(setup)
    call = operation(setup, control.complete_direct, setup.store, claimed, setup.claim)
    assert call.poll()[0] is False
    setup.fences.call("upsert", setup.store._local_claim_path(setup.request), "replacement")
    setup.client.execute()
    with pytest.raises(RuntimeError, match="owner changed"):
        call.poll()
    assert setup.store.load(setup.request, require_ready=False).state is SnapshotState.DIRECT_LOADING


def test_no_receiver_pending_claim_pins_exact_workset_until_reply(setup):
    broker = AgenticPWorksetLeaseBroker(4)
    broker.request(setup.request.snapshot_id, 8, 12)
    broker.service(NS(alloc=lambda size: torch.arange(size), free=lambda *_a: None))
    lease = broker.get(setup.request.snapshot_id)
    assert broker.begin_io_attempt(setup.request.snapshot_id, lease, setup.claim)
    call = operation(setup, control.claim_direct, setup.store, setup.request, setup.claim, 12)
    call.identity = (setup.request.snapshot_id, 12, lease.lease_id, setup.claim)
    call.poll()
    scheduler = NS(
        _agentic_direct_claim_calls={setup.request.snapshot_id: (lease, setup.request, call)},
        agentic_tp_direct_admission_active={setup.request.snapshot_id: (setup.request, 0, None, 12, lease)},
        agentic_tp_direct_mailbox=NS(receipt=lambda _sid: -1),
        agentic_early_direct_terminal={}, agentic_tp_direct_local_failed=set(),
        agentic_p_workset_broker=broker,
    )
    Scheduler._agentic_progress_cancelled_direct_claims(scheduler)
    assert lease.state == "io_reserved"
    assert scheduler._agentic_direct_claim_calls
    setup.client.execute()
    Scheduler._agentic_progress_cancelled_direct_claims(scheduler)
    assert lease.state == "releasing"
    assert not scheduler._agentic_direct_claim_calls
    assert setup.request.snapshot_id in scheduler.agentic_tp_direct_local_failed


def test_one_pending_claim_does_not_block_another_snapshot(setup):
    first = operation(setup, control.claim_direct, setup.store, setup.request, setup.claim, 12)
    first.poll()
    first_rpc = setup.client.queue.pop()
    request = RequestGeneration("independent", 1)
    manifest = replace(setup.manifest, request=request, direct_room=13)
    setup.raw.put(manifest.manifest_key, manifest.to_bytes())
    second = control.DirectControlCall((request.snapshot_id, 13, 4, "other"),
                                      control.claim_direct, setup.store, request, "other", 13)
    assert finish(setup, second).state is SnapshotState.DIRECT_LOADING
    assert not first_rpc[0].done()
    assert not first.settled


def test_sync_and_async_success_have_identical_lifecycle_records(setup, monkeypatch):
    monkeypatch.setattr(control.time, "time", lambda: 1234.0)
    claimed = setup.store.claim_direct(setup.request, setup.claim)
    received = setup.store.complete_direct_group(claimed, setup.claim)
    terminal = setup.store.commit_direct_bound(received, setup.claim)
    expected = {namespace: dict(values) for namespace, values in setup.client.records._data.items()}
    setup.client.records._data.clear()
    setup.raw.put(setup.manifest.manifest_key, setup.manifest.to_bytes())
    async_claimed = claim(setup)
    async_received = finish(setup, operation(setup, control.complete_direct,
                                             setup.store, async_claimed, setup.claim))
    async_terminal = finish(setup, operation(setup, control.commit_bound,
                                             setup.store, async_received, setup.claim))
    assert (async_claimed, async_received, async_terminal) == (claimed, received, terminal)
    assert setup.client.records._data == expected


def test_pending_bound_rank_failure_cannot_publish_group_abort(setup):
    claimed = claim(setup)
    received = finish(setup, operation(setup, control.complete_direct,
                                      setup.store, claimed, setup.claim))
    bound = operation(setup, control.commit_bound, setup.store, received, setup.claim)
    bound.poll()
    late_ack, value = setup.client.execute(ack=False)
    entry = receive_entry(claimed, abort_requested=True)
    entry._direct_control_calls = {"bound": bound}
    identity = entry.control_identity
    entry.workset_lease = None
    receipts = []
    owner = NS(agentic_early_direct_poll_lock=nullcontext(),
               agentic_tp_direct_admission_active={setup.request.snapshot_id: (setup.request, 0, None, 12, object())},
               agentic_early_direct_receives={setup.request.snapshot_id: entry},
               agentic_tp_direct_mailbox=NS(publish_receipt=lambda *args: receipts.append(args)))
    assert not Scheduler._agentic_abort_tp_direct_grant(owner, setup.request, setup.store, reason="rank_failure")
    assert not receipts
    late_ack.set_result(value)
    # Still no negative receipt while the terminal claim cleanup is pending.
    assert not Scheduler._agentic_abort_tp_direct_grant(owner, setup.request, setup.store, reason="rank_failure")
    assert not receipts
    while setup.client.queue:
        setup.client.execute()
        assert not Scheduler._agentic_abort_tp_direct_grant(owner, setup.request, setup.store, reason="rank_failure")
    assert receipts == [(setup.request.snapshot_id, 4)]
    assert entry.group_committed and not entry.abort_requested
    assert entry.control_identity is identity and bound.identity == identity
    assert setup.store.load(setup.request, require_ready=False).state is SnapshotState.CONSUMED


def test_join_rejects_same_generation_claim_with_different_wire_attempt(setup):
    claim(setup)
    stale = operation(setup, control.claim_direct, setup.store, setup.request, setup.claim, 99)
    with pytest.raises(SnapshotNotReadyError, match="offer changed"):
        finish(setup, stale)
    assert setup.store.load(setup.request, require_ready=False).direct_room == 12


def test_cancel_before_claim_submission_has_no_side_effect(setup):
    call = operation(setup, control.claim_direct, setup.store, setup.request, setup.claim, 12)
    assert call.poll(cancel=True) == (True, None)
    assert not setup.client.history


def test_route_publication_is_acknowledged_and_not_resubmitted(setup):
    markers = Records(setup.client, "markers")
    marker_store = NS(_records=markers, route_path=lambda request: request.snapshot_id,
                      _record_key=lambda path: f"route/{path}")
    call = operation(setup, control.publish_route, marker_store, setup.request, 1, 8)
    assert call.poll()[0] is False
    assert call.poll()[0] is False
    assert len(setup.client.history) == 1
    setup.client.execute()
    ready, value = call.poll()
    assert ready and value["route"] == "direct_complete"
    assert value["prefill_domain"] == 1
    assert markers.get(f"route/{setup.request.snapshot_id}") == value


@pytest.mark.parametrize("unknown", [False, True])
@pytest.mark.parametrize("phase", ["complete", "bound", "route"])
def test_failed_group_future_does_not_block_next_healthy_group(setup, unknown, phase):
    claimed = claim(setup)
    entry = receive_entry(claimed, completed_at=1.0,
                          prefill_domain=1 if phase == "route" else None)
    bad = operation(setup, control.complete_direct, setup.store, claimed, setup.claim)
    bad.poll()
    pending, _, _ = setup.client.queue.pop()
    pending.set_exception(ControlUnavailable("reply lost") if unknown else RuntimeError("owner changed"))
    entry._direct_control_calls = {phase: bad}
    if phase == "route":
        complete = operation(setup, control.complete_direct, setup.store, claimed, setup.claim)
        finish(setup, complete)
        entry._direct_control_calls["complete"] = complete
    other = RequestGeneration("healthy-after-error", 1)
    other_manifest = replace(claimed, request=other, direct_room=13, claim_id="healthy")
    setup.raw.put(other_manifest.manifest_key, other_manifest.to_bytes())
    setup.fences.call("claim", setup.store._local_claim_path(other), "healthy")
    healthy = receive_entry(other_manifest, lease_id=4, completed_at=1.0)
    receipts, statuses = {}, {}
    owner = NS(tp_rank=0, tp_size=8, agentic_early_direct_poll_lock=nullcontext(),
               agentic_tp_direct_local_failed=set(), agentic_early_claim_store=object(),
               agentic_early_direct_receives={setup.request.snapshot_id: entry, other.snapshot_id: healthy},
               agentic_tp_direct_admission_active={
                   setup.request.snapshot_id: (setup.request, 1, None, 12, entry.workset_lease),
                   other.snapshot_id: (other, 1, None, 13, healthy.workset_lease)},
               agentic_tp_direct_mailbox=NS(
                   _key=lambda sid: EventKey(sid, "direct-room:" + str(12 if sid == setup.request.snapshot_id else 13)),
                   publish_local_progress=lambda sid, value: statuses.update({sid: value}),
                   receipt=lambda sid: receipts.get(sid),
                   group_status=lambda sid: -1 if statuses.get(sid) == -1 else (4 if sid == setup.request.snapshot_id and phase == "bound" else 3),
                   publish_receipt=lambda sid, value: receipts.update({sid: value})))
    Scheduler._agentic_commit_tp_direct_groups(owner, setup.store)
    assert setup.request.snapshot_id in owner.agentic_tp_direct_local_failed if not unknown else not owner.agentic_tp_direct_local_failed
    assert other.snapshot_id in owner.agentic_early_direct_receives
    assert len(setup.client.queue) == 1  # Healthy CAS submitted in the same pass.
    setup.client.execute()
    Scheduler._agentic_commit_tp_direct_groups(owner, setup.store)
    assert healthy.group_committed and receipts[other.snapshot_id] == 3
    if unknown:
        assert setup.request.snapshot_id not in receipts
        assert not bad.settled
        assert setup.request.snapshot_id in owner.agentic_tp_direct_admission_active
        assert not Scheduler._agentic_drain_direct_control(owner, entry)
    else:
        assert receipts[setup.request.snapshot_id] == -1


def test_failed_cleanup_after_consumed_cannot_become_abortable(setup):
    claimed = claim(setup)
    received = finish(setup, operation(setup, control.complete_direct, setup.store, claimed, setup.claim))
    entry = receive_entry(claimed)
    owner = NS(agentic_tp_direct_local_failed=set())
    assert not Scheduler._agentic_poll_direct_control(owner, entry, "bound", control.commit_bound,
                                                      setup.store, received, setup.claim)[0]
    setup.client.execute()  # Authoritative CONSUMED commit.
    Scheduler._agentic_poll_direct_control(owner, entry, "bound", control.commit_bound,
                                          setup.store, received, setup.claim)
    pending, _, _ = setup.client.queue.pop()
    pending.set_exception(RuntimeError("terminal cleanup failed"))
    assert not Scheduler._agentic_poll_direct_control(owner, entry, "bound", control.commit_bound,
                                                      setup.store, received, setup.claim)[0]
    assert not owner.agentic_tp_direct_local_failed
    assert not Scheduler._agentic_drain_direct_control(owner, entry)
    assert setup.store.load(setup.request, require_ready=False).state is SnapshotState.CONSUMED


def test_completed_bound_retry_survives_native_handoff_before_followers_report_five(setup):
    claimed = claim(setup)
    received = finish(setup, operation(setup, control.complete_direct, setup.store, claimed, setup.claim))
    entry = receive_entry(claimed, completed_at=1.0, group_committed=True)
    bound = operation(setup, control.commit_bound, setup.store, received, setup.claim)
    finish(setup, bound)
    entry._direct_control_calls = {"bound": bound}
    identity = entry.control_identity
    receipts = []
    owner = NS(tp_rank=0, tp_size=8, agentic_early_direct_poll_lock=nullcontext(),
               agentic_tp_direct_local_failed=set(),
               agentic_early_direct_receives={setup.request.snapshot_id: entry},
               agentic_tp_direct_admission_active={setup.request.snapshot_id:
                   (setup.request, 1, None, 12, entry.workset_lease)})
    status = [4]
    def group_status(_sid):
        # The worker already copied the receive table at the start of this
        # pass. Native admission can now transfer the lease to Req and remove
        # the table entry while followers' status still reads 4.
        entry.workset_lease = None
        owner.agentic_early_direct_receives.clear()
        return status[0]
    owner.agentic_tp_direct_mailbox = NS(
        _key=lambda sid: EventKey(sid, "direct-room:12"),
        publish_local_progress=lambda *_a: None, receipt=lambda _sid: 4,
        group_status=group_status,
        publish_receipt=lambda *args: receipts.append(args))
    Scheduler._agentic_commit_tp_direct_groups(owner, setup.store)
    assert receipts == [(setup.request.snapshot_id, 4)]
    assert entry.control_identity is identity and bound.identity == identity
    assert not setup.client.queue
    status[0] = 5
    Scheduler._agentic_commit_tp_direct_groups(owner, setup.store)
    assert receipts[-1] == (setup.request.snapshot_id, 5)
    assert not owner.agentic_tp_direct_local_failed


@pytest.mark.parametrize("phase", ["complete", "bound"])
def test_pending_future_poll_and_cancel_drain_keep_identity_after_lease_detaches(setup, phase):
    claimed = claim(setup)
    entry = receive_entry(claimed)
    manifest = claimed
    method = control.complete_direct
    if phase == "bound":
        manifest = finish(setup, operation(setup, control.complete_direct, setup.store, claimed, setup.claim))
        method = control.commit_bound
    owner = NS(agentic_tp_direct_local_failed=set())
    assert not Scheduler._agentic_poll_direct_control(owner, entry, phase, method,
                                                      setup.store, manifest, setup.claim)[0]
    identity = entry.control_identity
    call = entry._direct_control_calls[phase]
    entry.workset_lease = None
    assert not Scheduler._agentic_poll_direct_control(owner, entry, phase, method,
                                                      setup.store, manifest, setup.claim)[0]
    assert not Scheduler._agentic_drain_direct_control(owner, entry)
    while setup.client.queue:
        setup.client.execute()
        settled = Scheduler._agentic_drain_direct_control(owner, entry)
    assert settled
    assert entry.control_identity is identity and call.identity == identity
    assert not owner.agentic_tp_direct_local_failed


def test_captured_control_identity_is_read_only_and_rejects_replacement_lease(setup):
    claimed = claim(setup)
    entry = receive_entry(claimed)
    with pytest.raises(AttributeError):
        entry.control_identity = (setup.request.snapshot_id, 12, 99, setup.claim)
    original = entry.control_identity
    entry.workset_lease = NS(lease_id=99)
    with pytest.raises(RuntimeError, match="identity"):
        Scheduler._agentic_poll_direct_control(NS(), entry, "complete", control.complete_direct,
                                               setup.store, claimed, setup.claim)
    assert entry.control_identity is original


def test_native_plan_timestamps_are_local_and_preserved_on_reinstall(setup):
    broker = AgenticPWorksetLeaseBroker(4)
    broker.request(setup.request.snapshot_id, 8, 12)
    plan = broker.prepare_tp_plan(1)
    frozen_at = broker._tp_plan_at
    broker.install_tp_plan(1, plan)
    assert broker._tp_plan_at == frozen_at
    broker.service(NS(alloc=lambda size: torch.arange(size), free=lambda *_a: None))
    lease = broker.get(setup.request.snapshot_id)
    assert lease.intent_at <= lease.plan_at <= lease.service_at <= lease.grant_at
    assert lease.receipt_submitted_at is None and lease.receipt_observed_at is None
