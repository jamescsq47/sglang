"""Independent CPU regressions for the A100 safety fixes on the H100 branch."""

import queue
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.srt.disaggregation.agentic_host_staging import (
    AgenticPHostStagingManager,
    HostStageState,
    SharedHostStagingLedger,
)
from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration
from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    AgenticRequestMetadata,
    DecodeKVCacheOffloadManager,
)
from sglang.srt.managers.scheduler import AgenticPWorksetLeaseBroker, Scheduler
from sglang.srt.disaggregation.agentic_tp_control import TPGroupMailbox
from sglang.srt.disaggregation.agentic_kv_lifecycle import SnapshotState
from sglang.srt.disaggregation.utils import DisaggregationMode


@pytest.fixture
def ledger_dir():
    with tempfile.TemporaryDirectory(prefix="agentic-safety-", dir="/dev/shm") as path:
        yield Path(path)


@pytest.mark.parametrize("tp_size", [2, 4])
def test_four_failed_direct_slots_return_after_all_rank_rollback(ledger_dir, tp_size):
    requests = [RequestGeneration(f"abort-slot-{i}", 0) for i in range(4)]
    store = SimpleNamespace(load=lambda *_a, **_kw: SimpleNamespace(state=SnapshotState.DIRECT_READY))
    owners = []
    for rank in range(tp_size):
        mailbox = TPGroupMailbox("rollback-test", tp_rank=rank, tp_size=tp_size, directory=str(ledger_dir))
        owners.append(SimpleNamespace(
            tp_size=tp_size, tp_rank=rank,
            disaggregation_mode=DisaggregationMode.PREFILL,
            _AGENTIC_TP_CONTROL_KEY=Scheduler._AGENTIC_TP_CONTROL_KEY,
            agentic_tp_direct_mailbox=mailbox,
            agentic_early_direct_poll_lock=threading.RLock(),
            agentic_tp_direct_admission_active={r.snapshot_id: (r, 0.0, None, 128, None) for r in requests},
            agentic_tp_direct_group_status={}, agentic_early_direct_receives={},
            agentic_tp_direct_local_failed={r.snapshot_id for r in requests},
            agentic_tp_direct_local_admitted=set(), agentic_tp_direct_local_rolled_back=set(),
            agentic_p_workset_broker=SimpleNamespace(
                install_tp_plan=lambda *_a, **_kw: None,
                cancel_unstarted=lambda *_a, **_kw: None,
            ),
            agentic_host_staging_manager=None,
        ))
        for request in requests:
            mailbox.publish_local_progress(request.snapshot_id, -1)
            if rank == 0:
                mailbox.publish_receipt(request.snapshot_id, -1)
        assert Scheduler._agentic_early_direct_slots_used(owners[-1]) == 4

    def control(action):
        return {
            Scheduler._AGENTIC_TP_CONTROL_KEY: True,
            "workset_plan_epoch": 1, "workset_allocation_plan": [],
            "direct_commands": [{"snapshot": r.snapshot_id, "action": action} for r in requests],
        }

    # Abort races a background start after begin_io_attempt but before the
    # receiver becomes visible.  No rank may acknowledge this unfinished DMA.
    owner = owners[0]
    broker = AgenticPWorksetLeaseBroker(page_size=4)
    broker.install_tp_plan = lambda *_a, **_kw: None
    owner.agentic_p_workset_broker = broker
    request = requests[0]
    lease = SimpleNamespace(lease_id="invisible-start", owner="direct", state="active", io_attempt=None)
    broker._leases[request.snapshot_id] = lease
    owner.agentic_tp_direct_admission_active[request.snapshot_id] = (request, 0.0, None, 128, lease)
    assert broker.begin_io_attempt(request.snapshot_id, lease, "attempt")
    for state in ("io_reserved", "io_inflight", "release_pending"):
        lease.state = state
        Scheduler._agentic_tp_consume_admission_control(owner, [control("abort")])
        assert request.snapshot_id not in owner.agentic_tp_direct_local_rolled_back
        assert not owner.agentic_tp_direct_mailbox.rollback_group_complete(request.snapshot_id)
    # Physical quiescence permits the ordinary rollback protocol to finish.
    lease.state = "retire_ready"
    assert broker.tp_retire_ready(request.snapshot_id, lease_id=lease.lease_id)
    # A later unrelated lease cannot make an old exact-lease ACK wait forever.
    assert broker.tp_retire_ready(request.snapshot_id, lease_id="already-retired")

    for rank, owner in enumerate(owners):
        Scheduler._agentic_tp_consume_admission_control(owner, [control("abort")])
        # Retry ACK and late ordinary failure must not erase rollback evidence.
        Scheduler._agentic_commit_tp_direct_groups(owner, store)
        for request in requests:
            owner.agentic_tp_direct_mailbox.publish_local_progress(request.snapshot_id, -1)
        Scheduler._agentic_commit_tp_direct_groups(owners[0], store)
        expected = -2 if rank == tp_size - 1 else -1
        for request in requests:
            assert owners[0].agentic_tp_direct_mailbox.receipt(request.snapshot_id) == expected
            assert owner.agentic_tp_direct_mailbox.local_status(request.snapshot_id) == -1

    for owner in owners:
        Scheduler._agentic_tp_consume_admission_control(owner, [control("clear")])
        assert Scheduler._agentic_early_direct_slots_used(owner) == 0
    for request in requests:
        assert not owners[0].agentic_tp_direct_mailbox.rollback_group_complete(request.snapshot_id)
    next_request = RequestGeneration("next-direct", 0)
    owners[0].agentic_tp_direct_admission_active[next_request.snapshot_id] = (next_request, 0.0, None, 128, None)
    assert Scheduler._agentic_early_direct_slots_used(owners[0]) == 1


def _ready_ledger(ledger_dir, request, **fields):
    ledger = SharedHostStagingLedger(str(ledger_dir / "ledger.json"))
    snapshot_id = request.snapshot_id
    ledger.offer(
        dict(snapshot_id=snapshot_id, token_count=8, byte_size=4096, tp_size=1)
    )

    def publish(entries):
        entries[snapshot_id].update(
            dict(p_owner="host", state=HostStageState.HOST_READY.value, **fields)
        )
        return True, True

    ledger._mutate(publish, event_snapshot_id=snapshot_id)
    return ledger


def _host_manager(ledger, request, *, owner="host", domain=0):
    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager._state_lock = threading.RLock()
    manager.owner = owner
    manager.arena_domain = domain
    manager.tp_rank = 0
    manager.tp_size = 1
    manager.ledger = ledger
    manager.workset_broker = AgenticPWorksetLeaseBroker(page_size=4)
    manager.loads = {}
    manager.host_ready = {
        request.snapshot_id: {
            "loading": False,
            "snapshot": object(),
            "offer": {"token_count": 8, "byte_size": 4096},
        }
    }
    manager._h2d_lane_reservations = {request.snapshot_id: 0}
    manager._control_wakeup = threading.Event()
    released = []
    manager._release_record = lambda record: released.append(record) or True
    return manager, released


@pytest.mark.parametrize("domain", [0, 1])
def test_assigned_unclaimed_host_snapshot_cannot_be_evicted(ledger_dir, domain):
    request = RequestGeneration("assigned-before-claim", 1)
    ledger = _ready_ledger(ledger_dir, request)
    assert ledger.assign_d2p_recovery_domain(request.snapshot_id, domain)
    assert not ledger.begin_host_eviction(
        request.snapshot_id, "host", tp_size=1, reason="pressure"
    )
    entry = ledger.get(request.snapshot_id)
    assert entry["state"] == HostStageState.HOST_READY.value
    assert entry["recovery_domain"] == domain


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_static_local_recovery_claim_with_domain(ledger_dir, tp_size):
    request = RequestGeneration("static-local-recovery", 1)
    ledger = _ready_ledger(ledger_dir, request, tp_size=tp_size, arena_domain=0)
    sid = request.snapshot_id
    before = ledger.get(sid)
    for owner, domain in [("foreign", 0), ("host", 1)]:
        assert not ledger.claim_d2p_recovery_rank(
            sid, owner, tp_rank=0, tp_size=tp_size,
            claim_id="child", recovery_domain=domain,
        )
        assert ledger.get(sid) == before
    for rank in range(tp_size):
        for _ in range(2):
            assert ledger.claim_d2p_recovery_rank(
                sid, "host", tp_rank=rank, tp_size=tp_size,
                claim_id="child", recovery_domain=0,
            )
        assert not ledger.begin_host_eviction(
            sid, "host", tp_size=tp_size, reason="pressure"
        )
    entry = ledger.get(sid)
    assert entry["recovery_domain"] == 0
    assert entry["recovery_owner"] == "host"
    assert len(entry["recovery_claims"]) == tp_size
    assert entry["state"] == HostStageState.H2D_LOADING.value
    assert not ledger.claim_d2p_recovery_rank(
        sid, "host", tp_rank=0, tp_size=tp_size,
        claim_id="other-child", recovery_domain=0,
    )


@pytest.mark.parametrize("claimed", [False, True])
def test_expired_assignment_allows_eviction_only_before_recovery_claim(
    ledger_dir, claimed
):
    request = RequestGeneration("expired-assignment", 1)
    ledger = _ready_ledger(ledger_dir, request)
    assert ledger.assign_d2p_recovery_domain(
        request.snapshot_id, 0, lease_seconds=1.0
    )
    assert ledger.get(request.snapshot_id)["recovery_assignment_expires_at"] > 0
    if claimed:
        assert ledger.claim_d2p_recovery_rank(
            request.snapshot_id, "recovery", tp_rank=0, tp_size=1,
            claim_id="selected-child", recovery_domain=0,
        )

    def expire(entries):
        entries[request.snapshot_id]["recovery_assignment_expires_at"] = 0.0
        return True, True

    ledger._mutate(expire, event_snapshot_id=request.snapshot_id)
    assert ledger.begin_host_eviction(
        request.snapshot_id, "host", tp_size=1, reason="pressure"
    ) is (not claimed)
    entry = ledger.get(request.snapshot_id)
    if claimed:
        assert entry["state"] == HostStageState.H2D_LOADING.value
        assert entry["recovery_owner"] == "recovery"
        assert entry["recovery_claims"]["0"]["claim_id"] == "selected-child"
    else:
        assert entry["state"] == HostStageState.EVICTING.value
        assert "recovery_domain" not in entry
        assert "recovery_assignment_expires_at" not in entry
        assert not ledger.claim_d2p_recovery_rank(
            request.snapshot_id, "recovery", tp_rank=0, tp_size=1,
            claim_id="late-child", recovery_domain=0,
        )


@pytest.mark.parametrize("claimed", [False, True])
def test_storage_owner_abort_cannot_mutate_foreign_recovery(ledger_dir, claimed):
    request = RequestGeneration("foreign-recovery", 1)
    ledger = _ready_ledger(ledger_dir, request, recovery_domain=1)
    if claimed:
        assert ledger.claim_d2p_recovery_rank(
            request.snapshot_id, "recovery", tp_rank=0, tp_size=1,
            claim_id="selected-child", recovery_domain=1,
        )
    manager, released = _host_manager(ledger, request)
    before = ledger.get(request.snapshot_id)
    record = manager.host_ready[request.snapshot_id]
    manager.abort_request("obsolete-direct-attempt", request)
    assert ledger.get(request.snapshot_id) == before
    assert manager.host_ready[request.snapshot_id] is record
    assert record["loading"] is False
    assert "abort_requested" not in record
    assert not getattr(manager, "_pending_host_abort_requests", {})
    assert not getattr(manager, "_prestart_recovery_aborts", {})
    assert released == []


@pytest.mark.parametrize("failure_point", ["cas", "post_cas_get"])
def test_one_abort_call_retries_ledger_fault_in_background(
    ledger_dir, monkeypatch, failure_point
):
    request = RequestGeneration("abort-retry", 1)
    ledger = _ready_ledger(ledger_dir, request, recovery_domain=0)
    manager, released = _host_manager(ledger, request)
    original_get = ledger.get
    method = "request_host_load_failure" if failure_point == "cas" else "get"
    original_method = getattr(ledger, method)
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected one-shot ledger fault")
        return original_method(*args, **kwargs)

    monkeypatch.setattr(ledger, method, fail_once)
    manager.abort_request("cancelled-child", request)
    expected = (
        HostStageState.HOST_READY if failure_point == "cas"
        else HostStageState.ABORTING
    )
    assert original_get(request.snapshot_id)["state"] == expected.value
    assert released == []
    assert manager.host_ready[request.snapshot_id]["loading"] is False
    assert request.snapshot_id in manager._pending_host_abort_requests

    # No second HTTP/scheduler cancellation is delivered. The normal control
    # worker owns retrying the intent retained from the one external call.
    manager._progress_host_abort_requests()
    assert original_get(request.snapshot_id)["state"] == HostStageState.FAILED.value
    assert not manager._pending_host_abort_requests
    assert not getattr(manager, "_prestart_recovery_aborts", {})
    assert request.snapshot_id not in manager.host_ready
    assert request.snapshot_id not in manager._h2d_lane_reservations
    assert len(released) == 1
    manager._progress_host_abort_requests()
    manager._progress_prestart_aborts()
    assert len(released) == 1


def test_decode_release_handoff_preserves_exact_tail_accounting(monkeypatch):
    snapshot_id = "detached:1"
    req = SimpleNamespace(
        rid="detached", req_pool_idx=3, kv_allocated_len=97,
        kv_committed_len=96, cache_protected_len=64,
    )
    manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
    manager.page_size = 64
    manager.tp_world_size = 1
    manager._agentic_candidates_lock = threading.RLock()
    manager._agentic_pending_release_lock = threading.RLock()
    manager.agentic_direct_candidates = {snapshot_id: {"req": req}}
    manager._agentic_release_ownership = {}
    manager._agentic_tp_pending_releases = {}
    manager._agentic_slow_active_ids = {snapshot_id: None}
    manager._decode_io_async_enabled = True
    manager._decode_io_events = queue.Queue()
    manager._decode_pending_release_tokens = 0
    manager._decode_commit_interval = 0
    manager._decode_scheduler_commit_events = 0
    manager._decode_scheduler_commit_seconds = 0
    monkeypatch.setattr(
        AgenticRequestMetadata, "from_req",
        staticmethod(lambda _req: SimpleNamespace(current=SimpleNamespace(snapshot_id=snapshot_id))),
    )
    enqueue = manager._enqueue_agentic_release
    observed = []

    def observe_handoff(*args, **kwargs):
        assert snapshot_id not in manager.agentic_direct_candidates
        assert manager._decode_io_events.empty()
        observed.append((manager.agentic_pending_release_token_count,
                         manager.agentic_pending_release_req_count))
        return enqueue(*args, **kwargs)

    manager._enqueue_agentic_release = observe_handoff
    assert manager.agentic_pending_release_token_count == 64
    manager._retire_candidate_for_release(snapshot_id, req, 0)
    assert observed == [(64, 1)]
    # The legacy queue scalar includes the cached prefix, but only the tail
    # absent from Radix accounting belongs in the detached-token total.
    assert manager._decode_pending_release_tokens == 128
    assert manager.agentic_pending_release_token_count == 64
    assert manager.agentic_pending_release_req_count == 1

    released = []

    def release(value, offset):
        released.append((value, offset))
        value.req_pool_idx = -1

    manager._release_finished_req = release
    manager._drain_decode_io_events()
    assert released == [(req, 0)]
    assert manager.agentic_pending_release_token_count == 0
    assert manager.agentic_pending_release_req_count == 0
    assert manager._decode_pending_release_tokens == 0
