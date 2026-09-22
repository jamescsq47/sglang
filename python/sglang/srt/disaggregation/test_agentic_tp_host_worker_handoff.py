"""CPU ownership/notification tests; fake DMA is not a GPU performance proof."""
from concurrent.futures import Future
import threading
from types import SimpleNamespace

import pytest

from sglang.srt.disaggregation.agentic_host_staging import AgenticPHostStagingManager
from sglang.srt.disaggregation.agentic_host_rpc import RemoteHostStagingLedger
from sglang.srt.disaggregation.test_agentic_host_rpc import connection
from sglang.srt.disaggregation.test_agentic_host_control import OWNER, SID, claim, ready


def manager(rank=0, size=2):
    m = object.__new__(AgenticPHostStagingManager)
    m.tp_rank, m.tp_size, m.owner = rank, size, "p"
    m._control_wakeup = threading.Event()
    m.loads, m.host_ready = {}, {"s": {"snapshot": object()}}
    m._h2d_lane_reservations = {"s": 0}
    m.tp_host_commit_snapshots, m.tp_host_admit_snapshots = ["s"], []
    m.workset_broker = SimpleNamespace(handoff_to_req=lambda *a: None)
    m.requests = []
    m._notify_scheduler = lambda *a: None
    m.releases = []
    m._release_record = lambda r: m.releases.append(r) or True
    m.future = Future()
    m.ledger = SimpleNamespace(
        is_event_control=True, get=lambda s: {"state": "consumed"},
        submit=lambda *a, **k: m.requests.append((a, k)) or m.future,
    )
    m.reports = []
    m.register_tp_host_progress("s", "attempt1", lambda k, v: m.reports.append((k, v)))
    req = SimpleNamespace(
        rid="r", _agentic_host_rank_loaded=True, _agentic_host_rank_token_count=16,
        _agentic_host_workset_lease=SimpleNamespace(owner="claim", lease_id="lease"),
        _agentic_host_remote_read_epoch=9,
    )
    return m, req


def bound_entry(m, *, epoch=9):
    return dict(state="consumed", recovery_owner=m.owner, recovery_claim_id="claim",
                remote_read_epoch=epoch, tp_size=m.tp_size, binder_acks=list(range(m.tp_size)),
                recovery_claims={str(rank): dict(claim_id="claim", lease_id="lease")
                                 for rank in range(m.tp_size)})


def mock_manifest_future(monkeypatch, future):
    calls = []
    def poll(holder, key, function, *args):
        pending = holder.__dict__.setdefault("_agentic_lifecycle_futures", {})
        if key not in pending:
            calls.append(key)
            pending[key] = future
        return (True, future.result()) if future.done() else (False, None)
    monkeypatch.setattr(
        "sglang.srt.disaggregation.agentic_lifecycle_control.poll_lifecycle_call", poll)
    return calls


@pytest.mark.parametrize("size", [2, 8])
def test_worker_ack_all_ranks_wait_native_admit_no_req_mutation(size):
    group = [manager(rank, size) for rank in range(size)]
    parent = SimpleNamespace(snapshot_id="s")
    for m, req in group:
        assert m.gate_request(req, parent)
        assert not m.releases and not m.requests  # Scheduler only handed Req.
        before = dict(vars(req))
        m._progress_tp_host_handoffs()
        assert vars(req) == before
        assert len(m.requests) == 1
        assert m.requests[0][1]["remote_read_epoch"] == 9
        assert m.requests[0][1]["lease_id"] == "lease"
    for m, _ in group[:-1]:
        m.future.set_result(True)
        m._progress_tp_host_handoffs()
        assert m.publish_tp_host_status("s", "attempt1", 3) == 4
    for m, req in group:
        assert m.gate_request(req, parent)  # Including ranks with completed ACK.
        assert not getattr(req, "_agentic_kv_gate_complete", False)
    last, _ = group[-1]
    last.future.set_result(True)
    last._progress_tp_host_handoffs()
    for m, req in group:
        assert m.reports[-1] == ("attempt1", 4)
        assert m.gate_request(req, parent)
        m.tp_host_admit_snapshots = ["s"]
        m.ledger.get = lambda s: None  # Advisory mirror may lag native command.
        assert m.gate_request(req, parent) is False
        assert req._agentic_kv_gate_complete
        assert len(m.requests) == 1
        assert len(m.releases) == 1


@pytest.mark.parametrize("ack_before_cancel", [True, False])
def test_cancel_before_admit_ignores_late_ack_and_does_not_reanimate(ack_before_cancel):
    m, req = manager()
    parent = SimpleNamespace(snapshot_id="s")
    assert m.gate_request(req, parent)
    m._progress_tp_host_handoffs()
    if ack_before_cancel:
        m.future.set_result(True)
        m._progress_tp_host_handoffs()
    m.publish_tp_host_status("s", "attempt1", 0, cancelled=True)
    m._cancel_tp_host_handoff("s")
    if not ack_before_cancel:
        m.future.set_result(True)
    m._progress_tp_host_handoffs()
    assert m.publish_tp_host_status("s", "attempt1", 4) == 0
    assert m.gate_request(req, parent)
    assert m.publish_tp_host_status("s", "attempt1", 5, cancelled=True) == 5
    assert m.publish_tp_host_status("s", "attempt1", 4) == 5
    assert not m._tp_host_handoff_jobs


def test_retry_and_new_control_identity_do_not_inherit_old_ready():
    m, req = manager()
    parent = SimpleNamespace(snapshot_id="s")
    m.gate_request(req, parent)
    m._progress_tp_host_handoffs()
    m.clear_tp_host_progress("s", "attempt1")
    m.register_tp_host_progress("s", "attempt2", lambda k, v: m.reports.append((k, v)))
    m.future.set_result(True)
    m._progress_tp_host_handoffs()
    assert m.publish_tp_host_status("s", "attempt2", 1) == 1
    assert ("attempt1", 4) not in m.reports
    assert m.publish_tp_host_status("s", "attempt1", 4) is None
    # A retry using the same control key must be able to report PREPARE again.
    assert m.publish_tp_host_status("s", "attempt2", 3) == 3
    assert m.publish_tp_host_status("s", "attempt2", 1) == 1
    assert m.publish_tp_host_status("s", "attempt2", 6) == 6
    assert m.publish_tp_host_status("s", "attempt2", 7) == 7


def test_cancel_while_worker_closes_host_does_not_block_scheduler_or_submit_ack():
    m, req = manager()
    m.gate_request(req, SimpleNamespace(snapshot_id="s"))
    entered, resume = threading.Event(), threading.Event()
    def close(record):
        entered.set()
        assert resume.wait(3)
        return True
    m._release_record = close
    thread = threading.Thread(target=m._progress_tp_host_handoffs)
    thread.start()
    try:
        assert entered.wait(3)
        assert m.publish_tp_host_status("s", "attempt1", 0, cancelled=True) == 0
        m._cancel_tp_host_handoff("s")
        assert not m.requests
    finally:
        resume.set()
        thread.join(3)
    assert not thread.is_alive()
    m._progress_tp_host_handoffs()
    assert not m.requests and not m._tp_host_handoff_jobs
    assert ("attempt1", 4) not in m.reports


@pytest.mark.parametrize("result", [False, ConnectionError("ambiguous reply")])
def test_failed_handed_ack_retains_context_without_admission_or_resubmit(result):
    m, req = manager()
    m.gate_request(req, SimpleNamespace(snapshot_id="s"))
    m._progress_tp_host_handoffs()
    if isinstance(result, Exception):
        m.future.set_exception(result)
    else:
        m.future.set_result(result)
    for _ in range(3):
        m._progress_tp_host_handoffs()
    assert m._tp_host_handoff_jobs["s"]["error"]
    assert len(m.requests) == 1
    assert ("attempt1", 4) not in m.reports
    assert not getattr(req, "_agentic_kv_gate_complete", False)


def test_pending_loaded_ack_does_not_block_other_lane_or_repeat_committed_ack():
    m, _ = manager()
    calls = []
    def poll(method, sid, owner, **kw):
        calls.append(sid)
        return (False, None) if sid == "pending" else (True, True)
    m.ledger.poll_call = poll
    def load(sid):
        return dict(request_generation=SimpleNamespace(snapshot_id=sid),
                    recovery_claim_id="claim", workset_lease=SimpleNamespace(lease_id="lease"),
                    record={"offer": {"token_count": 16, "byte_size": 4096}}, gpu_elapsed_ms=1)
    pending, ready = load("pending"), load("ready")
    assert not m._publish_d2p_hbm_ready(pending)
    assert m._publish_d2p_hbm_ready(ready)
    assert m._publish_d2p_hbm_ready(ready)
    assert calls == ["pending", "ready"]
    assert not pending.get("io_complete")


def prepared_load(m, req, *, lease_id="lease", epoch=9):
    req.origin_input_ids, req.extra_key = list(range(16)), None
    if hasattr(req, "_agentic_host_rank_loaded"):
        delattr(req, "_agentic_host_rank_loaded")
    lease = SimpleNamespace(owner="claim", lease_id=lease_id)
    record = m.host_ready["s"]
    record["offer"] = {"token_count": 16, "byte_size": 4096}
    load = dict(rid=req.rid, request_generation=SimpleNamespace(snapshot_id="s"),
                recovery_claim_id="claim", workset_lease=lease,
                remote_h2d_attempt=f"claim:epoch:{epoch}", record=record,
                device_indices=list(range(16)), radix_bound=True,
                gpu_elapsed_ms=1, io_complete=False)
    m.loads[req.rid] = load
    m._capture_tp_host_context(load)
    return load


@pytest.mark.parametrize("size", [2, 8])
def test_loaded_and_bound_ack_publish_without_scheduler_observation(size, monkeypatch):
    manifest_future = Future()
    manifest_future.set_result(True)
    mock_manifest_future(monkeypatch, manifest_future)
    group = [manager(rank, size) for rank in range(size)]
    parent = SimpleNamespace(snapshot_id="s")
    for m, req in group:
        load = prepared_load(m, req)
        m.ledger.poll_call = lambda *a, **k: (True, True)
        assert m._publish_d2p_hbm_ready(load)
        assert m.reports[-1] == ("attempt1", 2)  # No scheduler reduce needed.
        assert m.publish_tp_host_status("s", "attempt1", 1) == 2
        assert m.gate_request(req, parent, allow_bind=True)
        assert req._agentic_host_rank_loaded
        assert not m.requests  # Scheduler queued, never submitted an RPC.
        assert m.publish_tp_host_status("s", "attempt1", 3) == 2
        before = dict(vars(req))
        m._progress_tp_host_handoffs()
        assert m.requests[0][0][0] == "complete_host_bind_rank"
        assert not m.releases
        assert vars(req) == before
    for m, req in group[:-1]:
        m.future.set_result(True)
        m._progress_tp_host_handoffs()
        assert m.reports[-1] == ("attempt1", 2)
        assert m.publish_tp_host_status("s", "attempt1", 3) == 2
        assert m._tp_host_handoff_jobs["s"]["handed_requested"]
        assert not m.releases
    last, _ = group[-1]
    assert last.reports[-1] == ("attempt1", 2)
    last.future.set_result(True)
    last._progress_tp_host_handoffs()
    # No COMMIT command. All group binder evidence unlocks independent
    # cleanup/handed ACK; the existing final ADMIT remains mandatory.
    for m, req in group:
        m.future = Future()
        m.ledger.get = lambda sid, m=m: bound_entry(m)
        m.tp_host_commit_snapshots = []
        assert m.gate_request(req, parent)
        assert not m.releases
        m._progress_tp_host_handoffs()
        assert m.requests[-1][0][0] == "mark_d2p_recovery_phase_rank"
        assert len(m.releases) == 1
        m.future.set_result(True)
        m._progress_tp_host_handoffs()
        assert m.reports[-1] == ("attempt1", 4)
        assert ("attempt1", 3) not in m.reports
        assert m.gate_request(req, parent)
        m.tp_host_admit_snapshots = m.tp_host_commit_snapshots = ["s"]
        assert m.gate_request(req, parent) is False


@pytest.mark.parametrize("old_phase", [2, 3])
def test_same_control_key_retry_rejects_old_lease_epoch_completions(old_phase):
    m, req = manager()
    old = prepared_load(m, req, lease_id="old", epoch=8)
    old_context = old["tp_progress_context"]
    m._report_tp_host_completion("s", old_context, old_phase)
    assert m.reports[-1] == ("attempt1", old_phase)
    m._invalidate_tp_host_load(old)
    assert m.reports[-1] == ("attempt1", 0)
    new = prepared_load(m, req, lease_id="new", epoch=9)
    assert new["tp_progress_context"]["identity"] == ("claim", "new", 9)
    m._report_tp_host_completion("s", old_context, 3)
    assert m.publish_tp_host_status("s", "attempt1", 1) == 1
    m._report_tp_host_completion("s", new["tp_progress_context"], 2)
    assert m.reports[-1] == ("attempt1", 2)
    m._invalidate_tp_host_load(old)  # Late old cancellation cannot cancel new.
    assert not new["tp_progress_context"]["cancelled"]


@pytest.mark.parametrize("size", [2, 8])
def test_leader_manifest_barrier_waits_rank_skew_then_scheduler_reads_only_proof(size, monkeypatch):
    manifest = Future()
    calls = mock_manifest_future(monkeypatch, manifest)
    group = [manager(rank, size) for rank in range(size)]
    entries = []
    parent = SimpleNamespace(snapshot_id="s")
    for m, req in group:
        load = prepared_load(m, req)
        m._report_tp_host_completion("s", load["tp_progress_context"], 2)
        m._queue_tp_host_bound_ack(req, load)
        entry = bound_entry(m)
        entry.update(state="hbm_ready", binder_acks=[0])
        entries.append(entry)
        m.ledger.get = lambda sid, entry=entry: entry
        m._progress_tp_host_handoffs()
    for m, req in group:
        before = dict(vars(req))
        m.future.set_result(True)
        m._progress_tp_host_handoffs()
        assert vars(req) == before
        assert m.reports[-1][1] == 2
    leader = group[0][0]
    for _ in range(3):
        assert not leader._complete_shared_host_manifest(parent)
        leader._progress_tp_host_handoffs()
    assert not calls  # Last physical shard not bound: no manifest completion.
    entries[0].update(state="consumed", binder_acks=list(range(size)))
    leader._progress_tp_host_handoffs()
    assert len(calls) == 1
    for _ in range(3):
        assert not leader._complete_shared_host_manifest(parent)
        leader._progress_tp_host_handoffs()
    assert len(calls) == 1 and leader.reports[-1][1] == 2
    assert not leader.releases
    manifest.set_result(True)
    leader._progress_tp_host_handoffs()
    assert leader.reports[-1][1] == 4
    assert ("attempt1", 3) not in leader.reports
    # Native COMMIT now receives a durable exact-context proof; even missing
    # advisory ledger data must not make scheduler submit another RPC.
    leader.ledger.get = lambda sid: None
    for _ in range(5):
        assert leader._complete_shared_host_manifest(parent)
    assert len(calls) == 1 and not leader._agentic_lifecycle_futures


@pytest.mark.parametrize("mismatch", ["epoch", "claim", "lease", "owner"])
def test_manifest_refuses_other_attempt_before_submit(mismatch, monkeypatch):
    calls = mock_manifest_future(monkeypatch, Future())
    m, req = manager()
    load = prepared_load(m, req)
    m._queue_tp_host_bound_ack(req, load)
    entry = bound_entry(m)
    if mismatch == "epoch":
        entry["remote_read_epoch"] += 1
    elif mismatch == "claim":
        entry["recovery_claim_id"] = "other"
    elif mismatch == "lease":
        entry["recovery_claims"]["0"]["lease_id"] = "other"
    else:
        entry["recovery_owner"] = "other"
    m.ledger.get = lambda sid: entry
    m.future.set_result(True)
    for _ in range(3):
        m._progress_tp_host_handoffs()
    assert not calls and not m.releases
    assert not m._complete_shared_host_manifest(load["request_generation"])


@pytest.mark.parametrize("failure", [False, ConnectionError("manifest reply lost")])
def test_manifest_failure_is_retained_without_retry_or_commit(failure, monkeypatch):
    manifest = Future()
    calls = mock_manifest_future(monkeypatch, manifest)
    m, req = manager()
    load = prepared_load(m, req)
    m._queue_tp_host_bound_ack(req, load)
    m.ledger.get = lambda sid: bound_entry(m)
    m.future.set_result(True)
    m._progress_tp_host_handoffs()
    if isinstance(failure, Exception):
        manifest.set_exception(failure)
    else:
        manifest.set_result(failure)
    for _ in range(3):
        m._progress_tp_host_handoffs()
        assert not m._complete_shared_host_manifest(load["request_generation"])
    assert len(calls) == 1 and m._agentic_lifecycle_futures
    assert m._tp_host_handoff_jobs["s"].get("error") is not None
    assert not m.releases and ("attempt1", 3) not in m.reports


def test_cancel_manifest_pending_drains_exact_future_before_retry(monkeypatch):
    manifest = Future()
    calls = mock_manifest_future(monkeypatch, manifest)
    m, req = manager()
    load = prepared_load(m, req)
    m._queue_tp_host_bound_ack(req, load)
    m.ledger.get = lambda sid: bound_entry(m)
    m.future.set_result(True)
    m._progress_tp_host_handoffs()
    assert len(calls) == 1
    m._cancel_tp_host_handoff("s")
    m._progress_tp_host_handoffs()
    assert m._tp_host_handoff_jobs and m._agentic_lifecycle_futures
    assert not m._complete_shared_host_manifest(load["request_generation"])
    # Already submitted lifecycle I/O may finish; it cannot publish phase3
    # for this cancelled load or a later attempt reusing the control key.
    manifest.set_result(True)
    m._progress_tp_host_handoffs()
    assert not m._tp_host_handoff_jobs and not m._agentic_lifecycle_futures
    new = prepared_load(m, req, lease_id="retry", epoch=10)
    assert m.publish_tp_host_status("s", "attempt1", 1) == 1
    assert not m._complete_shared_host_manifest(new["request_generation"])
    assert ("attempt1", 3) not in m.reports and not m.releases


@pytest.mark.parametrize("size", [2, 8])
def test_source_release_and_prune_retain_manifest_finalization_evidence(size, monkeypatch):
    from sglang.srt.disaggregation.agentic_host_control import InMemoryHostStagingLedger
    from sglang.srt.disaggregation.test_agentic_host_control import age
    ledger = InMemoryHostStagingLedger()
    ready(ledger, size)
    claims = [claim(ledger, rank, size) for rank in range(size)]
    for command in claims:
        assert ledger.mark_d2p_recovery_phase_rank(SID, OWNER, phase="io_inflight", **command)
        assert ledger.apply_recovery_event(
            SID, OWNER, remote_read_epoch=1, event="loaded", **command)
    for command in claims:
        assert ledger.apply_recovery_event(
            SID, OWNER, remote_read_epoch=1, event="bound", **command)
    before = ledger.get(SID)
    assert before["state"] == "consumed"
    # Source Host extents may disappear before P's native COMMIT.  Pruning
    # must nevertheless retain the bound attempt until scheduler handoff.
    for rank in range(size):
        assert ledger.complete_source_host_release_rank(
            SID, OWNER, tp_rank=rank, tp_size=size)
    age(ledger)
    ledger.prune(0, 0)
    after = ledger.get(SID)
    for field in ("binder_acks", "recovery_claims", "recovery_claim_id",
                  "recovery_owner", "remote_read_epoch"):
        assert after[field] == before[field]

    manifest = Future()
    manifest.set_result(True)
    calls = mock_manifest_future(monkeypatch, manifest)
    m, req = manager(size=size)
    m.owner = OWNER
    m.ledger.get = ledger.get
    m.register_tp_host_progress(SID, "source-freed", lambda *a: m.reports.append(a))
    parent = SimpleNamespace(snapshot_id=SID)
    record = {"offer": {"token_count": 16}, "network_host": True}
    m.host_ready[SID] = record
    command = claims[0]
    load = dict(request_generation=parent, recovery_claim_id=command["claim_id"],
                workset_lease=SimpleNamespace(lease_id=command["lease_id"]),
                remote_h2d_attempt=command["claim_id"] + ":epoch:1", record=record)
    m._capture_tp_host_context(load)
    m._queue_tp_host_bound_ack(req, load)
    m.future.set_result(True)
    m._progress_tp_host_handoffs()
    assert m.reports[-1] == ("source-freed", 4)
    assert m._complete_shared_host_manifest(parent)
    assert len(calls) == 1 and len(m.releases) == 1


def test_cancel_pending_bound_ack_retains_no_stale_phase_or_worker_req_writes():
    m, req = manager()
    load = prepared_load(m, req)
    m._report_tp_host_completion("s", load["tp_progress_context"], 2)
    assert m._queue_tp_host_bound_ack(req, load)
    m._progress_tp_host_handoffs()
    m._cancel_tp_host_handoff("s")
    m.future.set_result(True)
    before = dict(vars(req))
    m._progress_tp_host_handoffs()
    assert vars(req) == before
    assert not m._tp_host_handoff_jobs
    assert not m.releases  # Existing scheduler rollback/abort owns cleanup.
    assert m.reports[-1] == ("attempt1", 0)


@pytest.mark.parametrize("failure", [False, ConnectionError("lost bound receipt")])
def test_failed_bound_ack_cannot_close_host_or_publish_commit(failure):
    m, req = manager()
    load = prepared_load(m, req)
    m._report_tp_host_completion("s", load["tp_progress_context"], 2)
    m._queue_tp_host_bound_ack(req, load)
    m._progress_tp_host_handoffs()
    if isinstance(failure, Exception):
        m.future.set_exception(failure)
    else:
        m.future.set_result(failure)
    for _ in range(2):
        m._progress_tp_host_handoffs()
    assert m._tp_host_handoff_jobs["s"]["error"]
    assert len(m.requests) == 1 and not m.releases
    assert m.publish_tp_host_status("s", "attempt1", 3) == 2
    assert m.gate_request(req, SimpleNamespace(snapshot_id="s"))


def test_tp1_keeps_legacy_completion_path():
    m, req = manager(size=1)
    load = prepared_load(m, req)
    assert "tp_progress_context" not in load


@pytest.mark.parametrize("size", [2, 8])
@pytest.mark.parametrize("close_fails_once", [False, True])
def test_real_consumed_bound_ack_cancel_drains_mapping_jobs_and_control(
    connection, size, close_fails_once,
):
    """Real ledger rejects CONSUMED abort; local cancelled mapping still drains.

    Scheduler physical-broker rollback is tested independently. Here we prove
    that finishing that rollback is not sufficient without Host/job cleanup.
    """
    _, client, _ = connection
    ledger = RemoteHostStagingLedger(client, "d2p")
    ready(ledger, size)
    commands = [dict(claim(ledger, rank, size), remote_read_epoch=1) for rank in range(size)]
    for command in commands:
        assert ledger.mark_d2p_recovery_phase_rank(
            SID, OWNER, phase="io_inflight",
            **{k: v for k, v in command.items() if k != "remote_read_epoch"},
        )
        assert ledger.complete_d2p_host_load_rank(SID, OWNER, **command)
    for command in commands:
        assert ledger.complete_host_bind_rank(SID, OWNER, **command)
    assert ledger.get(SID)["state"] == "consumed"

    m, req = manager(size=size)
    m.ledger, m.owner = ledger, OWNER
    m.register_tp_host_progress(SID, "actual", lambda *a: None)
    m.loads, m.active, m.aborting, m.host_ready = {}, {}, {}, {}
    m._h2d_lane_reservations = {SID: 0}
    m._h2d_resident_reservations = {SID}
    closes = []
    def close(*, unlink):
        closes.append(unlink)
        if close_fails_once and len(closes) == 1:
            raise OSError("retry local mapping close")
    m.host_ready[SID] = {"loading": "h2d_prepared", "remote_host": True,
                         "snapshot": SimpleNamespace(close=close)}
    m._release_record = AgenticPHostStagingManager._release_record.__get__(m)
    m.workset_broker = SimpleNamespace(
        slow_owner=lambda *a: "slow", direct_owner=lambda *a: "direct",
        owner_has_unretired_work=lambda *a, **kw: False,
    )
    command = commands[0]
    req._agentic_host_remote_read_epoch = 1
    lease = SimpleNamespace(owner=command["claim_id"], lease_id=command["lease_id"])
    # A cancelled bound-ready completion context exists, but has not handed
    # the lease to Req nor submitted the handed ACK.
    progress = m._tp_host_progress[SID]
    m._tp_host_handoff_jobs = {SID: dict(
        progress=progress, cancelled=False, ready=False, future=None,
        bound_ready=True, handed_requested=False, record=m.host_ready[SID],
        rid=req.rid, claim_id=lease.owner, lease_id=lease.lease_id, epoch=1,
    )}
    parent = SimpleNamespace(snapshot_id=SID)
    m.abort_request(req.rid, parent)
    assert not m.tp_host_control_quiescent(SID, req.rid)
    m._progress_host_abort_requests()
    if close_fails_once:
        assert SID in m._pending_host_abort_requests
        assert SID in m.host_ready
        m._progress_host_abort_requests()
    m._progress_tp_host_handoffs()
    assert SID not in m.host_ready
    assert not m._pending_host_abort_requests and not m._tp_host_handoff_jobs
    assert not m.tp_host_control_quiescent(SID, req.rid)  # Scheduler lane not yet retired.
    m._release_h2d_lane(SID)
    assert m.tp_host_control_quiescent(SID, req.rid)
    assert not m._h2d_lane_reservations and not m._h2d_resident_reservations
    assert len(closes) == (2 if close_fails_once else 1)
