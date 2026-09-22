"""Real loopback control-RPC tests; no GPU, RDMA, files or NFS needed."""

from concurrent.futures import Future, ThreadPoolExecutor
import json
import threading
from types import SimpleNamespace

import pytest

from sglang.srt.disaggregation.agentic_control_rpc import (
    ControlRPCClient, ControlRPCServer, RemoteCallError,
)
from sglang.srt.disaggregation.agentic_host_rpc import (
    HostLedgerService, RemoteHostStagingLedger,
)
from sglang.srt.disaggregation.agentic_host_staging import (
    AgenticPHostStagingManager, SharedHostStagingLedger,
)
from sglang.srt.disaggregation.test_agentic_host_control import (
    OWNER, SID, claim, offer, ready,
)


@pytest.fixture
def connection():
    server = ControlRPCServer("host-test", "secret")
    services = [HostLedgerService(server, direction) for direction in ("d2p", "p2d")]
    client = ControlRPCClient(server.address, run_id="host-test", token="secret")
    client.wait_ready()
    try:
        yield server, client, services
    finally:
        client.close()
        server.close()


@pytest.mark.parametrize("size", [1, 2, 8])
def test_real_rpc_host_lifecycle_and_guarded_receipts(connection, size):
    _, client, _ = connection
    ledger = RemoteHostStagingLedger(client, "d2p")
    ready(ledger, size)
    commands = [dict(claim(ledger, rank, size), remote_read_epoch=1) for rank in range(size)]
    for command in commands:
        assert ledger.mark_d2p_recovery_phase_rank(
            SID, OWNER, phase="io_inflight", **{k: v for k, v in command.items() if k != "remote_read_epoch"},
        )
        assert ledger.complete_d2p_host_load_rank(SID, OWNER, **command)
    assert ledger.get(SID)["state"] == "hbm_ready"
    for command in commands:
        assert ledger.complete_host_bind_rank(SID, OWNER, **command)
    assert ledger.get(SID)["state"] == "consumed"
    for command in commands:
        assert ledger.mark_d2p_recovery_phase_rank(SID, OWNER, phase="handed", **command)
        assert not ledger.complete_d2p_host_load_rank(
            SID, OWNER, **dict(command, remote_read_epoch=0),
        )
    assert RemoteHostStagingLedger(client, "p2d").get(SID) is None


def test_completion_without_attempt_identity_fails_closed(connection):
    _, client, _ = connection
    ledger = RemoteHostStagingLedger(client, "d2p")
    ready(ledger)
    before = ledger.get(SID)
    with pytest.raises(RemoteCallError, match="exact claim/lease/epoch"):
        ledger.complete_d2p_host_load_rank(SID, OWNER, tp_rank=0, tp_size=1)
    assert ledger.get(SID) == before


def test_reads_do_not_issue_rpc_and_values_are_detached(connection, monkeypatch):
    _, client, _ = connection
    ledger = RemoteHostStagingLedger(client, "d2p")
    offer(ledger)
    def forbidden(*args, **kwargs):
        raise AssertionError("local read attempted network RPC")
    monkeypatch.setattr(client, "call", forbidden)
    monkeypatch.setattr(client, "submit", forbidden)
    value = ledger.get(SID)
    value["rank_offers"].clear()
    assert ledger.snapshot_entries()[SID]["rank_offers"]
    assert ledger.get(SID)["state"] == "offered"


def test_scheduler_poll_call_does_not_wait_or_claim_early_success(connection, monkeypatch):
    server, client, services = connection
    ledger = RemoteHostStagingLedger(client, "d2p")
    offer(ledger)
    entered, release, wakeup = threading.Event(), threading.Event(), threading.Event()
    original = server._services[services[0].service]["claim"]
    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return original(*args, **kwargs)
    monkeypatch.setitem(server._services[services[0].service], "claim", delayed)
    try:
        assert ledger.poll_call("claim", SID, OWNER, wakeup=wakeup) == (False, None)
        assert entered.wait(3)
        # Retry only checks the same pending Future; no duplicate RPC, no wait.
        assert ledger.poll_call("claim", SID, OWNER, wakeup=wakeup) == (False, None)
        assert len(ledger._pending) == 1
        assert ledger.get(SID)["state"] == "offered"
    finally:
        release.set()
    assert wakeup.wait(3)
    done, value = ledger.poll_call("claim", SID, OWNER)
    assert done and value["state"] == "host_reserved"
    assert ledger.get(SID)["state"] == "host_reserved"
    assert not ledger._pending


def test_failed_poll_receipt_is_retained_not_reissued(connection, monkeypatch):
    _, client, _ = connection
    ledger = RemoteHostStagingLedger(client, "d2p")
    failed = Future()
    failed.set_exception(ConnectionError("reply lost"))
    submissions = []
    def submit(*args, **kwargs):
        submissions.append(args)
        return failed
    monkeypatch.setattr(ledger, "submit", submit)
    for _ in range(2):
        with pytest.raises(ConnectionError, match="reply lost"):
            ledger.poll_call("claim", SID, OWNER)
    assert len(submissions) == 1
    assert len(ledger._pending) == 1


def test_pushed_watcher_has_initial_snapshot_and_independent_queues(connection):
    _, client, _ = connection
    ledger = RemoteHostStagingLedger(client, "d2p")
    offer(ledger)
    watchers = [ledger.create_watcher(), ledger.create_watcher()]
    try:
        assert all(w.initial_entries[SID]["state"] == "offered" for w in watchers)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(w.wait_events) for w in watchers]
            ledger.claim(SID, OWNER)
            for future in futures:
                assert future.result(timeout=3)[-1]["entry"]["state"] == "host_reserved"
    finally:
        for watcher in watchers:
            watcher.close()


def test_rpc_and_factory_do_not_touch_files(connection, monkeypatch):
    import fcntl
    import os
    import sglang.srt.disaggregation.agentic_control_rpc as rpc
    _, client, _ = connection
    monkeypatch.setattr(rpc, "get_control_client", lambda: client)
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "tcp://unused:1")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_STAGING_LEDGER_PATH", "/unmounted/nfs/run/d2p.json")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_P2D_STAGING_LEDGER_PATH", "/unmounted/nfs/run/p2d.json")
    def forbidden(*args, **kwargs):
        raise AssertionError("Host RPC touched filesystem")
    with monkeypatch.context() as patch:
        patch.setattr("builtins.open", forbidden)
        patch.setattr(fcntl, "flock", forbidden)
        for name in ("open", "fdopen", "stat", "scandir", "mkdir", "makedirs", "unlink", "replace"):
            patch.setattr(os, name, forbidden)
        ledger = SharedHostStagingLedger("/unmounted/nfs/run/d2p.json")
        assert isinstance(ledger, RemoteHostStagingLedger)
        ready(ledger, 2)
        assert ledger.get(SID)["state"] == "host_ready"
        assert len(ledger.snapshot_entries()) == 1


def test_scheduler_abort_only_enqueues_existing_worker_context():
    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager.ledger = SimpleNamespace(is_event_control=True)
    manager._control_wakeup = threading.Event()
    parent = SimpleNamespace(snapshot_id=SID)
    manager.abort_request("child", parent)
    assert manager._pending_host_abort_requests[SID] == {
        "rid": "child", "request_generation": parent,
    }
    assert manager._control_wakeup.is_set()


@pytest.mark.parametrize("size", [1, 2, 8])
def test_p2d_receiver_attempts_are_fenced(connection, size):
    _, client, _ = connection
    ledger = RemoteHostStagingLedger(client, "p2d")
    ready(ledger, size)
    commands = [dict(tp_rank=rank, tp_size=size, decode_domain=3,
                     attempt_id="receiver-" + str(rank)) for rank in range(size)]
    for command in commands:
        assert ledger.begin_host_load_rank(SID, OWNER, **command)
        assert ledger.begin_host_load_rank(SID, OWNER, **command)
        before = ledger.get(SID)
        assert not ledger.begin_host_load_rank(SID, OWNER, **dict(command, attempt_id="replacement"))
        assert not ledger.complete_host_load_rank(SID, OWNER, **dict(command, decode_domain=4))
        assert not ledger.complete_host_load_rank(SID, OWNER, **dict(command, attempt_id="stale"))
        assert not ledger.request_host_load_failure(SID, OWNER, reason="stale", **dict(command, attempt_id="stale"))
        assert ledger.get(SID) == before
    for rank, command in enumerate(commands):
        assert ledger.complete_host_load_rank(SID, OWNER, **command)
        assert ledger.complete_host_load_rank(SID, OWNER, **command)
        assert ledger.get(SID)["state"] == ("consumed" if rank + 1 == size else "h2d_loading")
    with pytest.raises(RemoteCallError, match="destination/attempt"):
        ledger.complete_host_load_rank(SID, OWNER, tp_rank=0, tp_size=size)


def test_p2d_unstarted_rank_joins_abort_without_stealing_receiver(connection):
    _, client, _ = connection
    ledger = RemoteHostStagingLedger(client, "p2d")
    ready(ledger, 2)
    first = dict(tp_rank=0, tp_size=2, decode_domain=3, attempt_id="rank0")
    peer = dict(tp_rank=1, tp_size=2, decode_domain=3, attempt_id="rank1")
    before = ledger.get(SID)
    assert not ledger.mark_host_load_rank_drained(SID, OWNER, **peer)
    assert ledger.get(SID) == before
    assert ledger.begin_host_load_rank(SID, OWNER, **first)
    assert ledger.request_host_load_failure(SID, OWNER, reason="cancel", **first)
    assert not ledger.mark_host_load_rank_drained(SID, OWNER, **dict(first, attempt_id="replacement"))
    assert ledger.mark_host_load_rank_drained(SID, OWNER, **first)
    assert ledger.mark_host_load_rank_drained(SID, OWNER, **peer)
    assert ledger.get(SID)["state"] == "failed"


def test_p2d_scheduler_release_waits_for_ack_without_blocking(connection, monkeypatch):
    from sglang.srt.disaggregation.p2d_host_staging import AgenticPToDHostStagingManager
    server, client, services = connection
    ledger = RemoteHostStagingLedger(client, "p2d")
    manager = object.__new__(AgenticPToDHostStagingManager)
    manager.ledger, manager.tp_size = ledger, 1
    manager._lock = threading.RLock()
    manager._active, manager._results, manager._prepared = {}, {}, {}
    manager._candidates = {"p2d:7": object()}
    manager._candidate_wakeup = threading.Event()
    req = SimpleNamespace(bootstrap_room=7)
    entered, release = threading.Event(), threading.Event()
    original = server._services[services[1].service]["arbitrate_p2d_release"]
    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return original(*args, **kwargs)
    monkeypatch.setitem(server._services[services[1].service], "arbitrate_p2d_release", delayed)
    try:
        assert not manager.prepare_scheduler_release(req)
        assert entered.wait(3)
        assert not manager.prepare_scheduler_release(req)
        assert not getattr(req, "_agentic_p2d_host_terminal", False)
        assert "p2d:7" in manager._candidates
    finally:
        release.set()
    assert manager._candidate_wakeup.wait(3)
    assert manager.prepare_scheduler_release(req)
    assert req._agentic_p2d_host_terminal
    assert not manager._candidates


def test_p2d_scheduler_never_waits_for_offer_worker_lock():
    from sglang.srt.disaggregation.p2d_host_staging import AgenticPToDHostStagingManager
    manager = object.__new__(AgenticPToDHostStagingManager)
    manager.ledger = SimpleNamespace(is_event_control=True)
    manager._lock = threading.RLock()
    req = SimpleNamespace(bootstrap_room=7)
    locked, release = threading.Event(), threading.Event()
    def owner():
        with manager._lock:
            locked.set()
            assert release.wait(3)
    thread = threading.Thread(target=owner)
    thread.start()
    try:
        assert locked.wait(3)
        assert not manager.prepare_scheduler_release(req)
        assert not manager.cancel_watch(req)
    finally:
        release.set()
        thread.join(3)


def test_prestart_retry_posts_frozen_identity_and_worker_completes():
    manager = object.__new__(AgenticPHostStagingManager)
    manager.ledger = SimpleNamespace(is_event_control=True)
    manager._state_lock = threading.RLock()
    manager._prestart_remote_retries = {}
    manager._control_wakeup = threading.Event()
    manager._notify_scheduler = lambda *args: None
    parent = SimpleNamespace(snapshot_id=SID)
    entry = {"recovery_claim_id": "old", "remote_read_epoch": 4}
    req = SimpleNamespace(rid="request")
    calls = []
    def finish(worker_req, worker_parent, frozen):
        assert worker_req is not req
        calls.append(dict(frozen))
        return True
    manager._complete_prestart_remote_retry = finish
    assert not manager._fence_prestart_remote_retry(req, parent, entry)
    assert not calls  # Scheduler did not perform cancellation or network RPC.
    entry["remote_read_epoch"] = 99
    manager._progress_prestart_remote_retries()
    assert calls == [{"recovery_claim_id": "old", "remote_read_epoch": 4}]
    assert manager._fence_prestart_remote_retry(
        req, parent, {"recovery_claim_id": "old", "remote_read_epoch": 4},
    )
    assert not manager._prestart_remote_retries


@pytest.mark.parametrize("size", [2, 8])
@pytest.mark.parametrize("cancel_before_admit", [False, True])
@pytest.mark.parametrize("hold_multiple", [False, True])
def test_slow_handed_ack_is_not_rank_local_prefill_admission(
    connection, monkeypatch, size, cancel_before_admit, hold_multiple,
):
    """Delay one real broker ACK: no rank may enter Forward independently."""
    server, client, services = connection
    ledger = RemoteHostStagingLedger(client, "d2p")
    ready(ledger, size)
    commands = [dict(claim(ledger, rank, size), remote_read_epoch=1) for rank in range(size)]
    for command in commands:
        assert ledger.mark_d2p_recovery_phase_rank(
            SID, OWNER, phase="io_inflight", **{k: v for k, v in command.items() if k != "remote_read_epoch"},
        )
        assert ledger.complete_d2p_host_load_rank(SID, OWNER, **command)
    for command in commands:
        assert ledger.complete_host_bind_rank(SID, OWNER, **command)
    managers, requests = [], []
    parent = SimpleNamespace(snapshot_id=SID)
    for rank, command in enumerate(commands):
        manager = object.__new__(AgenticPHostStagingManager)
        manager.ledger, manager.owner = ledger, OWNER
        manager.tp_rank, manager.tp_size = rank, size
        manager._state_lock = threading.RLock()
        manager._control_wakeup = threading.Event()
        manager.loads, manager.host_ready = {}, {}
        manager._h2d_lane_reservations = {SID: 0}
        manager.tp_host_commit_snapshots = [SID]
        manager.tp_host_admit_snapshots = []
        manager.workset_broker = SimpleNamespace(
            handoff_to_req=lambda *args: None, abort_bind=lambda *args, **kwargs: None,
        )
        manager.tree_cache = SimpleNamespace()
        req = SimpleNamespace(
            rid="child", _agentic_host_rank_loaded=True, _agentic_host_rank_token_count=16,
            _agentic_host_workset_lease=SimpleNamespace(owner=command["claim_id"], lease_id=command["lease_id"]),
            _agentic_host_remote_read_epoch=1,
        )
        managers.append(manager)
        requests.append(req)
    entered, release = threading.Event(), threading.Event()
    blocked_rank = size // 2 if hold_multiple else size - 1
    original = server._services[services[0].service]["mark_d2p_recovery_phase_rank"]
    def delayed(*args, **kwargs):
        if kwargs.get("phase") == "handed" and kwargs["tp_rank"] == blocked_rank:
            entered.set()
            assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setitem(server._services[services[0].service], "mark_d2p_recovery_phase_rank", delayed)
    try:
        assert all(manager.gate_request(req, parent) is True for manager, req in zip(managers, requests))
        assert entered.wait(3)
        # Every earlier RPC completed while the last rank's ACK is held.
        for key, future in list(ledger._pending.items()):
            if json.loads(key)[2]["tp_rank"] < blocked_rank:
                future.result(timeout=3)
        assert all(manager.gate_request(req, parent) is True for manager, req in zip(managers, requests))
        assert all(req._agentic_host_handoff_ready for req in requests[:blocked_rank])
        assert all(not getattr(req, "_agentic_host_handoff_ready", False) for req in requests[blocked_rank:])
        assert not any(getattr(req, "_agentic_kv_gate_complete", False) for req in requests)
        assert all(hasattr(req, "_agentic_host_workset_lease") for req in requests)
    finally:
        release.set()
    for future in list(ledger._pending.values()):
        future.result(timeout=3)
    assert all(manager.gate_request(req, parent) is True for manager, req in zip(managers, requests))
    assert all(req._agentic_host_handoff_ready for req in requests)
    if cancel_before_admit:
        for manager, req in zip(managers, requests):
            manager.rollback_bound_parent(req, parent)
            assert not hasattr(req, "_agentic_host_handoff_ready")
            assert not hasattr(req, "_agentic_host_workset_lease")
            assert not manager._h2d_lane_reservations
        return
    def forbidden(*args, **kwargs):
        raise AssertionError("final TP admission attempted metadata RPC")
    monkeypatch.setattr(ledger, "poll_call", forbidden)
    # Once the leader admitted exact handoff-ready ranks, a lagging advisory
    # mirror must not turn one rank back into a queue waiter.
    monkeypatch.setattr(ledger, "get", lambda sid: {"state": "hbm_ready"})
    for manager, req in zip(managers, requests):
        # Same native rank0 command observed by every physical rank.
        manager.tp_host_admit_snapshots = [SID]
        assert manager.gate_request(req, parent) is False
        assert req._agentic_kv_gate_complete
        assert not hasattr(req, "_agentic_host_handoff_ready")
        assert not manager._h2d_lane_reservations
