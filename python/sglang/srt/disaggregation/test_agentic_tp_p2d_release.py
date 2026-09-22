"""R15: release authorization/marker I/O must not wait on scheduler threads."""
import queue
import threading
from collections import deque
from types import SimpleNamespace

import pytest

from sglang.srt.disaggregation.agentic_tp_control import TPGroupMailbox
from sglang.srt.disaggregation.agentic_tp_events import TPEventClient, TPEventServer
from sglang.srt.disaggregation.agentic_tp_socket_mailbox import SocketTPGroupMailbox
from sglang.srt.disaggregation.agentic_tp import request_generation_key
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.p2d_host_staging import AgenticPToDHostStagingManager
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin as Prefill
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler


def forbidden(*args, **kwargs):
    raise AssertionError("unexpected scheduler I/O or repeated authorization")


def test_socket_grammar_abort_never_decides_release_from_rank_local_rpc(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "enabled")
    req = SimpleNamespace()
    manager = SimpleNamespace(cancel_watch=forbidden)
    assert not Prefill._prefill_grammar_release_safe(req, manager)
    assert not Prefill._prefill_grammar_release_safe(req, None)


def test_legacy_grammar_abort_keeps_existing_release_decision(monkeypatch):
    monkeypatch.delenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", raising=False)
    req = SimpleNamespace()
    assert Prefill._prefill_grammar_release_safe(req, None)
    for safe in (False, True):
        manager = SimpleNamespace(cancel_watch=lambda _: safe)
        assert Prefill._prefill_grammar_release_safe(req, manager) is safe


def prefill(mailbox, manager):
    return SimpleNamespace(
        tp_size=2, tp_rank=0, disaggregation_mode=DisaggregationMode.PREFILL,
        _AGENTIC_TP_CONTROL_KEY=Scheduler._AGENTIC_TP_CONTROL_KEY,
        _prefill_transfer_tp_background_enabled=True,
        _prefill_transfer_async_release_enabled=True,
        _prefill_transfer_key=Prefill._prefill_transfer_key,
        agentic_tp_direct_admission_active={}, agentic_early_direct_receives={},
        agentic_p2d_host_staging_manager=manager, agentic_host_staging_manager=None,
        agentic_tp_p2d_cleanup_mailbox=mailbox,
        agentic_tp_p2d_sender_mailbox=SimpleNamespace(
            publish_local=lambda *args: None,
            group_status=lambda k: int(KVPoll.Success),
            transfer_group_status=lambda k: (int(KVPoll.Success), False)),
        agentic_tp_p2d_receiver_mailbox=SimpleNamespace(receipt=forbidden),
        _publish_deferred_prefill_ready=forbidden,
    )


@pytest.mark.parametrize("outcome", [KVPoll.Success, KVPoll.Failed])
@pytest.mark.parametrize("tp_size", [2, 8])
def test_all_rank_release_barrier_not_physical_sender_success(tmp_path, outcome, tp_size):
    ranks = [TPGroupMailbox("p2d-cleanup", tp_rank=r, tp_size=tp_size,
                           directory=str(tmp_path)) for r in range(tp_size)]
    manager = SimpleNamespace(prepare_scheduler_release=lambda req: True,
                              cancel_watch=lambda req: True)
    s = prefill(ranks[0], manager)
    req = SimpleNamespace(rid="r", bootstrap_room=7)
    s.disagg_prefill_inflight_queue = [req]
    assert Scheduler._agentic_tp_prepare_admission_control(s)[
        "prefill_transfer_statuses"] == [int(KVPoll.Transferring)]
    assert Prefill._prefill_transfer_authorize_release(s, req, int(outcome)) == outcome
    assert Scheduler._agentic_tp_prepare_admission_control(s)[
        "prefill_transfer_statuses"] == [int(KVPoll.Transferring)]
    for rank in ranks[1:]:
        rank.publish_local("release-ready:" + request_generation_key("r", 7), int(outcome))
    assert Scheduler._agentic_tp_prepare_admission_control(s)[
        "prefill_transfer_statuses"] == [int(outcome)]
    # Release-ready is not the old already-freed acknowledgement.
    assert ranks[0].group_status(request_generation_key("r", 7)) is None


@pytest.mark.parametrize("outcome", [KVPoll.Success, KVPoll.Failed])
def test_publication_retry_does_not_repeat_consuming_cas_or_change_failure(outcome):
    calls = []
    def authorize(req):
        calls.append(1)
        req._agentic_p2d_host_terminal = True
        return True
    m = SimpleNamespace(prepare_scheduler_release=authorize, cancel_watch=authorize)
    mb = SimpleNamespace(publish_local=forbidden)
    s = prefill(mb, m)
    req = SimpleNamespace(rid="r", bootstrap_room=7)
    with pytest.raises(AssertionError):
        Prefill._prefill_transfer_authorize_release(s, req, int(outcome))
    m.prepare_scheduler_release = m.cancel_watch = forbidden
    mb.publish_local = lambda *args: None
    assert Prefill._prefill_transfer_progress_tp_req_once(s, req) == outcome
    assert Prefill._prefill_transfer_authorize_release(s, req, int(outcome)) == outcome
    assert calls == [1]


def test_live_host_claim_does_not_authorize_release():
    s = prefill(SimpleNamespace(publish_local=forbidden),
                SimpleNamespace(prepare_scheduler_release=lambda req: False))
    req = SimpleNamespace(rid="r", bootstrap_room=7)
    assert Prefill._prefill_transfer_authorize_release(s, req, int(KVPoll.Success)) == KVPoll.Transferring
    assert not hasattr(req, "_agentic_p2d_release_authorized")


def test_controller_p2d_broadcast_consumes_only_native_released_events():
    s = prefill(SimpleNamespace(), None)
    s._prefill_native_release_enabled = True
    s._prefill_transfer_poll_lock = threading.Lock()
    s.disagg_prefill_inflight_queue = [
        SimpleNamespace(rid="old", bootstrap_room=1),
        SimpleNamespace(rid="ready", bootstrap_room=2),
    ]
    s._prefill_native_completed = {
        request_generation_key("ready", 2): int(KVPoll.Success)
    }
    s.agentic_tp_p2d_sender_mailbox.group_status = forbidden
    s.agentic_tp_p2d_sender_mailbox.transfer_group_status = forbidden
    control = Scheduler._agentic_tp_prepare_admission_control(s)
    assert control["prefill_transfer_keys"] == [("ready", 2)]
    assert control["prefill_transfer_statuses"] == [int(KVPoll.Success)]


def completed_host_manager(outcome):
    manager = AgenticPToDHostStagingManager.__new__(AgenticPToDHostStagingManager)
    manager._lock = threading.Lock()
    manager.ledger = SimpleNamespace(is_event_control=True)
    manager._results = {"p2d:7": int(outcome)}
    manager._active, manager._prepared, manager._candidates = {}, {}, {}
    return manager


@pytest.mark.parametrize("outcome", [KVPoll.Success, KVPoll.Failed])
def test_host_physical_terminal_survives_lock_contention_and_result_consumption(outcome):
    manager = completed_host_manager(outcome)
    req = SimpleNamespace(bootstrap_room=7, _agentic_p2d_host_snapshot_id="p2d:7")
    # Before observing the physical fence, contention cannot imply completion.
    with manager._lock:
        assert manager.poll(req) == KVPoll.Transferring
    assert manager.poll(req) == outcome
    manager._results.clear()
    with manager._lock:
        assert manager.poll(req) == outcome
    assert not getattr(req, "_agentic_p2d_host_terminal", False)
    assert not hasattr(req, "_agentic_p2d_release_authorized")
    # A fresh generation cannot inherit the old request's cached fence.
    other = SimpleNamespace(bootstrap_room=8, _agentic_p2d_host_snapshot_id="p2d:8")
    assert manager.poll(other) is None


@pytest.mark.parametrize("tp_size", [2, 8])
def test_socket_host_success_cannot_regress_while_release_lock_busy(tp_size):
    server = TPEventServer("p2d-terminal-test", "test-only")
    clients, schedulers, requests, managers = [], [], [], []
    key = request_generation_key("r", 7)
    try:
        for rank in range(tp_size):
            client = TPEventClient(server.address, run_id="p2d-terminal-test",
                                   token="test-only", group="P", rank=rank, size=tp_size)
            clients.append(client)
            client.wait_ready()
            def mailbox(namespace):
                return SocketTPGroupMailbox(namespace, tp_rank=rank,
                                            tp_size=tp_size, client=client)
            manager = completed_host_manager(KVPoll.Success)
            s = prefill(mailbox("p2d-cleanup"), manager)
            s.tp_size, s.tp_rank = tp_size, rank
            s.agentic_tp_p2d_sender_mailbox = mailbox("p2d-sender")
            req = SimpleNamespace(rid="r", bootstrap_room=7,
                                  _agentic_p2d_host_snapshot_id="p2d:7")
            schedulers.append(s)
            requests.append(req)
            managers.append(manager)
            Prefill._prefill_transfer_progress_tp_req_once(s, req)
            client.flush()

        leader = schedulers[0].agentic_tp_p2d_sender_mailbox
        with clients[0]._condition:
            assert clients[0]._condition.wait_for(
                lambda: leader.transfer_group_status(key)[0] == KVPoll.Success, 3)
        Prefill._prefill_transfer_progress_tp_req_once(schedulers[0], requests[0])
        clients[0].flush()
        for client, s, req, manager in zip(clients, schedulers, requests, managers):
            with client._condition:
                assert client._condition.wait_for(
                    lambda: s.agentic_tp_p2d_sender_mailbox.receipt(key) == KVPoll.Success, 3)
            with manager._lock:
                for _ in range(3):
                    assert Prefill._prefill_transfer_progress_tp_req_once(s, req) == KVPoll.Success
                    assert Prefill._prefill_transfer_authorize_release(s, req, int(KVPoll.Success)) == KVPoll.Transferring
                    client.flush()  # old code disconnects the entire TP group here
                assert not hasattr(req, "_agentic_p2d_release_authorized")
            assert Prefill._prefill_transfer_authorize_release(s, req, int(KVPoll.Success)) == KVPoll.Success
            # Consumed manager results cannot resurrect the transfer either.
            assert not manager._results
            assert manager.poll(req) == KVPoll.Success
            client.flush()
        release = schedulers[0].agentic_tp_p2d_cleanup_mailbox
        with clients[0]._condition:
            assert clients[0]._condition.wait_for(
                lambda: release.transfer_group_status("release-ready:" + key)[0] == KVPoll.Success, 3)
        assert not server.errors
    finally:
        for client in clients:
            client.close()
        server.close()


def test_failed_launch_publishes_physical_terminal_before_release_ready():
    events = []
    s = prefill(SimpleNamespace(publish_local=lambda k, status: events.append(("release", status))),
                SimpleNamespace(cancel_watch=lambda req: events.append(("cancel", None)) or True))
    s.agentic_tp_p2d_sender_mailbox.publish_local = lambda k, status: events.append(("physical", status))
    req = SimpleNamespace(rid="r", bootstrap_room=7)
    Prefill._prefill_transfer_authorize_release(s, req, int(KVPoll.Failed))
    assert events == [("physical", int(KVPoll.Failed)), ("cancel", None), ("release", int(KVPoll.Failed))]
    events.clear()
    Prefill._prefill_transfer_authorize_release(s, req, int(KVPoll.Failed))
    assert events == [("release", int(KVPoll.Failed))]


def test_live_host_winner_can_finish_after_native_failure_as_before():
    manager = SimpleNamespace(cancel_watch=lambda req: False,
                              prepare_scheduler_release=lambda req: True)
    s = prefill(SimpleNamespace(publish_local=lambda *a: None), manager)
    req = SimpleNamespace(rid="r", bootstrap_room=7)
    assert Prefill._prefill_transfer_authorize_release(s, req, int(KVPoll.Failed)) == KVPoll.Transferring
    # No ownership was released: the pre-existing Host-winning path may finish.
    assert not hasattr(req, "_agentic_p2d_release_authorized")
    assert Prefill._prefill_transfer_authorize_release(s, req, int(KVPoll.Success)) == KVPoll.Success


def test_consumer_exception_report_allows_peer_to_finish(tmp_path):
    sender = [TPGroupMailbox("p2d-sender", tp_rank=r, tp_size=2,
                            directory=str(tmp_path)) for r in range(2)]
    releases = [TPGroupMailbox("p2d-cleanup", tp_rank=r, tp_size=2,
                              directory=str(tmp_path)) for r in range(2)]
    key = request_generation_key("r", 7)
    for mb in sender:
        mb.publish_local(key, int(KVPoll.Bootstrapping))
    s = Prefill.__new__(Prefill)
    s.tp_size, s.tp_rank = 2, 1
    s._prefill_transfer_tp_background_enabled = True
    s._prefill_transfer_async_release_enabled = True
    s._prefill_transfer_stop = threading.Event()
    s._prefill_ready_condition = threading.Condition()
    s._prefill_ready_publish_condition = threading.Condition()
    s._prefill_ready_next_publish_sequence = 0
    s._prefill_transfer_poll_lock = threading.Lock()
    s._prefill_transfer_active_reqs = {}
    s._prefill_ready_queued_keys = {key}
    s._prefill_transfer_interval = 0.001
    s.agentic_tp_p2d_sender_mailbox = sender[1]
    s.agentic_p2d_host_staging_manager = None
    req = SimpleNamespace(rid="r", bootstrap_room=7,
                          disagg_kv_sender=SimpleNamespace(
                              fence_failed_launch=lambda error: int(KVPoll.Failed)))
    s._prefill_ready_queue = deque([req])
    def publish_release(k, value):
        releases[1].publish_local(k, value)
        s._prefill_transfer_stop.set()
    s.agentic_tp_p2d_cleanup_mailbox = SimpleNamespace(publish_local=publish_release)
    s._prefill_transfer_progress_tp_req_once = forbidden
    s._prefill_transfer_consumer_worker(0)
    assert sender[1].local_status(key) == KVPoll.Failed
    # Rank0 must see the failed physical report, not wait forever for metadata.
    leader = prefill(releases[0], None)
    leader.agentic_tp_p2d_sender_mailbox = sender[0]
    leader.agentic_tp_p2d_receiver_mailbox = SimpleNamespace(receipt=lambda key: None)
    peer_req = SimpleNamespace(rid="r", bootstrap_room=7, disagg_p_ready_notified=True,
                               disagg_kv_sender=SimpleNamespace(poll=lambda: int(KVPoll.Bootstrapping)))
    assert Prefill._prefill_transfer_progress_tp_req_once(leader, peer_req) == KVPoll.Failed
    Prefill._prefill_transfer_authorize_release(leader, peer_req, int(KVPoll.Failed))
    assert releases[0].transfer_group_status("release-ready:" + key) == (int(KVPoll.Failed), True)


def test_watch_registration_never_takes_ledger_lock_and_cannot_resurrect():
    m = AgenticPToDHostStagingManager.__new__(AgenticPToDHostStagingManager)
    m._async_watch_registration = True
    m._watch_registrations = queue.SimpleQueue()
    m._candidate_wakeup = threading.Event()
    m._lock = threading.Lock()
    m._active, m._prepared, m._results, m._candidates = {}, {}, {}, {}
    req = SimpleNamespace(bootstrap_room=7)
    with m._lock:
        assert m.watch(req, [])
    # Release wins before the offer worker consumes the queued registration.
    req._agentic_p2d_host_terminal = True
    m._drain_watch_registrations()
    assert m._candidates == {}


def marker_queue(rank):
    q = DecodePreallocQueue.__new__(DecodePreallocQueue)
    q._async_progress_enabled = q._async_ready_marker_enabled = True
    q._ready_marker_work = queue.SimpleQueue()
    q._ready_marker_pending = {}
    q.tp_size, q.tp_rank = 2, rank
    return q


def test_ready_marker_scheduler_does_no_io_and_waits_for_all_ranks(tmp_path, monkeypatch):
    path = tmp_path / "7.ready"
    path.touch()
    leader, follower = marker_queue(0), marker_queue(1)
    with monkeypatch.context() as patch:
        patch.setattr("os.open", forbidden)
        patch.setattr("os.unlink", forbidden)
        leader._consume_p_ready_marker(str(path))
    leader._background_consume_p_ready_markers()
    assert path.exists()
    follower._consume_p_ready_marker(str(path))
    follower._background_consume_p_ready_markers()
    assert path.exists()  # Only rank0 cleans up.
    leader._background_consume_p_ready_markers()
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_marker_partial_cleanup_retries_without_recreating_acks(tmp_path, monkeypatch):
    path = tmp_path / "7.ready"
    path.touch()
    leader, follower = marker_queue(0), marker_queue(1)
    for q in (follower, leader):
        q._consume_p_ready_marker(str(path))
    follower._background_consume_p_ready_markers()
    import os
    unlink = os.unlink
    def partial(p):
        if str(p).endswith("rank-1.admitted"):
            raise OSError("injected NFS failure")
        unlink(p)
    with monkeypatch.context() as patch:
        patch.setattr(os, "unlink", partial)
        leader._background_consume_p_ready_markers()
    assert not path.exists()
    assert leader._ready_marker_pending[str(path)] == [True, True]
    with monkeypatch.context() as patch:
        patch.setattr(os, "open", forbidden)
        leader._background_consume_p_ready_markers()
    assert list(tmp_path.iterdir()) == []


def test_admission_uses_cached_ready_without_remote_stat(monkeypatch):
    dr = SimpleNamespace(req=SimpleNamespace(rid="r", bootstrap_room=7),
                         waiting_for_input=True, _async_p_ready=True)
    prealloc = SimpleNamespace(queue=[dr], _async_progress_enabled=True,
                               _requires_p_ready=lambda req: True, p_ready_dir="/nfs")
    transfers = SimpleNamespace(queue=[])
    s = SimpleNamespace(tp_size=2, tp_rank=0, disaggregation_mode=DisaggregationMode.DECODE,
                        _AGENTIC_TP_CONTROL_KEY=Scheduler._AGENTIC_TP_CONTROL_KEY,
                        disagg_decode_prealloc_queue=prealloc,
                        disagg_decode_transfer_queue=transfers,
                        agentic_tp_p2d_receiver_mailbox=SimpleNamespace(),
                        agentic_tp_p2d_admission_mailbox=SimpleNamespace(
                            group_status=lambda key: int(KVPoll.Success)))
    monkeypatch.setattr("os.path.exists", forbidden)
    assert Scheduler._agentic_tp_prepare_admission_control(s)["decode_admit_keys"] == [("r", 7)]
