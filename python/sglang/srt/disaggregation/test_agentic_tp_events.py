"""CPU-only protocol/fence tests. No model, CUDA or shared filesystem control."""

import multiprocessing as mp
import json
import queue
import threading

import pytest

from sglang.srt.disaggregation.agentic_tp_events import (
    ControlUnavailable,
    EventKey,
    TPEventClient,
    TPEventServer,
    _ConnectionProgress,
    _Peer,
)


def test_report_contract_cannot_change_after_scheduler_state():
    server = TPEventServer("mode-run", "secret")
    client = TPEventClient(
        server.address, run_id="mode-run", token="secret", group="P", rank=0, size=1
    )
    key = EventKey("generation", "attempt")
    try:
        client.wait_ready()
        client.report_state("host", key, -1)
        client.report_state("host", key, 5)
        client.flush()
        assert client.local_status("host", key) == 5
        client.report("host", key, 6)
        with pytest.raises(ControlUnavailable):
            client.flush()
    finally:
        client.close()
        server.close()


def connect(server, rank=0, size=2, group="prefill"):
    client = TPEventClient(
        server.address, run_id="run", token="secret", group=group, rank=rank, size=size
    )
    client.wait_ready()
    return client


def wait(client, predicate, timeout=5):
    # Condition, not sleep/poll: evaluate predicate under the cache lock so a
    # notification cannot be lost between checking state and going to sleep.
    with client._condition:
        assert client._condition.wait_for(predicate, timeout)


@pytest.fixture
def server():
    value = TPEventServer("run", "secret")
    yield value
    value.close()


def test_progress_negative_sticky_reordered_and_attempt_isolation(server):
    a, b = connect(server), connect(server, 1)
    key, newer = EventKey("request:4", "try1"), EventKey("request:4", "try2")
    try:
        a.report("host", key, 3)
        a.report("host", key, 1)  # delayed progress cannot go backwards
        b.report("host", key, -1)
        b.report("host", key, 4)  # delayed success cannot erase failure
        a.report("host", newer, 1)
        a.flush()
        b.flush()
        wait(a, lambda: a.group_status("host", key) == -1)
        assert a.entry("host", key)["reports"] == {"0": 3, "1": -1}
        assert a.group_status("host", newer) is None
        assert a.entry("other", key) is None
    finally:
        a.close()
        b.close()


def test_sparse_failure_does_not_fabricate_group_completion(server):
    leader, follower = connect(server), connect(server, 1)
    key = EventKey("cancelled:0", "native-cancel")
    try:
        assert not leader.any_negative_report("admission", key)
        follower.report_state("admission", key, -1)
        wait(leader, lambda: leader.any_negative_report("admission", key))
        assert leader.group_status("admission", key) is None
        with pytest.raises(ValueError, match="rank zero"):
            follower.any_negative_report("admission", key)
        # A different attempt and an explicit retirement do not inherit it.
        assert not leader.any_negative_report("admission", EventKey("cancelled:0", "new"))
        leader.clear("admission", key)
        leader.flush()
        assert not leader.any_negative_report("admission", key)
    finally:
        leader.close()
        follower.close()


def test_empty_exception_latches_once_and_disconnects_whole_group(server):
    leader, follower = connect(server), connect(server, 1)
    try:
        server._fail("prefill", queue.Full())
        server._fail("prefill", OSError("secondary closed socket"))
        wait(leader, lambda: leader._error is not None)
        wait(follower, lambda: follower._error is not None)
        assert list(server.errors) == [("prefill", "Full: ")]
        assert server._groups["prefill"]["failed"] == "Full: "
        with pytest.raises(ControlUnavailable):
            leader.group_status("admission", EventKey("r:0", "attempt"))
    finally:
        leader.close()
        follower.close()


class _SendTimeoutSocket:
    def __init__(self, sock):
        self.sock = sock

    def __getattr__(self, name):
        return getattr(self.sock, name)

    def sendall(self, frame):
        raise TimeoutError(110, "Connection timed out")


def test_diagnostics_tolerate_peer_closing_during_accept():
    class ClosedSocket:
        def getsockname(self):
            raise OSError("socket closed")

        getpeername = getsockname

    progress = _ConnectionProgress(ClosedSocket())
    assert progress.local is None and progress.remote is None


def test_send_timeout_diagnostic_precedes_mirror_lock_and_is_one_shot(server, monkeypatch, caplog):
    client = connect(server, size=1)
    try:
        client.flush()
        client._socket = _SendTimeoutSocket(client._socket)
        captured = threading.Event()
        capture = client._progress.capture_failure

        def capture_and_signal(**kwargs):
            result = capture(**kwargs)
            captured.set()
            return result

        monkeypatch.setattr(client._progress, "capture_failure", capture_and_signal)
        with client._condition:
            client.report("private-namespace", EventKey("private-request", "private-attempt"), 1)
            # The failing sender must capture evidence even though the mirror
            # lock (also used by the scheduler) is currently held elsewhere.
            assert captured.wait(2)
            diagnostic = client._progress.failure
            assert diagnostic["origin"] == "send"
            assert diagnostic["send_stage"] == "sendall"
            assert diagnostic["send_operation"] == "report"
            assert diagnostic["local"] and diagnostic["remote"] == server.address
            assert diagnostic["acked_seq"] > 0
            assert diagnostic["send_seq"] > diagnostic["last_send_seq"]
            assert diagnostic["last_receive_age_s"] is not None
        wait(client, lambda: client._error is not None)
        first = dict(diagnostic)
        client._fail(OSError("secondary shutdown"), origin="recv")
        assert client._progress.failure == first
        assert "Connection timed out" in client._error
        assert "private-request" not in json.dumps(diagnostic)
        assert "private-attempt" not in json.dumps(diagnostic)
        assert "private-namespace" not in json.dumps(diagnostic)
        assert "secret" not in json.dumps(diagnostic)
        messages = [record.message for record in caplog.records
                    if "TP event client failed:" in record.message]
        assert len(messages) == 1
    finally:
        client.close()


def test_client_apply_failure_not_reported_as_socket_failure(server):
    client = connect(server, size=1)
    try:
        server._groups["prefill"]["peers"][0].send({"type": "private-invalid-payload"})
        wait(client, lambda: client._error is not None)
        diagnostic = client._progress.failure
        assert diagnostic["origin"] == "apply"
        assert diagnostic["recv_stage"] == "apply"
        assert diagnostic["recv_operation"] == "unknown"
        assert "private-invalid-payload" not in json.dumps(diagnostic)
        assert diagnostic["last_apply_age_s"] is not None
    finally:
        client.close()


def test_client_socket_loss_records_receive_origin(server):
    client = connect(server, size=1)
    try:
        server._groups["prefill"]["peers"][0].sock.shutdown(2)
        wait(client, lambda: client._error is not None)
        assert client._progress.failure["origin"] == "recv"
        assert client._progress.failure["recv_stage"] == "socket_read"
    finally:
        client.close()


def test_server_apply_diagnostic_identifies_peer_and_ack_semantics(server, monkeypatch):
    client = connect(server, size=1)
    peer = server._groups["prefill"]["peers"][0]
    try:
        def fail_apply(*args):
            raise RuntimeError("injected apply failure")

        monkeypatch.setattr(server, "_apply", fail_apply)
        client.report("host", EventKey("r", "attempt"), 1)
        wait(client, lambda: client._error is not None)
        diagnostic = peer.progress.failure
        assert diagnostic["origin"] == "apply"
        assert diagnostic["recv_stage"] == "apply"
        assert diagnostic["recv_operation"] == "report"
        assert diagnostic["group"] == "prefill" and diagnostic["rank"] == 0
        assert diagnostic["last_receive_seq"] == 1
        assert diagnostic["ack_semantics"] == "enqueued"
    finally:
        client.close()


def test_pending_cumulative_views_bound_tp8_c128_burst_without_losing_fences():
    peer = _Peer(None, 1024)  # A paused socket writer, unchanged queue bound.
    for rank in range(8):
        for i in range(128):
            entry = dict(identity=["prepared", f"r{i}", "attempt"], command_id=1,
                         reports={str(r): 1 for r in range(rank + 1)},
                         command_acks=list(range(rank + 1)))
            peer.send(dict(type="update", entry=entry), cumulative=True)
            peer.send(dict(type="ack", seq=rank * 128 + i + 1))
    # 1024 rank snapshots collapse to 128 exact identities, not an unbounded queue.
    assert len(peer.outbox._pending) == 129
    for i in range(128):
        value = json.loads(peer.outbox.get()[4:])
        assert value["entry"]["identity"][1] == f"r{i}"
        assert value["entry"]["reports"] == {str(r): 1 for r in range(8)}
        assert value["entry"]["command_acks"] == list(range(8))
    assert json.loads(peer.outbox.get()[4:]) == dict(type="ack", seq=1024)
    assert not peer.outbox._coalesced


def test_pending_views_preserve_barriers_command_order_and_free_before_reuse():
    peer = _Peer(None, 32)
    def update(reports, command_id=1, *, cumulative=True, **extra):
        entry = dict(identity=["workset", "r", "attempt"], command_id=command_id,
                     reports=reports, **extra)
        peer.send(dict(type="update", entry=entry), cumulative=cumulative)
        reports["mutated_after_enqueue"] = 99  # Queued frames must be frozen.
    update({"0": 1})
    peer.send(dict(type="ack", seq=1))
    update({"0": 1, "1": 1})
    update({"0": 1, "1": 1}, cumulative=False, receipt=-1)
    update({"0": 2, "1": 1})
    peer.send(dict(type="ack", seq=2))
    update({}, command_id=2, cumulative=False, command="FREE")
    peer.send(dict(type="update", entry=dict(identity=["workset", "next", "attempt"],
              command_id=1, reports={}, command="GRANT")))
    frames = [json.loads(peer.outbox.get()[4:]) for _ in range(6)]
    assert frames[0]["entry"]["reports"] == {"0": 1, "1": 1}
    assert frames[1]["entry"]["receipt"] == -1
    assert frames[2]["entry"]["reports"] == {"0": 2, "1": 1}
    assert frames[3] == dict(type="ack", seq=2)
    assert [frame["entry"]["command"] for frame in frames[4:]] == ["FREE", "GRANT"]
    assert not peer.outbox._coalesced


def test_replaceable_states_and_distinct_commands_never_coalesce():
    peer = _Peer(None, 3)
    entry = dict(identity=["state", "r", "attempt"], command_id=0, reports={"0": -1})
    peer.send(dict(type="update", entry=entry))
    entry["reports"] = {"0": 1}
    peer.send(dict(type="update", entry=entry))
    entry["command_id"] = 1
    peer.send(dict(type="update", entry=entry), cumulative=True)
    entry["command_id"] = 2
    with pytest.raises(queue.Full, match="pending-state"):
        peer.send(dict(type="update", entry=entry), cumulative=True)
    assert [json.loads(peer.outbox.get()[4:])["entry"]["reports"]["0"] for _ in range(3)] == [-1, 1, 1]


def test_transfer_failure_waits_all_physical_fences_and_rollback(server):
    a, b = connect(server), connect(server, 1)
    key = EventKey("request:5", "attempt")
    codes = dict(failed=0, success=4, transferring=3)
    try:
        a.report_transfer("direct", key, 0, **codes)
        b.report_transfer("direct", key, 3, **codes)
        a.flush()
        b.flush()
        wait(a, lambda: a.transfer_group_status("direct", key) == (3, True))
        b.report_transfer("direct", key, 4, **codes)
        b.flush()
        wait(a, lambda: a.transfer_group_status("direct", key) == (0, True))
        a.report_rollback("direct", key)
        a.flush()
        assert not a.rollback_group_complete("direct", key)
        b.report_rollback("direct", key)
        b.flush()
        wait(a, lambda: a.rollback_group_complete("direct", key))
    finally:
        a.close()
        b.close()


def test_latejoin_receipt_command_and_cleared_tombstone(server):
    a = connect(server)
    key = EventKey("request:7", "attempt")
    a.report("host", key, 2)
    a.publish_command("host", key, {"action": "START", "owner": "P0"})
    a.publish_receipt("host", key, 2)
    a.flush()
    b = connect(server, 1)
    try:
        assert b.entry("host", key)["reports"] == {}
        assert b.entry("host", key)["command"]["action"] == "START"
        assert b.entry("host", key)["receipt"] == 2
        with pytest.raises(ValueError, match="rank zero"):
            b.clear("host", key)
        with pytest.raises(ValueError, match="rank zero"):
            b.publish_receipt("host", key, 9)
        a.ack_command("host", key, 1)
        b.ack_command("host", key, 1)
        a.flush()
        b.flush()
        wait(a, lambda: a.command_complete("host", key))
        a.clear("host", key)
        a.flush()
        b.report("host", key, 4)  # late report after final cleanup
        b.flush()
        wait(b, lambda: b.entry("host", key) is None)
        assert a.entry("host", key) is None
    finally:
        a.close()
        b.close()


def test_socket_loss_invalidates_cached_success_and_refuses_reconnect(server):
    a, b = connect(server), connect(server, 1)
    key = EventKey("request:3", "attempt")
    a.report("host", key, 4)
    b.report("host", key, 4)
    a.flush()
    b.flush()
    wait(a, lambda: a.group_status("host", key) == 4)
    b.close()
    wait(a, lambda: a._error is not None)
    with pytest.raises(ControlUnavailable):
        a.group_status("host", key)
    with pytest.raises(ControlUnavailable):
        a.publish_receipt("host", key, 4)
    replacement = TPEventClient(
        server.address, run_id="run", token="secret", group="prefill", rank=1, size=2
    )
    try:
        with pytest.raises(ControlUnavailable):
            replacement.wait_ready()
    finally:
        replacement.close()
        a.close()


def test_run_auth_isolation_and_entry_bound(server):
    wrong = TPEventClient(
        server.address, run_id="oldrun", token="secret", group="prefill", rank=0, size=1
    )
    try:
        with pytest.raises(ControlUnavailable):
            wrong.wait_ready()
    finally:
        wrong.close()
    bounded = TPEventServer("run", "secret", max_entries=1)
    a = connect(bounded, size=1)
    try:
        first = EventKey("a:0", "attempt")
        a.report("host", first, 1)
        a.clear("host", first)
        a.flush()
        a.report("host", EventKey("b:0", "attempt"), 1)
        with pytest.raises(ControlUnavailable):
            a.flush()
        assert "limit" in bounded.errors[-1][1]
    finally:
        a.close()
        bounded.close()


def _rank_process(address, rank, size, done, release, result):
    client = None
    try:
        client = TPEventClient(
            address, run_id="run", token="secret", group="decode", rank=rank, size=size
        )
        client.wait_ready()
        key = EventKey("cross-process:0", "attempt2")
        with client._condition:
            assert client._condition.wait_for(
                lambda: client.entry("d2h", key)
                and client.entry("d2h", key)["command"] == "START",
                10,
            )
        client.report("d2h", key, 1)
        client.report("d2h", key, 2)
        client.report_rollback("d2h", key)
        client.flush()
        done.put(rank)
        assert release.wait(15)
        result.put((rank, None))
    except Exception as exc:
        result.put((rank, repr(exc)))
    finally:
        if client:
            client.close()


@pytest.mark.parametrize("size", [2, 8])
def test_real_process_group_commands_and_all_shard_completion(server, size):
    ctx = mp.get_context("spawn")
    done, result, release = ctx.Queue(), ctx.Queue(), ctx.Event()
    a = connect(server, size=size, group="decode")
    processes = [
        ctx.Process(
            target=_rank_process,
            args=(server.address, rank, size, done, release, result),
        )
        for rank in range(1, size)
    ]
    key = EventKey("cross-process:0", "attempt2")
    try:
        for process in processes:
            process.start()
        a.publish_command("d2h", key, "START")
        a.report("d2h", key, 2)
        a.report_rollback("d2h", key)
        a.flush()
        assert {done.get(timeout=15) for _ in processes} == set(range(1, size))
        wait(a, lambda: a.group_status("d2h", key) == 2)
        wait(a, lambda: a.rollback_group_complete("d2h", key))
        a.publish_receipt("d2h", key, 2)
        a.flush()
        release.set()
        assert all(result.get(timeout=15)[1] is None for _ in processes)
    finally:
        release.set()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        a.close()
        done.close()
        result.close()


def test_local_reads_return_detached_mirror(server):
    a = connect(server, size=1)
    key = EventKey("local:0", "attempt")
    try:
        a.report("host", key, 1)
        a.flush()
        # The cache access takes no blocking socket path and returns a detached
        # value; consumer mutation cannot corrupt the authoritative mirror.
        value = a.entry("host", key)
        value["reports"]["0"] = 999
        assert a.group_status("host", key) == 1
    finally:
        a.close()


def test_client_backpressure_fails_closed_without_silent_drop(server, monkeypatch):
    a = connect(server, size=1)
    key = EventKey("pressure:0", "attempt")
    try:
        a.report("host", key, 1)
        a.flush()

        def full(_):
            raise queue.Full

        monkeypatch.setattr(a._outbox, "put_nowait", full)
        with pytest.raises(ControlUnavailable, match="queue full"):
            a.report("host", key, 2)
        with pytest.raises(ControlUnavailable):
            a.local_status("host", key)
    finally:
        monkeypatch.undo()
        a.close()


def test_server_backpressure_invalidates_entire_group(server, monkeypatch):
    a, b = connect(server), connect(server, 1)
    try:
        with server._lock:
            peer = server._groups["prefill"]["peers"][1]

            def full(_, **kwargs):
                raise queue.Full("slow subscriber")

            monkeypatch.setattr(peer, "send", full)
        a.publish_command("host", EventKey("pressure:0", "attempt"), "START")
        wait(a, lambda: a._error is not None)
        wait(b, lambda: b._error is not None)
        with pytest.raises(ControlUnavailable):
            a.group_status("host", EventKey("pressure:0", "attempt"))
    finally:
        a.close()
        b.close()


def test_reports_only_go_to_leader_and_publisher_and_are_deduplicated(server):
    a, b, c = [connect(server, rank, size=3) for rank in range(3)]
    key = EventKey("leader-only:0", "attempt")
    try:
        b.report("host", key, 2)
        b.flush()
        sequence = b._seq
        b.report("host", key, 2)
        b.report("host", key, 1)
        assert b._seq == sequence  # unchanged/stale report did not hit socket
        wait(a, lambda: a.local_status("host", key, rank=1) == 2)
        assert b.local_status("host", key) == 2
        assert c.entry("host", key) is None  # not merely filtered payload
        for client in (b, c):
            with pytest.raises(ValueError, match="rank zero"):
                client.group_status("host", key)
            with pytest.raises(ValueError, match="rank zero"):
                client.transfer_group_status("host", key)
            with pytest.raises(ValueError, match="rank zero"):
                client.rollback_group_complete("host", key)
        a.publish_command("host", key, "START")
        a.flush()
        wait(c, lambda: c.command("host", key) == "START")
        assert c.entry("host", key)["reports"] == {}
    finally:
        a.close()
        b.close()
        c.close()


@pytest.mark.parametrize(
    "rank,size", [(False, 2), (0, True), (0.0, 2), (0, 2.0), (0, 257)]
)
def test_client_rejects_noninteger_rank_and_size(server, rank, size):
    with pytest.raises(ValueError):
        connect(server, rank, size)


def test_commands_cannot_overwrite_unconsumed_phase(server):
    a, b = connect(server), connect(server, 1)
    key = EventKey("commands:0", "attempt")
    try:
        a.publish_command("host", key, "PREPARE")
        a.flush()
        wait(b, lambda: b.command("host", key) == "PREPARE")
        assert b.next_command("host", key) == (1, "PREPARE")
        assert b.next_command("host", key) is None
        a.ack_command("host", key, 1)
        a.flush()
        assert not a.command_complete("host", key)
        # Follower has not completed PREPARE: overwriting it is a protocol bug,
        # not permission to skip PREPARE and start DMA against unprepared pages.
        a.publish_command("host", key, "START")
        with pytest.raises(ControlUnavailable):
            a.flush()
        assert "not acknowledged" in server.errors[-1][1]
    finally:
        a.close()
        b.close()


def test_commands_ack_all_ranks_before_next_and_exact_retry(server):
    a, b = connect(server), connect(server, 1)
    key = EventKey("commands:0", "attempt")
    try:
        a.publish_command("host", key, "PREPARE", command_id=1)
        a.publish_command("host", key, "PREPARE", command_id=1)
        a.flush()
        wait(b, lambda: b.command("host", key) == "PREPARE")
        assert a.next_command("host", key) == (1, "PREPARE")
        assert b.next_command("host", key) == (1, "PREPARE")
        a.ack_command("host", key, 1)
        b.ack_command("host", key, 1)
        a.flush()
        b.flush()
        wait(a, lambda: a.command_complete("host", key))
        a.publish_command("host", key, "START")
        a.flush()
        wait(b, lambda: b.command("host", key) == "START")
        assert b.next_command("host", key) == (2, "START")
        assert not a.command_complete("host", key)
    finally:
        a.close()
        b.close()


def test_unexpected_server_worker_exception_is_fail_closed(server, monkeypatch):
    a, b = connect(server), connect(server, 1)
    try:

        def crash(*_):
            raise RuntimeError("injected control worker failure")

        monkeypatch.setattr(server, "_apply", crash)
        a.report("host", EventKey("fault:0", "attempt"), 1)
        wait(a, lambda: a._error is not None)
        wait(b, lambda: b._error is not None)
        assert "injected control worker failure" in server.errors[-1][1]
    finally:
        a.close()
        b.close()


def test_cross_group_receipts_are_explicit_and_observer_cleanup_is_not_producer_cleanup():
    server = TPEventServer(
        "run", "secret", receipt_observers={("decode", "p2d-receiver"): ["prefill"]}
    )
    d0, d1 = [connect(server, rank, group="decode") for rank in range(2)]
    p0, p1 = [connect(server, rank, group="prefill") for rank in range(2)]
    key = EventKey("request@123", "bootstrap:123")
    try:
        d0.report("p2d-receiver", key, 3)
        d1.report("p2d-receiver", key, 3)
        d0.publish_receipt("p2d-receiver", key, 4)
        d0.flush()
        wait(p0, lambda: p0.observed_receipt("decode", "p2d-receiver", key) == 4)
        wait(p1, lambda: p1.observed_receipt("decode", "p2d-receiver", key) == 4)
        assert p0.entry("p2d-receiver", key) is None
        p0.clear_observed("decode", "p2d-receiver", key)
        p0.flush()
        wait(p1, lambda: p1.observed_receipt("decode", "p2d-receiver", key) is None)
        assert d0.receipt("p2d-receiver", key) == 4
        assert d0.local_status("p2d-receiver", key, rank=0) == 3
        d0.publish_receipt("p2d-receiver", key, 4)
        d0.flush()
        assert p0.observed_receipt("decode", "p2d-receiver", key) is None
        with pytest.raises(ValueError, match="mapping"):
            p0.observed_receipt("unknown", "p2d-receiver", key)
        with pytest.raises(ValueError, match="rank zero"):
            p1.clear_observed("decode", "p2d-receiver", key)
    finally:
        for client in (p0, p1, d0, d1):
            client.close()
        server.close()


def test_late_join_receipt_observer_and_source_failure():
    server = TPEventServer(
        "run", "secret", receipt_observers={("decode", "p2d-receiver"): ["prefill"]}
    )
    d0 = connect(server, size=1, group="decode")
    key = EventKey("request@123", "bootstrap:123")
    d0.publish_receipt("p2d-receiver", key, 4)
    d0.flush()
    p0 = connect(server, size=1, group="prefill")
    try:
        assert p0.observed_receipt("decode", "p2d-receiver", key) == 4
        d0.close()
        wait(p0, lambda: "decode" in p0._observer_failures)
        with pytest.raises(ControlUnavailable):
            p0.observed_receipt("decode", "p2d-receiver", key)
    finally:
        p0.close()
        d0.close()
        server.close()
