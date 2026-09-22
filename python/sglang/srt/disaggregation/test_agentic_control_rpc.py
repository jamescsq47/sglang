import queue
import threading
from concurrent.futures import TimeoutError

import pytest

from sglang.srt.disaggregation.agentic_control_rpc import (
    ControlRPCClient,
    ControlRPCServer,
    ControlUnavailable,
    RemoteCallError,
    get_control_client,
    set_control_rpc_blocking_allowed,
)


def test_deferred_start_and_scheduler_blocking_guard():
    server = ControlRPCServer("run", "secret", start=False)
    server.register_service("test", {"value": lambda: 4})
    server.start()
    server.start()
    client = connect(server)
    previous = set_control_rpc_blocking_allowed(False)
    try:
        with pytest.raises(RuntimeError, match="scheduler"):
            client.call("test", "value")
        with pytest.raises(RuntimeError, match="scheduler"):
            client.wait_for_record("test", "key", bool, timeout=0)
        assert client.submit("test", "value").result(timeout=3) == 4
    finally:
        set_control_rpc_blocking_allowed(previous)
        client.close()
        server.close()


def test_record_condition_multiple_waiters_timeout_and_disconnect(server):
    client = connect(server)
    results = []
    failures = []

    def waiter(key):
        try:
            results.append(client.wait_for_record("host", key, bool, timeout=3))
        except ControlUnavailable as exc:
            failures.append(exc)

    try:
        threads = [threading.Thread(target=waiter, args=("same",)) for _ in range(4)]
        for thread in threads:
            thread.start()
        server.publish("host", "same", {"ready": True})
        for thread in threads:
            thread.join(3)
        assert results == [{"ready": True}] * 4
        assert client.wait_for_record("host", "missing", bool, timeout=0.01) is None
        thread = threading.Thread(target=waiter, args=("never",))
        thread.start()
        server.close()
        thread.join(3)
        assert len(failures) == 1
    finally:
        client.close()


def test_mutation_timeout_is_uncertain_and_invalidates_cache(server):
    gate = threading.Event()
    entered = threading.Event()
    committed = []

    def slow():
        entered.set()
        gate.wait(3)
        committed.append("HOST")
        return "committed"

    server.register_service("host", {"slow": slow})
    client = connect(server, call_timeout=0.03)
    try:
        client.subscribe("host").ready.result(timeout=3)
        with pytest.raises(ControlUnavailable, match="uncertain"):
            client.call("host", "slow")
        assert entered.is_set()
        with pytest.raises(ControlUnavailable):
            client.cache_get("host", "snapshot")
        with pytest.raises(ControlUnavailable):
            client.submit("host", "slow")
    finally:
        gate.set()
        client.close()


def connect(server, **kwargs):
    client = ControlRPCClient(server.address, run_id="run", token="secret", **kwargs)
    client.wait_ready()
    return client


@pytest.fixture
def server():
    value = ControlRPCServer("run", "secret")
    yield value
    value.close()


def test_atomic_push_before_result_and_no_shared_mutable_values(server):
    data = {"lease": None}

    def claim(key, lease_id):
        data["lease"] = lease_id
        server.publish("host", key, data)
        return data

    server.register_service("host", {"claim": claim})
    client = connect(server)
    try:
        sub = client.subscribe("host", queue_events=True)
        sub.ready.result(timeout=3)
        result = client.call("host", "claim", "snapshot:4", lease_id="lease1")
        assert result == {"lease": "lease1"}
        assert client.cache_get("host", "snapshot:4") == result
        assert sub.get_nowait()["revision"] == 1
        result["lease"] = "caller mutation"
        data["lease"] = "handler mutation"
        assert client.cache_get("host", "snapshot:4") == {"lease": "lease1"}
    finally:
        client.close()


def test_late_subscription_streams_large_snapshot_without_giant_frame(server):
    # More than the framing limit in aggregate, but each metadata record fits.
    blob = "a" * 65536
    for i in range(80):
        server.publish("large", str(i), {"value": blob})
    client = connect(server)
    try:
        sub = client.subscribe("large", queue_events=True)
        server.publish("large", "0", {"value": "new"})
        sub.ready.result(timeout=5)
        # The new value is either inside the atomic snapshot or the next delta,
        # never older state overwriting an already received later value.
        done = client.submit("missing", "method")
        with pytest.raises(RemoteCallError):
            done.result(timeout=5)
        assert len(client.cache_snapshot("large")) == 80
        assert client.cache_get("large", "0") == {"value": "new"}
    finally:
        client.close()


def test_delete_and_ordered_events(server):
    server.register_service("records", {"publish": server.publish})
    client = connect(server)
    try:
        sub = client.subscribe("records", queue_events=True)
        sub.ready.result(timeout=3)
        for value in (1, 2, None):
            client.call("records", "publish", "records", "key", value)
        events = [sub.get_nowait() for _ in range(3)]
        assert [v["revision"] for v in events] == [1, 2, 3]
        assert [v["value"] for v in events] == [1, 2, None]
        assert client.cache_get("records", "key") is None
        assert client.cache_snapshot("records") == {}
    finally:
        client.close()


def test_committed_lost_reply_retry_returns_exact_result_once(server, monkeypatch):
    calls = []

    def allocate(snapshot, attempt):
        calls.append((snapshot, attempt))
        server.publish("host", snapshot, {"attempt": attempt, "lease_id": "lease1"})
        return {"lease_id": "lease1"}

    server.register_service("host", {"allocate": allocate})
    original = server._reply
    dropped = threading.Event()

    def drop_first(peer, response):
        if not dropped.is_set():
            dropped.set()
            return
        original(peer, response)

    monkeypatch.setattr(server, "_reply", drop_first)
    client = connect(server)
    try:
        client.subscribe("host").ready.result(timeout=3)
        first = client.submit("host", "allocate", "s:4", "attempt1")
        assert dropped.wait(3)
        with pytest.raises(TimeoutError):
            first.result(timeout=0.02)
        second = client.retry(first)
        assert second.operation_id == first.operation_id
        assert second.result(timeout=3) == {"lease_id": "lease1"}
        assert calls == [("s:4", "attempt1")]
        assert client.cache_get("host", "s:4")["lease_id"] == "lease1"
    finally:
        client.close()


def test_retired_retry_never_executes_again():
    server = ControlRPCServer("run", "secret", retry_window=1)
    calls = []
    server.register_service("test", {"append": lambda value: calls.append(value)})
    client = connect(server)
    try:
        first = client.submit("test", "append", 1)
        first.result(timeout=3)
        client.call("test", "append", 2)
        with pytest.raises(RemoteCallError, match="retired"):
            client.retry(first).result(timeout=3)
        assert calls == [1, 2]
    finally:
        client.close()
        server.close()


def test_handler_exception_is_cached_not_reexecuted(server):
    calls = []

    def fail():
        calls.append(1)
        raise ValueError("not a successful ownership commit")

    server.register_service("test", {"fail": fail})
    client = connect(server)
    try:
        future = client.submit("test", "fail")
        with pytest.raises(RemoteCallError, match="ValueError"):
            future.result(timeout=3)
        with pytest.raises(RemoteCallError, match="ValueError"):
            client.retry(future).result(timeout=3)
        assert calls == [1]
    finally:
        client.close()


def test_disconnect_fails_pending_and_cache_without_reconnect(server, monkeypatch):
    server.publish("host", "key", "retained")
    server.register_service("test", {"value": lambda: 1})
    monkeypatch.setattr(server, "_reply", lambda peer, response: None)
    client = connect(server, client_id="one-session")
    try:
        client.subscribe("host").ready.result(timeout=3)
        pending = client.submit("test", "value")
        server.close()
        with pytest.raises(ControlUnavailable):
            pending.result(timeout=3)
        with pytest.raises(ControlUnavailable):
            client.cache_get("host", "key")
        with pytest.raises(ControlUnavailable):
            client.retry(pending)
    finally:
        client.close()


def test_explicit_service_only_auth_and_identity_reuse(server):
    client = connect(server, client_id="session")
    with pytest.raises(RemoteCallError):
        client.call("__dict__", "anything")
    client.close()
    reused = ControlRPCClient(
        server.address, run_id="run", token="secret", client_id="session"
    )
    bad = ControlRPCClient(server.address, run_id="old-run", token="secret")
    try:
        with pytest.raises(ControlUnavailable):
            reused.wait_ready()
        with pytest.raises(ControlUnavailable):
            bad.wait_ready()
    finally:
        reused.close()
        bad.close()


def test_pending_capacity_rejects_without_sending_extra_operation(server, monkeypatch):
    count = []
    server.register_service("test", {"touch": lambda: count.append(1)})
    monkeypatch.setattr(server, "_reply", lambda peer, response: None)
    client = connect(server, max_pending=1)
    try:
        first = client.submit("test", "touch")
        with pytest.raises(ControlUnavailable, match="pending"):
            client.submit("test", "touch")
        assert client._operation == first.operation_id
    finally:
        client.close()


def test_local_queue_pressure_and_subscriber_pressure_fail_closed(server, monkeypatch):
    client = connect(server)
    try:

        def full(_):
            raise queue.Full

        monkeypatch.setattr(client._outbox, "put_nowait", full)
        with pytest.raises(ControlUnavailable, match="queue full"):
            client.submit("test", "method")
    finally:
        monkeypatch.undo()
        client.close()


def test_shared_env_factory_reuses_only_current_process(server, monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "%s:%s" % server.address)
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_RUN_ID", "run")
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_TOKEN", "secret")
    first = get_control_client()
    try:
        assert get_control_client() is first
    finally:
        first.close()


def test_watchers_have_independent_atomic_initial_views_and_delta_queues(server):
    server.publish("host", "s:0", {"stage": "ready"})
    server.register_service("records", {"publish": server.publish})
    client = connect(server)
    try:
        base = client.subscribe("host")
        base.ready.result(timeout=3)
        a, b = client.watch("host"), client.watch("host")
        a.ready.result(timeout=3)
        b.ready.result(timeout=3)
        assert a.initial_revision == b.initial_revision == 1
        assert a.initial_snapshot == b.initial_snapshot == {"s:0": {"stage": "ready"}}
        client.call("records", "publish", "host", "s:0", {"stage": "loading"})
        event_a, event_b = a.get_nowait(), b.get_nowait()
        assert event_a == event_b
        event_a["value"]["stage"] = "local mutation"
        assert event_b["value"]["stage"] == "loading"
        assert client.cache_get("host", "s:0")["stage"] == "loading"
        a.close()
        client.call("records", "publish", "host", "s:0", None)
        with pytest.raises(queue.Empty):
            a.get_nowait()
        assert b.get_nowait()["value"] is None
        assert client.cache_revision("host") == 3
    finally:
        client.close()


def test_watch_created_before_initial_snapshot_and_failure_notification(server):
    server.publish("host", "s:0", 1)
    client = connect(server)
    try:
        watcher = client.watch("host")
        watcher.ready.result(timeout=3)
        assert watcher.initial_snapshot == {"s:0": 1}
        watcher.changed.clear()
        client.close()
        assert watcher.changed.wait(3)
        with pytest.raises(ControlUnavailable):
            client.check_health()
    finally:
        client.close()


def test_slow_watcher_overflow_is_explicit_failure(server):
    client = connect(server, queue_capacity=2)
    try:
        watcher = client.watch("host")
        watcher.ready.result(timeout=3)
        # Wait for each cache delivery without consuming the ordered queue.
        for i in range(2):
            watcher.changed.clear()
            server.publish("host", str(i), i)
            assert watcher.changed.wait(3)
        server.publish("host", "overflow", 3)
        # Failure thread wakes the same event and all subsequent ownership reads
        # must reject stale state. No implicit record/lease eviction is done.
        client._threads[1].join(timeout=3)
        with pytest.raises(ControlUnavailable):
            client.cache_get("host", "0")
    finally:
        client.close()


def test_exact_operation_identity_reuse_with_changed_arguments_fails(server):
    server.register_service("test", {"value": lambda value: value})
    client = connect(server)
    try:
        first = client.submit("test", "value", 1)
        assert first.result(timeout=3) == 1
        first._request["args"] = [2]  # malformed retry injected deliberately
        with pytest.raises(ControlUnavailable):
            client.retry(first).result(timeout=3)
    finally:
        client.close()


def test_publication_failure_after_core_commit_poison_closes_stale_mirrors(server):
    owner = {"snapshot": "D"}
    server.publish("host", "snapshot", "D")

    def commit():
        owner["snapshot"] = "HOST"
        server.publish("host", "snapshot", object())  # cannot serialize

    server.register_service("host", {"commit": commit})
    writer, observer = connect(server), connect(server)
    try:
        writer.subscribe("host").ready.result(timeout=3)
        observer.subscribe("host").ready.result(timeout=3)
        with pytest.raises(ControlUnavailable):
            writer.submit("host", "commit").result(timeout=3)
        observer._threads[1].join(timeout=3)
        with pytest.raises(ControlUnavailable):
            observer.cache_get("host", "snapshot")
        assert owner["snapshot"] == "HOST"  # not a fake rollback/release
        assert server._closed
    finally:
        writer.close()
        observer.close()
