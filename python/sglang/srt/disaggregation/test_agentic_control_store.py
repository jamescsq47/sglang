"""Metadata adapters: no control files, exact exclusion and TP admission."""

from concurrent.futures import Future, ThreadPoolExecutor
import threading

import pytest

from sglang.srt.disaggregation.agentic_control_rpc import ControlRPCClient, ControlRPCServer
from sglang.srt.disaggregation.agentic_control_store import ControlKV, MemoryControlStore
from sglang.srt.disaggregation.agentic_early_claim import AgenticEarlyClaimStore
from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration


@pytest.fixture
def broker():
    server = ControlRPCServer("test", "secret")
    memory = MemoryControlStore(server.publish)
    server.register_service("records", memory.methods())
    client = ControlRPCClient(server.address, run_id="test", token="secret")
    client.wait_ready()
    try:
        yield server, client, memory
    finally:
        client.close()
        server.close()


def test_exact_owner_guard_and_parallel_claims(broker):
    _, client, _ = broker
    records = ControlKV("test", client=client)
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda i: records.call("claim", "generation", str(i)), range(8)))
    assert sum(result[1] for result in results) == 1
    owner = records.get("generation")
    assert records.call("claim", "generation", owner) == [True, False]
    assert records.call("remove", "generation", "stale-owner") == -1
    assert records.get("generation") == owner
    assert records.call("remove", "generation", owner) == 0


@pytest.mark.parametrize("tp", [1, 2, 8])
def test_ready_latch_waits_for_all_admitted_shards(broker, tp):
    _, client, _ = broker
    records = ControlKV("ready", client=client)
    records.call("upsert", "room.ready", {"num_kv_tokens": 8192})
    for rank in range(tp):
        assert records.call("admit_ready", "room.ready", "D0", rank, tp) == (rank == tp - 1)
        assert (records.get("room.ready") is None) == (rank == tp - 1)
    assert records.call("admit_ready", "room.ready", "D0", 0, tp)
    with pytest.raises(RuntimeError, match="republish"):
        records.call("upsert", "room.ready", {})


def test_early_claim_payloads_and_independent_event_watchers_no_files(broker, monkeypatch, tmp_path):
    _, client, _ = broker
    import sglang.srt.disaggregation.agentic_control_store as module
    records = ControlKV("early", client=client)
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "enabled")
    monkeypatch.setattr(module, "control_kv", lambda *args: records)
    root = tmp_path / "must-not-exist"
    store = AgenticEarlyClaimStore(str(root))
    req = RequestGeneration("rid:with:colons", 3)
    with store.watch_arrivals(max_age_seconds=60) as a, store.watch_arrivals(max_age_seconds=60) as b:
        payload = store.publish_arrival(req, prompt_token_count=9000, target_prefill_domain=1)
        assert a.poll(1) == [(req, payload)]
        assert b.poll(1) == [(req, payload)]
        assert a.poll(0) == []
        assert store.claim_generation_producer(req, "D0")
        assert not store.claim_generation_producer(req, "D1")
        assert store.wait_generation_producer(req, "D0", timeout_seconds=0)
        store.publish_direct_abort(req, claim_id="attempt-2")
        assert store.read_direct_abort(req, claim_id="attempt-1", not_before=0, max_age_seconds=60) is None
        assert store.read_direct_abort(req, claim_id="attempt-2", not_before=0, max_age_seconds=60)
        store.publish_route(req, route="host_ready", prefill_domain=1)
        assert store.read_route(req)["route"] == "host_ready"
        store.remove_arrival(req)
        assert a.poll(0) == []
    assert not root.exists()


def test_nonowning_notifications_do_not_wait_for_handler(broker):
    _, client, _ = broker
    records = ControlKV("notify", client=client)
    future = records.notify("upsert", "ready", {"ready": True})
    future.result(timeout=3)
    assert records.get("ready") == {"ready": True}


def test_failed_notification_remains_failed_on_later_checks(broker):
    _, client, _ = broker
    records = ControlKV("notify-failed", client=client)
    failed = Future()
    failed.set_exception(RuntimeError("publication rejected"))
    records._pending.append(failed)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="publication rejected"):
            records.check()
    assert records._pending[0] is failed


def test_failed_ready_retirement_does_not_create_false_terminal():
    published = []
    fail_delete = [True]
    def publish(namespace, key, value):
        if value is None and fail_delete[0]:
            raise RuntimeError("injected publication failure")
        published.append(value)
    state = MemoryControlStore(publish)
    state.upsert("ready", "room", {"sequence": 1})
    with pytest.raises(RuntimeError, match="injected"):
        state.admit_ready("ready", "room", "D", 0, 1)
    assert ("ready", "room") not in state._retired_ready
    assert "room" in state._data["ready"]
    fail_delete[0] = False
    assert state.admit_ready("ready", "room", "D", 0, 1)
    assert published[-1] is None


def test_cached_namespace_does_not_wait_for_other_namespace_subscription(monkeypatch):
    import sglang.srt.disaggregation.agentic_control_store as module
    entered, release = threading.Event(), threading.Event()

    def construct(namespace):
        if namespace == module.record_namespace("test-delayed", "default"):
            entered.set()
            assert release.wait(3)
        return object()

    monkeypatch.setattr(module, "_stores", {})
    monkeypatch.setattr(module, "ControlKV", construct)
    cached = module.control_kv("test-cached")
    with ThreadPoolExecutor(2) as pool:
        delayed = pool.submit(module.control_kv, "test-delayed")
        try:
            assert entered.wait(2)
            immediate = pool.submit(module.control_kv, "test-cached")
            assert immediate.result(timeout=1) is cached
        finally:
            release.set()
        assert delayed.result(timeout=2) is not None
