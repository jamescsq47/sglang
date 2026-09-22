"""Background workset command delivery uses events, never scheduler ticks."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from sglang.srt.disaggregation.agentic_tp_events import (
    ControlUnavailable, EventKey, TPEventClient, TPEventServer,
)


@pytest.fixture
def clients():
    server = TPEventServer("workset-run", "test-token")
    peers = [TPEventClient(
        server.address, run_id="workset-run", token="test-token",
        group="P", rank=rank, size=2,
    ) for rank in range(2)]
    for peer in peers:
        peer.wait_ready()
    try:
        yield peers
    finally:
        for peer in peers:
            peer.close()
        server.close()


def test_worker_wakes_without_model_scheduler_and_does_not_ack(clients):
    leader, follower = clients
    key = EventKey("generation", "workset-1")
    with ThreadPoolExecutor(1) as worker:
        pending = worker.submit(follower.wait_commands, "workset", timeout=3)
        leader.publish_command("workset", key, {"sequence": 1})
        assert pending.result(timeout=4) == [(key, 1, {"sequence": 1})]
    assert not leader.command_complete("workset", key)
    assert follower.wait_commands("workset", timeout=0) == []


def test_other_namespace_does_not_supply_workset_commands(clients):
    leader, follower = clients
    key = EventKey("generation", "attempt")
    leader.publish_command("direct", key, {"action": "START"})
    leader.flush()
    assert follower.wait_commands("workset", timeout=0.02) == []
    assert follower.wait_commands("direct", timeout=3) == [
        (key, 1, {"action": "START"})
    ]


def test_disconnect_wakes_waiter_and_does_not_return_cached_success(clients):
    _, follower = clients
    with ThreadPoolExecutor(1) as worker:
        pending = worker.submit(follower.wait_commands, "workset", timeout=3)
        follower.close()
        with pytest.raises(ControlUnavailable):
            pending.result(timeout=4)


@pytest.mark.parametrize("namespace,limit", [("", 1), ("workset", 0)])
def test_invalid_wait_rejected(clients, namespace, limit):
    with pytest.raises(ValueError):
        clients[0].wait_commands(namespace, limit=limit, timeout=0)


def test_update_subscription_coalesces_without_historical_scan(clients):
    leader, follower = clients
    key = EventKey("generation", "attempt")
    leader.subscribe_updates("prepared")
    leader.report("prepared", key, 1)
    leader.report("prepared", key, 2)
    leader.flush()
    # Wait for the final group value; one dirty key represents all updates.
    follower.report("prepared", key, 2)
    follower.flush()
    with leader._condition:
        assert leader._condition.wait_for(
            lambda: leader.group_status("prepared", key) == 2, timeout=3
        )
    assert leader.drain_update_keys("prepared") == [key]
    assert leader.drain_update_keys("prepared") == []
    assert leader._update_inbox_size == 0
    assert "unwatched" not in leader._update_inboxes


def test_update_inbox_overflow_is_not_silently_dropped(clients):
    _, follower = clients
    follower.subscribe_updates("prepared")
    follower._update_inbox_limit = 1
    with follower._condition:
        follower._enqueue_update_event({"identity": ["prepared", "g1", "a"]})
        with pytest.raises(ControlUnavailable, match="inbox full"):
            follower._enqueue_update_event({"identity": ["prepared", "g2", "a"]})
    assert follower.drain_update_keys("prepared") == [EventKey("g1", "a")]
