"""No-GPU tests of the typed remote Host bridge control backend."""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import pytest

from sglang.srt.disaggregation.agentic_remote_host import ReadReceipt
from sglang.srt.disaggregation.agentic_remote_host_control import (
    RemoteHostControlClient,
    RemoteHostControlState,
)
from sglang.srt.disaggregation.agentic_remote_host_engine import UnfencedRemoteRead
from sglang.srt.disaggregation.test_agentic_remote_host_engine import bridge, publish


class LocalRPC:
    def __init__(self, state):
        self.state = state
        self.lose_ack = None
        self.lifecycle = {}
        self.mirror_cleanup_hint = state._ledger_lookup is None
        if self.mirror_cleanup_hint:
            # Legacy bridge fixtures do not instantiate a Host ledger; make
            # their explicit completion call the simulated authority.
            state._ledger_lookup = lambda direction, sid: self.lifecycle.get(
                (direction, sid)
            )

    def call(self, service, method, operation, payload):
        assert (service, method) == ("remote_host", "call")
        if operation == "prepare_cleanup" and self.mirror_cleanup_hint:
            self.lifecycle[(payload["direction"], payload["snapshot_id"])] = payload[
                "terminal_entry"
            ]
        result = self.state.handle(operation, payload)
        if operation == self.lose_ack:
            self.lose_ack = None
            raise ConnectionError("injected lost acknowledgment")
        return result


def controlled(tmp_path, rpc, rank=0, size=1, node="source", direction="p2d"):
    value, agent = bridge(tmp_path, rank, size, node, direction)
    value._control = RemoteHostControlClient(rpc, direction)
    return value, agent


@pytest.mark.parametrize("direction", ["p2d", "d2p"])
@pytest.mark.parametrize("size", [1, 2, 8])
def test_file_free_complete_group_and_release(tmp_path, monkeypatch, size, direction):
    from sglang.srt.disaggregation import agentic_remote_host_engine as engine

    state = RemoteHostControlState()
    rpc = LocalRPC(state)
    sources = [
        controlled(tmp_path, rpc, r, size, direction=direction) for r in range(size)
    ]
    targets = [
        controlled(tmp_path, rpc, r, size, "target", direction) for r in range(size)
    ]

    def no_files(*args, **kwargs):
        raise AssertionError("remote Host control touched files")

    monkeypatch.setattr(engine, "_read_json", no_files)
    monkeypatch.setattr(engine, "_write_json", no_files)
    with monkeypatch.context() as patch:
        patch.setattr("builtins.open", no_files)
        patch.setattr("pathlib.Path.open", no_files)
        patch.setattr("fcntl.flock", no_files)
        grants = [publish(source) for source, _ in sources]
        for rank, (target, _) in enumerate(targets):
            receipt = target.load("s", grants[rank], [1, 2, 3], attempt_id="one")
            assert receipt.outcome == "loaded"
            if rank + 1 < size:
                assert not sources[rank][0].cleanup_source("s", {"state": "consumed"})
        for source, agent in sources:
            assert source.cleanup_source("s", {"state": "consumed"})
            assert source.cleanup_source("s", {"state": "consumed"})
            assert len(agent.deregistered) == 1
    assert state._entries[(direction, "s")]["released"] == list(range(size))


def test_same_attempt_not_reposted_or_changed(tmp_path):
    rpc = LocalRPC(RemoteHostControlState())
    source, _ = controlled(tmp_path, rpc)
    target, agent = controlled(tmp_path, rpc, node="target")
    grant = publish(source)
    first = target.load("s", grant, [1, 2, 3], attempt_id="one")
    assert target.load("s", grant, [1, 2, 3], attempt_id="one") == first
    assert len(agent.reads) == 1
    with pytest.raises(UnfencedRemoteRead):
        target.load("s", grant, [4, 5, 6], attempt_id="one")
    reincarnated, _ = controlled(tmp_path, rpc, node="target")
    with pytest.raises(UnfencedRemoteRead):
        reincarnated.load("s", grant, [1, 2, 3], attempt_id="one")


def test_partial_failure_requires_real_peer_fences(tmp_path):
    rpc = LocalRPC(RemoteHostControlState())
    source0, agent0 = controlled(tmp_path, rpc, 0, 2)
    source1, _ = controlled(tmp_path, rpc, 1, 2)
    target, _ = controlled(tmp_path, rpc, 0, 2, "target")
    grant = publish(source0)
    publish(source1)
    with pytest.raises(RuntimeError, match="cancelled"):
        target.load("s", grant, [1, 2, 3], attempt_id="one", cancel_check=lambda: True)
    assert not source0.cleanup_source("s", {"state": "failed"})
    assert not agent0.deregistered
    terminal = {"state": "failed", "loader_drained_ranks": [0, 1]}
    assert source0.cleanup_source("s", terminal)
    assert source1.cleanup_source("s", terminal)


def test_unfenced_dma_never_releases_host(tmp_path):
    rpc = LocalRPC(RemoteHostControlState())
    source, source_agent = controlled(tmp_path, rpc)
    target, target_agent = controlled(tmp_path, rpc, node="target")
    grant = publish(source)
    target_agent.status, target_agent.release_raises = "ERR", True
    with pytest.raises(UnfencedRemoteRead):
        target.load("s", grant, [1, 2, 3], attempt_id="one")
    assert not source.cleanup_source(
        "s", {"state": "failed", "loader_drained_ranks": [0]}
    )
    assert not source_agent.deregistered


def test_retry_epoch_and_stale_attempt_are_fenced(tmp_path):
    rpc = LocalRPC(RemoteHostControlState())
    sources = [controlled(tmp_path, rpc, r, 2) for r in range(2)]
    targets = [controlled(tmp_path, rpc, r, 2, "target") for r in range(2)]
    grants = [publish(source) for source, _ in sources]
    targets[0][0].load("s", grants[0], [1, 2, 3], attempt_id="old")
    targets[1][0].cancel_unstarted("s", attempt_id="old")
    for rank, (target, _) in enumerate(targets):
        target.load("s", grants[rank], [4, 5, 6], attempt_id="new")
    with pytest.raises(ValueError, match="stale"):
        targets[0][0]._claim_attempt("s", "old")
    assert sources[0][0].cleanup_source("s", {"state": "consumed"})


def test_cleanup_seals_against_new_reader_before_deregister(tmp_path):
    rpc = LocalRPC(RemoteHostControlState())
    source, agent = controlled(tmp_path, rpc)
    target, _ = controlled(tmp_path, rpc, node="target")
    publish(source)
    source_shard = source._exports["s"].shard
    rpc.lifecycle[("p2d", "s")] = {"state": "evicting"}
    plan = rpc.state.handle(
        "prepare_cleanup",
        {
            "direction": "p2d",
            "snapshot_id": "s",
            "rank": 0,
            "export_id": source_shard.export_id,
            "terminal_entry": {"state": "evicting"},
        },
    )
    assert plan["attempt"] is None
    assert not agent.deregistered  # worker has not executed the physical plan
    with pytest.raises(ValueError, match="retired"):
        target._claim_attempt("s", "late")
    assert source.cleanup_source("s", {"state": "evicting"})


def test_cleanup_ack_loss_is_idempotent(tmp_path):
    rpc = LocalRPC(RemoteHostControlState())
    source, agent = controlled(tmp_path, rpc)
    publish(source)
    rpc.lose_ack = "complete_cleanup"
    with pytest.raises(ConnectionError):
        source.cleanup_source("s", {"state": "evicting"})
    assert source._exports["s"].closed
    assert source.cleanup_source("s", {"state": "evicting"})
    assert len(agent.deregistered) == 1


def test_concurrent_source_cleanup_deregisters_once(tmp_path):
    rpc = LocalRPC(RemoteHostControlState())
    source, agent = controlled(tmp_path, rpc)
    publish(source)
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(
            pool.map(
                lambda _: source.cleanup_source("s", {"state": "evicting"}), range(8)
            )
        )
    assert len(agent.deregistered) == 1


def test_dropped_destination_ack_does_not_fake_a_drain(tmp_path):
    rpc = LocalRPC(RemoteHostControlState())
    source, source_agent = controlled(tmp_path, rpc)
    target, agent = controlled(tmp_path, rpc, node="target")
    grant = publish(source)
    rpc.lose_ack = "reserve_destination"
    with pytest.raises(UnfencedRemoteRead):
        target.load("s", grant, [1, 2, 3], attempt_id="one")
    assert not agent.reads
    assert not source.cleanup_source(
        "s", {"state": "failed", "loader_drained_ranks": [0]}
    )
    assert not source_agent.deregistered
    target.load("s", grant, [1, 2, 3], attempt_id="one")
    assert len(agent.reads) == 1


def test_rpc_roundtrip_uses_no_files_and_keeps_receipt_after_cleanup(tmp_path):
    from sglang.srt.disaggregation.agentic_control_rpc import (
        ControlRPCServer,
        ControlRPCClient,
    )

    server = ControlRPCServer("run", "secret")
    state = RemoteHostControlState(
        ledger_lookup=lambda direction, sid: {"state": "consumed"}
    )
    server.register_service("remote_host", {"call": state.handle})
    client = ControlRPCClient(server.address, run_id="run", token="secret")
    try:
        source, _ = controlled(tmp_path, client)
        target, agent = controlled(tmp_path, client, node="target")
        grant = publish(source)
        receipt = target.load("s", grant, [1, 2, 3], attempt_id="one")
        assert source.cleanup_source("s", {"state": "consumed"})
        assert target.load("s", grant, [1, 2, 3], attempt_id="one") == receipt
        assert target.cancel_unstarted("s", attempt_id="one") == receipt
        assert len(agent.reads) == 1
    finally:
        client.close()
        server.close()


def test_cancel_and_destination_reservation_cannot_both_win(tmp_path):
    rpc = LocalRPC(RemoteHostControlState())
    source, _ = controlled(tmp_path, rpc)
    target, _ = controlled(tmp_path, rpc, node="target")
    publish(source)
    target._claim_attempt("s", "one")
    shard = source._exports["s"].shard.to_dict()
    payload = {"rank": 0, "attempt_id": "one", "engine_id": "target"}
    signature = {
        "engine": "target",
        "incarnation": "worker",
        "shard": shard,
        "spans_digest": "0" * 64,
        "span_count": 1,
    }

    def execute(method):
        try:
            return method, target._control.call(
                method,
                "s",
                **payload,
                **({"signature": signature} if method == "reserve_destination" else {})
            )
        except ValueError:
            return method, "blocked"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = dict(pool.map(execute, ["reserve_destination", "cancel_unstarted"]))
    # reserve winning means cancel is blocked; cancel winning means reserve
    # observes a drained receipt and may not submit DMA.
    if results["reserve_destination"] is None:
        assert results["cancel_unstarted"] == "blocked"
    else:
        assert results["reserve_destination"]["outcome"] == "drained"


def test_receipt_identity_and_detached_reads(tmp_path):
    rpc = LocalRPC(RemoteHostControlState())
    source, _ = controlled(tmp_path, rpc)
    target, _ = controlled(tmp_path, rpc, node="target")
    publish(source)
    target._claim_attempt("s", "one")
    shard = source._exports["s"].shard
    bad = asdict(
        ReadReceipt("s", "wrong-export", 0, 1, shard.layout, 3, "one", "drained")
    )
    with pytest.raises(ValueError, match="mismatch"):
        target._control.call("receipt", "s", engine_id="target", receipt=bad)
    descriptor = source._source_descriptor("s")
    descriptor["shard"]["export_id"] = "changed"
    assert source._source_descriptor("s")["shard"]["export_id"] == shard.export_id


def test_generations_bounded_and_directions_independent(tmp_path):
    rpc = LocalRPC(RemoteHostControlState(max_generations=1))
    source, _ = controlled(tmp_path, rpc)
    publish(source)
    assert source.cleanup_source("s", {"state": "evicting"})
    with pytest.raises(RuntimeError, match="capacity"):
        publish(source, "other")
    value = source._control.call("descriptor", "s", rank=0)
    assert value is not None  # immutable retirement tombstone retained


def test_stale_terminal_copy_cannot_retire_a_new_recovery_attempt(tmp_path):
    ledger = {"state": "loading", "remote_read_epoch": 2}
    rpc = LocalRPC(RemoteHostControlState(ledger_lookup=lambda direction, sid: ledger))
    source, agent = controlled(tmp_path, rpc)
    target, _ = controlled(tmp_path, rpc, node="target")
    publish(source)
    target.cancel_unstarted("s", attempt_id="claim:epoch:1")
    target._claim_attempt("s", "claim:epoch:2")
    stale = {"state": "failed", "remote_read_epoch": 1, "loader_drained_ranks": [0]}
    assert not source.cleanup_source("s", stale)
    assert not agent.deregistered
    assert rpc.state._entries[("p2d", "s")]["sealed"] is None
    # Even with a genuine current failure, no stale rank-drained proof is used.
    ledger["state"] = "failed"
    assert not source.cleanup_source("s", stale)
    target.cancel_unstarted("s", attempt_id="claim:epoch:2")
    assert source.cleanup_source("s", stale)
    assert len(agent.deregistered) == 1


def test_cleanup_requires_authority_and_sealed_retry_cannot_start_new_dma(tmp_path):
    state = RemoteHostControlState()
    rpc = LocalRPC(state)
    source, _ = controlled(tmp_path, rpc)
    target, _ = controlled(tmp_path, rpc, node="target")
    publish(source)
    state._ledger_lookup = None
    with pytest.raises(RuntimeError, match="authoritative ledger"):
        source.cleanup_source("s", {"state": "evicting"})
    state._ledger_lookup = lambda direction, sid: {"state": "failed"}
    target.cancel_unstarted("s", attempt_id="one")
    assert source.cleanup_source("s", {"state": "failed"})
    assert target._claim_attempt("s", "one") is None  # harmless exact retry
    with pytest.raises(ValueError, match="retired"):
        target._claim_attempt("s", "two")
    result = target.cancel_unstarted("s", attempt_id="one")
    assert result.outcome == "drained"
