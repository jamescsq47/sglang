"""Integrated broker cutover test; physical DMA is a fenced FakeNixl only.

Uses the actual server bootstrap and adapters together. No GPU/performance
claim: these tests check cross-service ownership and absence of control files.
"""

import os
from pathlib import Path

import pytest

from sglang.srt.disaggregation.agentic_control_rpc import ControlRPCClient, RemoteCallError
from sglang.srt.disaggregation.agentic_control_server import create_server
from sglang.srt.disaggregation.agentic_early_claim import AgenticEarlyClaimStore
from sglang.srt.disaggregation.agentic_host_rpc import RemoteHostStagingLedger
from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration
from sglang.srt.disaggregation.test_agentic_remote_host_engine import bridge, publish


@pytest.fixture
def broker(monkeypatch):
    import sglang.srt.disaggregation.agentic_control_rpc as rpc
    server = create_server("integration-run", "secret", ("127.0.0.1", 0))
    client = ControlRPCClient(server.address, run_id="integration-run", token="secret")
    client.wait_ready()
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "integration-enabled")
    monkeypatch.setattr(rpc, "get_control_client", lambda: client)
    try:
        yield client
    finally:
        client.close()
        server.close()


def _no_files(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("integrated Host/control path touched filesystem")
    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr("fcntl.flock", forbidden)
    for name in ("open", "stat", "scandir", "mkdir", "makedirs", "unlink", "replace"):
        monkeypatch.setattr(os, name, forbidden)


def _ready(ledger, sid, size, grants):
    owner = "source-host"
    for rank in range(size):
        ledger.offer({"snapshot_id": sid, "tp_rank": rank, "tp_size": size,
                      "token_count": 3, "byte_size": 24, "source_host_node": "source"})
    for rank in range(size):
        assert ledger.claim_rank(sid, owner, tp_rank=rank, tp_size=size)
        assert ledger.publish_rank_grant(sid, owner, {
            **grants[rank], "kind": "shared_host_extent", "tp_rank": rank,
        }, tp_rank=rank, tp_size=size)
    for rank in range(size):
        assert ledger.complete_p2d_host_write_rank(sid, owner, tp_rank=rank, tp_size=size)
    assert ledger.get(sid)["state"] == "host_ready"
    return owner


@pytest.mark.parametrize("size", [2, 8])
@pytest.mark.parametrize("cancel", [False, True])
def test_real_broker_records_host_remote_dma_ownership(broker, monkeypatch, size, cancel):
    # This path is deliberately not a usable directory. All control and
    # lifecycle access after startup must go through the real broker.
    scope = Path("/must-not-access-nfs") / f"tp{size}-cancel{cancel}"
    sources = [bridge(scope, rank, size) for rank in range(size)]
    targets = [bridge(scope, rank, size, "target") for rank in range(size)]
    ledger = RemoteHostStagingLedger(broker, "p2d")
    request = RequestGeneration("integrated-host", int(cancel))
    sid = request.snapshot_id
    with monkeypatch.context() as trapped:
        _no_files(trapped)
        early = AgenticEarlyClaimStore(str(scope))
        with early.watch_arrivals(max_age_seconds=60) as arrivals:
            marker = early.publish_arrival(request, prompt_token_count=7, target_prefill_domain=0)
            assert arrivals.poll(1) == [(request, marker)]
            assert early.claim_generation_producer(request, "source")
            assert not early.claim_generation_producer(request, "other-source")
            grants = [publish(source, sid) for source, _ in sources]
            owner = _ready(ledger, sid, size, grants)
            early.publish_route(request, route="host_ready", prefill_domain=0)
            commands = [dict(tp_rank=rank, tp_size=size, decode_domain=0,
                             attempt_id=f"receiver-{rank}") for rank in range(size)]
            for command in commands:
                assert ledger.begin_host_load_rank(sid, owner, **command)
            # A forged/stale terminal hint is not an authoritative fence.
            assert not sources[0][0].cleanup_source(sid, {"state": "consumed"})
            if cancel:
                targets[0][0].load(sid, grants[0], [1, 2, 3], attempt_id="physical-1")
                assert ledger.request_host_load_failure(sid, owner, reason="cancel", **commands[0])
                assert not ledger.mark_host_load_rank_drained(
                    sid, owner, **dict(commands[0], attempt_id="stale-receiver"),
                )
                for rank, (target, _) in enumerate(targets):
                    if rank:
                        target.cancel_unstarted(sid, attempt_id="physical-1")
                    assert ledger.mark_host_load_rank_drained(sid, owner, **commands[rank])
                    if rank + 1 < size:
                        assert not sources[rank][0].cleanup_source(sid, {
                            "state": "failed", "loader_drained_ranks": list(range(size)),
                        })
                assert ledger.get(sid)["state"] == "failed"
            else:
                for rank, (target, _) in enumerate(targets):
                    assert target.load(sid, grants[rank], [1, 2, 3], attempt_id="physical-1").outcome == "loaded"
                    assert not ledger.complete_host_load_rank(
                        sid, owner, **dict(commands[rank], attempt_id="stale-receiver"),
                    )
                    assert ledger.complete_host_load_rank(sid, owner, **commands[rank])
                    assert ledger.complete_host_load_rank(sid, owner, **commands[rank])
                    if rank + 1 < size:
                        assert not sources[rank][0].cleanup_source(sid, {"state": "consumed"})
                assert ledger.get(sid)["state"] == "consumed"
            terminal = ledger.get(sid)
            for rank, (source, agent) in enumerate(sources):
                assert source.cleanup_source(sid, terminal)
                assert source.cleanup_source(sid, terminal)
                assert len(agent.deregistered) == 1
                assert ledger.complete_source_host_release_rank(sid, owner, tp_rank=rank, tp_size=size)
            with pytest.raises(RemoteCallError, match="retired"):
                targets[0][0]._claim_attempt(sid, "late-physical-attempt")
            assert early.read_route(request)["route"] == "host_ready"
            early.remove_arrival(request)
            assert arrivals.poll(0) == []


@pytest.mark.parametrize("size", [2, 8])
def test_real_broker_d2p_exact_lease_handoff_and_prune(broker, monkeypatch, size):
    scope = Path("/must-not-access-nfs") / f"d2p-tp{size}"
    sources = [bridge(scope, rank, size, direction="d2p") for rank in range(size)]
    targets = [bridge(scope, rank, size, "target", "d2p") for rank in range(size)]
    ledger = RemoteHostStagingLedger(broker, "d2p")
    sid = "integrated-recovery:0"
    with monkeypatch.context() as trapped:
        _no_files(trapped)
        grants = [publish(source, sid) for source, _ in sources]
        owner = _ready(ledger, sid, size, grants)
        commands = []
        for rank in range(size):
            identity = dict(tp_rank=rank, tp_size=size, claim_id="recovery")
            assert ledger.claim_d2p_recovery_rank(sid, owner, recovery_domain=0, **identity)
            assert ledger.attach_d2p_recovery_lease_rank(sid, owner, lease_id=100 + rank, **identity)
            command = dict(identity, lease_id=100 + rank)
            assert ledger.mark_d2p_recovery_phase_rank(sid, owner, phase="io_inflight", **command)
            commands.append(dict(command, remote_read_epoch=1))
        for rank, (target, _) in enumerate(targets):
            target.load(sid, grants[rank], [4, 5, 6], attempt_id="recovery:epoch:1")
            before = ledger.get(sid)
            assert not ledger.complete_d2p_host_load_rank(sid, owner, **dict(commands[rank], remote_read_epoch=0))
            assert not ledger.complete_d2p_host_load_rank(sid, owner, **dict(commands[rank], lease_id=999))
            assert ledger.get(sid) == before
            assert ledger.complete_d2p_host_load_rank(sid, owner, **commands[rank])
        assert ledger.get(sid)["state"] == "hbm_ready"
        assert not sources[0][0].cleanup_source(sid, {"state": "consumed"})
        for command in commands:
            assert ledger.complete_host_bind_rank(sid, owner, **command)
        for rank, (source, agent) in enumerate(sources):
            assert source.cleanup_source(sid, ledger.get(sid))
            assert len(agent.deregistered) == 1
            assert ledger.complete_source_host_release_rank(sid, owner, tp_rank=rank, tp_size=size)
        ledger.prune(0, 0)
        assert ledger.get(sid) is not None  # DMA/bind does not imply scheduler handoff.
        for command in commands:
            assert ledger.mark_d2p_recovery_phase_rank(sid, owner, phase="handed", **command)
        ledger.prune(0, 0)
        assert ledger.get(sid) is None
        with pytest.raises(RemoteCallError, match="retired"):
            ledger.offer({"snapshot_id": sid, "tp_rank": 0, "tp_size": size})
