"""CPU-only bridge lifecycle tests; no claims about real RDMA performance."""
import threading
import queue
from types import SimpleNamespace

import pytest

from sglang.srt.disaggregation.agentic_remote_host import RemoteHostTransport
from sglang.srt.disaggregation.agentic_remote_host_engine import (
    RemoteHostEngineBridge, UnfencedRemoteRead, create_remote_host_bridge,
)
from sglang.srt.disaggregation.test_agentic_remote_host import FakeNixl


class Buffer:
    device = SimpleNamespace(index=0)
    shape = (1024, 1, 2)

    def __init__(self, address):
        self.address = address

    def data_ptr(self):
        return self.address


class Dtype:
    itemsize = 2

    def __str__(self):
        return "float16"


def bridge(tmp_path, rank=0, size=1, node="source", direction="p2d"):
    pool = SimpleNamespace(k_buffer=[Buffer(10000)], v_buffer=[Buffer(20000)],
        head_num=1, head_dim=2, store_dtype=Dtype(), layer_num=1)
    config = SimpleNamespace(tp_size=size, control_directory=str(tmp_path),
                             node_id=node, engine_id=node)
    agent = FakeNixl()
    agent.status = "DONE"
    transport = RemoteHostTransport(agent)
    result = RemoteHostEngineBridge(pool, 1, rank, size, direction, config,
                                    transport_factory=lambda: transport)
    return result, agent


def publish(source, snapshot_id="s"):
    snapshot = SimpleNamespace(kv_buffer=Buffer(1000), token_count=3, byte_size=24)
    source.export_snapshot(snapshot_id, snapshot)
    return {"token_count": 3, "byte_size": 24, "remote_host_node": "source"}


@pytest.mark.parametrize("size", [1, 2, 8])
def test_whole_tp_commit_required_before_source_release(tmp_path, size):
    sources = [bridge(tmp_path, rank, size) for rank in range(size)]
    targets = [bridge(tmp_path, rank, size, "target") for rank in range(size)]
    grants = [publish(source) for source, _ in sources]
    for rank, (target, _) in enumerate(targets):
        target.load("s", grants[rank], [10, 11, 12], attempt_id="epoch:1")
        assert not sources[rank][0].cleanup_source("s", {"state": "h2d_loading"})
        if rank < size - 1:
            assert not sources[rank][0].cleanup_source("s", {"state": "consumed"})
    for source, agent in sources:
        assert source.cleanup_source("s", {"state": "consumed"})
        assert len(agent.deregistered) == 1
        assert source.cleanup_source("s", {"state": "consumed"})


def test_failed_partial_tp_requires_peer_drain_proof(tmp_path):
    source0, _ = bridge(tmp_path, 0, 2)
    source1, _ = bridge(tmp_path, 1, 2)
    grant = publish(source0)
    publish(source1)
    target, _ = bridge(tmp_path, 0, 2, "target")
    with pytest.raises(RuntimeError, match="cancelled"):
        target.load("s", grant, [1, 2, 3], attempt_id="one", cancel_check=lambda: True)
    assert not source0.cleanup_source("s", {"state": "failed"})
    terminal = {"state": "failed", "loader_drained_ranks": [0, 1]}
    assert source0.cleanup_source("s", terminal)
    assert source1.cleanup_source("s", terminal)


def test_unfenced_read_never_releases_source(tmp_path):
    source, source_agent = bridge(tmp_path)
    grant = publish(source)
    target, agent = bridge(tmp_path, node="target")
    agent.status, agent.release_raises = "ERR", True
    with pytest.raises(UnfencedRemoteRead):
        target.load("s", grant, [1, 2, 3], attempt_id="one")
    assert not source.cleanup_source("s", {"state": "failed", "loader_drained_ranks": [0]})
    assert not source_agent.deregistered


def test_duplicate_read_not_reposted_and_peer_metadata_retired(tmp_path):
    source, _ = bridge(tmp_path)
    grant = publish(source)
    target, agent = bridge(tmp_path, node="target")
    receipt = target.load("s", grant, [1, 2, 3], attempt_id="one")
    assert target.load("s", grant, [1, 2, 3], attempt_id="one") == receipt
    assert len(agent.reads) == 1
    assert agent.removed_peers == ["source-agent"]
    with pytest.raises(RuntimeError, match="destination"):
        target.load("s", grant, [4, 5, 6], attempt_id="one")


def test_next_epoch_after_cancel_can_use_another_engine(tmp_path):
    source, _ = bridge(tmp_path)
    grant = publish(source)
    old, _ = bridge(tmp_path, node="old")
    with pytest.raises(RuntimeError, match="cancelled"):
        old.load("s", grant, [1, 2, 3], attempt_id="one", cancel_check=lambda: True)
    new, _ = bridge(tmp_path, node="new")
    new.load("s", grant, [5, 6, 7], attempt_id="two")
    assert source.cleanup_source("s", {"state": "consumed"})


def test_unread_eviction_and_final_release(tmp_path):
    source, _ = bridge(tmp_path)
    publish(source)
    assert source.cleanup_source("s", {"state": "evicting"})
    publish(source, "final")
    assert source.cleanup_source("final", {"state": "consumed", "reason": "agent_final"})


def test_cancel_unstarted_rank_allows_next_tp_epoch(tmp_path):
    sources = [bridge(tmp_path, rank, 2) for rank in range(2)]
    grants = [publish(source) for source, _ in sources]
    targets = [bridge(tmp_path, rank, 2, "target") for rank in range(2)]
    targets[0][0].load("s", grants[0], [1, 2, 3], attempt_id="old")
    receipt = targets[1][0].cancel_unstarted("s", attempt_id="old")
    assert receipt.outcome == "drained"
    assert targets[1][0].cancel_unstarted("s", attempt_id="old") == receipt
    for rank, (target, _) in enumerate(targets):
        target.load("s", grants[rank], [4, 5, 6], attempt_id="new")
    assert sources[0][0].cleanup_source("s", {"state": "consumed"})


def test_cancel_unstarted_cannot_cancel_executing_worker(tmp_path):
    source, _ = bridge(tmp_path)
    publish(source)
    target, _ = bridge(tmp_path, node="target")
    lock = target._read_locks.setdefault(("s", "one"), threading.Lock())
    with lock, pytest.raises(UnfencedRemoteRead):
        target.cancel_unstarted("s", attempt_id="one")


def test_failed_export_cannot_release_registered_arena(tmp_path):
    source, agent = bridge(tmp_path)
    agent.metadata_raises = agent.deregister_raises = True
    with pytest.raises(RuntimeError):
        publish(source)
    with pytest.raises(RuntimeError):
        source.cleanup_source("s", {"state": "failed"})
    assert "s" in source._failed_exports
    agent.deregister_raises = False
    assert source.cleanup_source("s", {"state": "failed"})


def test_unknown_partial_registration_stays_quarantined(tmp_path):
    source, agent = bridge(tmp_path)
    def partially_registered(*args, **kwargs):
        raise RuntimeError("native registration failed after partial work")
    agent.register_memory = partially_registered
    with pytest.raises(RuntimeError):
        publish(source)
    assert not source.cleanup_source("s", {"state": "failed"})
    assert "s" in source._failed_exports


def test_disabled_has_no_pool_or_environment_requirements(monkeypatch):
    monkeypatch.delenv("SGLANG_AGENTIC_MULTINODE_ENABLED", raising=False)
    assert create_remote_host_bridge(None, 1, 0, 1, "p2d") is None


def test_p2d_worker_remote_branch_never_opens_source_path(tmp_path, monkeypatch):
    from sglang.srt.disaggregation import p2d_host_staging as module
    source, _ = bridge(tmp_path)
    grant = publish(source)
    grant["arena_path"] = "/proc/SOURCE-PID/fd/NOT-LOCAL"
    target, agent = bridge(tmp_path, node="target")
    manager = module.AgenticPToDHostLoadManager.__new__(module.AgenticPToDHostLoadManager)
    manager._remote_bridge = target
    manager._stop = threading.Event()
    manager._work = queue.SimpleQueue()
    manager.chunk_tokens = 1024
    manager._dma_quarantine = []
    entry = {"state": "h2d_loading"}
    def complete(*args, **kwargs):
        entry["state"] = "consumed"
        return True
    manager.ledger = SimpleNamespace(get=lambda _: entry, complete_host_load_rank=complete)
    manager.tp_rank, manager.tp_size = 0, 1
    completions = []
    manager._finish_h2d_success = completions.append
    index_fences = []
    receiver = SimpleNamespace(snapshot_id="s", abort_pending=False,
                               _owner="source", _grant=grant,
                               _remote_indices_ready=SimpleNamespace(
                                   synchronize=lambda: index_fences.append("ready")))
    load = target.load
    def load_after_fence(*args, **kwargs):
        assert index_fences == ["ready"]
        return load(*args, **kwargs)
    target.load = load_after_fence
    manager._work.put((receiver, [1, 2, 3]))
    manager._work.put(None)
    def forbidden(*args, **kwargs):
        raise AssertionError("remote payload must not be mmap'ed on the target")
    monkeypatch.setattr(module, "SharedMHAHostSnapshot", forbidden)
    manager._worker(0, None, None, ())
    assert len(agent.reads) == 1
    assert len(completions) == 1 and completions[0]["snapshot"] is None
    assert source.cleanup_source("s", entry)
