"""CPU lifecycle tests for real D2P engine hooks; no GPU/RDMA claims."""

import tempfile
from concurrent.futures import Future
from unittest.mock import Mock
from types import SimpleNamespace

import pytest

from sglang.srt.disaggregation.agentic_host_staging import (
    AgenticDHostStagingClient, AgenticPHostStagingManager,
    HostCopyWorkerPool, HostStageState, SharedHostStagingLedger,
)
from sglang.srt.disaggregation.agentic_multinode_d2p import (
    RemoteHostDescriptor, RemoteOnlyArena, SourceLocalD2PArena,
)
from sglang.srt.disaggregation.agentic_remote_host_engine import UnfencedRemoteRead


class FakeArena:
    capacity_bytes = 1024
    used_bytes = 0

    def can_reserve(self, size, watermark):
        return self.used_bytes + size <= self.capacity_bytes * watermark

    def create(self, sid, tokens, pool, size):
        snapshot = SimpleNamespace(path="/proc/source-only/fd/3", file_offset=0,
            token_count=tokens, byte_size=size)
        snapshot.materialize = lambda: snapshot
        self.used_bytes += size
        return snapshot

    def release(self, snapshot):
        self.used_bytes -= snapshot.byte_size
        return True

    def usage(self):
        return self.used_bytes / self.capacity_bytes


class FakeBridge:
    def __init__(self):
        self.exported = []
        self.cleanup_allowed = False

    def export_snapshot(self, sid, snapshot):
        self.exported.append(sid)

    def cleanup_source(self, sid, entry):
        return self.cleanup_allowed


@pytest.fixture
def ledger():
    with tempfile.TemporaryDirectory(prefix="test-d2p-", dir="/dev/shm") as directory:
        yield SharedHostStagingLedger(directory + "/ledger.json")


def offer(ledger, rank=0, size=1, sid="request:0", **extra):
    return ledger.offer(dict(snapshot_id=sid, request_id="request", generation=0,
        token_count=16, token_digest="digest", byte_size=128, tp_rank=rank,
        tp_size=size, d_pid=100 + rank, source_numa_node=rank % 2,
        arena_numa_node=0, arena_domain=0, source_host_node="decode-node",
        source_host_engine="decode0", **extra))


def source(ledger, rank=0, size=1):
    client = SimpleNamespace(ledger=ledger, tp_rank=rank, tp_size=size,
        device_pool=object(), page_size=1, source_numa_node=rank % 2)
    config = SimpleNamespace(node_id="decode-node", engine_id="decode0", run_id="run")
    return SourceLocalD2PArena(client=client, config=config, arena=FakeArena(),
        bridge=FakeBridge(), start_thread=False)


def ready_group(ledger, size=1):
    for rank in range(size):
        offer(ledger, rank, size)
    sources = [source(ledger, rank, size) for rank in range(size)]
    for manager in sources:
        manager.ensure_grant(ledger.get("request:0"))
    assert ledger.get("request:0")["state"] == "host_writing"
    for rank, manager in enumerate(sources):
        manager.export_after_d2h("request:0")
        assert ledger.complete_host_write("request:0", 100 + rank,
            tp_rank=rank, tp_size=size)
        if rank + 1 < size:
            assert ledger.get("request:0")["state"] == "host_writing"
    assert ledger.get("request:0")["state"] == "host_ready"
    return sources


def test_tp8_source_local_grants_and_allrank_durability(ledger):
    sources = ready_group(ledger, 8)
    entry = ledger.get("request:0")
    assert entry["p_owner"] == "d-host:decode0"
    assert len(entry["grants"]) == 8
    assert all(grant["remote_host_node"] == "decode-node" for grant in entry["grants"])
    assert all(manager.arena.used_bytes == 128 for manager in sources)
    dummy_p = SimpleNamespace(arena_domain=0, arena_numa_node=0, tp_rank=0)
    assert not AgenticPHostStagingManager._offer_targets_this_arena(dummy_p, entry)


def test_remote_source_cannot_be_retargeted_to_other_engine(ledger):
    current = offer(ledger)
    current["source_host_engine"] = "other-d"
    with pytest.raises(RuntimeError, match="identity"):
        source(ledger).ensure_grant(current)


def test_source_cannot_release_only_hbm_ready(ledger):
    manager = ready_group(ledger)[0]
    manager.bridge.cleanup_allowed = True
    entry = dict(ledger.get("request:0"), state="hbm_ready")
    assert not manager._cleanup_one("request:0", entry)
    assert manager.arena.used_bytes == 128


def test_eviction_fenced_before_extent_and_ledger_released(ledger):
    manager = ready_group(ledger)[0]
    assert ledger.begin_host_eviction("request:0", manager.owner, tp_size=1, reason="pressure")
    current = ledger.get("request:0")
    assert not manager._cleanup_one("request:0", current)
    assert manager.arena.used_bytes == 128
    manager.bridge.cleanup_allowed = True
    assert manager._cleanup_one("request:0", current)
    assert manager.arena.used_bytes == 0
    assert ledger.get("request:0")["state"] == "recompute_required"
    assert ledger.get("request:0")["source_host_released_ranks"] == [0]


def test_unexported_failed_label_does_not_prove_writer_fence(ledger):
    current = offer(ledger)
    manager = source(ledger)
    manager.ensure_grant(current)
    entry = dict(ledger.get("request:0"), state="failed")
    assert not manager._cleanup_one("request:0", entry)
    assert manager.arena.used_bytes == 128


def test_source_abort_progresses_after_all_writer_drain(ledger):
    manager = source(ledger)
    manager.ensure_grant(offer(ledger))
    assert ledger.transition("request:0", HostStageState.ABORTING, owner=manager.owner)
    manager.progress()
    assert manager.arena.used_bytes == 128
    assert ledger.mark_writer_rank_drained("request:0", 100, tp_rank=0, tp_size=1)
    manager.progress()
    assert manager.arena.used_bytes == 0
    assert ledger.get("request:0")["state"] == "failed"


def test_source_failed_export_cannot_release_registered_mapping(ledger):
    manager = source(ledger)
    manager.ensure_grant(offer(ledger))
    manager.bridge.export_snapshot = Mock(side_effect=RuntimeError("register cleanup failed"))
    with pytest.raises(RuntimeError):
        manager.export_after_d2h("request:0")
    assert ledger.transition("request:0", HostStageState.ABORTING, owner=manager.owner)
    assert ledger.mark_writer_rank_drained("request:0", 100, tp_rank=0, tp_size=1)
    manager.progress()
    assert manager.arena.used_bytes == 128
    manager.bridge.cleanup_allowed = True
    manager.progress()
    assert manager.arena.used_bytes == 0


def test_recovery_epoch_common_tp8_idempotent_and_advances_after_retry(ledger):
    ready_group(ledger, 8)
    assert ledger.assign_d2p_recovery_domain("request:0", 0)
    for rank in range(8):
        assert ledger.claim_d2p_recovery_rank("request:0", "p-group:p0",
            tp_rank=rank, tp_size=8, claim_id="child", recovery_domain=0)
        assert ledger.get("request:0")["remote_read_epoch"] == 1
    assert ledger.claim_d2p_recovery_rank("request:0", "p-group:p0",
        tp_rank=0, tp_size=8, claim_id="child", recovery_domain=0)
    assert ledger.get("request:0")["remote_read_epoch"] == 1
    assert ledger.request_d2p_retry("request:0", "p-group:p0", reason="read_failed")
    for rank in range(8):
        assert ledger.complete_d2p_retry_rank("request:0", "p-group:p0", tp_rank=rank, tp_size=8)
    for rank in range(8):
        assert ledger.claim_d2p_recovery_rank("request:0", "p-group:p0",
            tp_rank=rank, tp_size=8, claim_id="child", recovery_domain=0)
        assert ledger.get("request:0")["remote_read_epoch"] == 2


def test_remote_import_does_not_open_source_memfd(ledger):
    ready_group(ledger)
    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager.owner, manager.tp_rank, manager.tp_size = "p-group:p0", 0, 1
    manager._remote_host_bridge = object()
    manager.host_ready = {}
    manager.arena_domain = 0
    record = manager._import_remote_host_record("request:0", ledger.get("request:0"))
    assert isinstance(record["snapshot"], RemoteHostDescriptor)
    assert record["network_host"]
    assert record["snapshot"].materialize() is record["snapshot"]


def test_remote_unfenced_future_preserves_entire_workset():
    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    future = Future()
    future.set_exception(UnfencedRemoteRead("backend cancel failed"))
    load = {"remote_h2d_future": future}
    assert not manager._discard_failed_h2d_load("child", load)
    assert load["remote_h2d_unfenced"]
    assert not manager._discard_failed_h2d_load("child", load)


def test_remote_pending_future_does_not_release_workset():
    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    future = Future()
    future.set_running_or_notify_cancel()
    assert not manager._discard_failed_h2d_load("child", {"remote_h2d_future": future})


def test_prune_waits_for_source_physical_release(ledger):
    manager = ready_group(ledger)[0]
    assert ledger.transition("request:0", HostStageState.FAILED, owner=manager.owner)
    ledger.prune(older_than_seconds=-1)
    assert ledger.get("request:0") is not None
    assert ledger.complete_source_host_release_rank("request:0", manager.owner,
        tp_rank=0, tp_size=1)
    ledger.prune(older_than_seconds=-1)
    assert ledger.get("request:0") is None


def test_p_has_no_unused_source_host_allocation():
    arena = RemoteOnlyArena("/dev/shm/unused")
    assert arena.capacity_bytes == 0 and arena.path is None and arena.usage() == 0


def test_network_h2d_submission_is_only_io_worker_job(ledger):
    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager.ledger = ledger
    manager._remote_host_bridge = SimpleNamespace(load=Mock())
    manager._h2d_host_copy_pool = SimpleNamespace(submit=Mock(return_value=Future()))
    ready_group(ledger)
    ledger.assign_d2p_recovery_domain("request:0", 0)
    ledger.claim_d2p_recovery_rank("request:0", "p-group:p0", tp_rank=0,
        tp_size=1, claim_id="child", recovery_domain=0)
    load = dict(record={"network_host": True,
                        "snapshot": RemoteHostDescriptor({"byte_size":128,"token_count":16})},
        device_indices=[1,2,3], workset_lease=SimpleNamespace(), io_inflight=True,
        request_generation=SimpleNamespace(snapshot_id="request:0"))
    assert manager._start_h2d_chunk(load)
    assert load["remote_h2d_attempt"] == "child:epoch:1"
    manager._remote_host_bridge.load.assert_not_called()
    manager._h2d_host_copy_pool.submit.assert_called_once()


def test_worker_future_returns_elapsed_not_read_receipt():
    pool = HostCopyWorkerPool("test-remote-host", 1)
    assert isinstance(pool.submit(lambda: object()).result(timeout=2), float)


def test_unstarted_read_receipt_failure_never_releases_destination():
    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager._remote_host_bridge = SimpleNamespace(cancel_unstarted=Mock(
        side_effect=RuntimeError("control filesystem unavailable")))
    manager.workset_broker = Mock()
    load = dict(record={"network_host": True}, remote_h2d_future=Future(),
        remote_h2d_attempt="child:epoch:1",
        request_generation=SimpleNamespace(snapshot_id="request:0"))
    assert not manager._discard_failed_h2d_load("child", load)
    manager.workset_broker.request_release.assert_not_called()
    manager._remote_host_bridge.cancel_unstarted.assert_called_once_with(
        "request:0", attempt_id="child:epoch:1")


def test_peer_host_pressure_uses_rank0_group_eviction(ledger):
    sources = ready_group(ledger, 2)
    for rank in range(2):
        offer(ledger, rank, 2, sid="next:0")
    sources[0].ensure_grant(ledger.get("next:0"))
    # Rank0 has ample capacity, but peer cannot allocate its matching shard.
    sources[1].arena.used_bytes = 1024
    sources[1].ensure_grant(ledger.get("next:0"))
    assert ledger.get("next:0")["source_host_pressure_ranks"] == {"1":128}
    sources[1].progress()
    assert ledger.get("request:0")["state"] == "host_ready"
    sources[0].progress()
    assert ledger.get("request:0")["state"] == "evicting"


def test_source_extent_create_failure_aborts_group_before_d2h(ledger):
    manager = source(ledger)
    manager.arena.create = Mock(side_effect=MemoryError("backing extent"))
    with pytest.raises(MemoryError):
        manager.ensure_grant(offer(ledger))
    assert ledger.get("request:0")["state"] == "aborting"
    assert manager.records["request:0"]["snapshot"] is None
    ledger.mark_writer_rank_drained("request:0", 100, tp_rank=0, tp_size=1)
    manager.progress()
    assert not manager.records
    assert ledger.get("request:0")["state"] == "failed"
