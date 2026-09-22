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


def _consumed_workset_group(ledger, size):
    sources = ready_group(ledger, size)
    owner, claim = sources[0].owner, "slow:request:0:child"
    for rank in range(size):
        assert ledger.claim_d2p_recovery_rank("request:0", owner,
            tp_rank=rank, tp_size=size, claim_id=claim)
        assert ledger.attach_d2p_recovery_lease_rank("request:0", owner,
            tp_rank=rank, tp_size=size, claim_id=claim, lease_id=rank + 1)
        assert ledger.mark_d2p_recovery_phase_rank("request:0", owner,
            tp_rank=rank, tp_size=size, claim_id=claim, lease_id=rank + 1,
            phase="io_inflight")
    for rank in range(size):
        assert ledger.complete_d2p_host_load_rank("request:0", owner,
            tp_rank=rank, tp_size=size)
    for rank in range(size):
        assert ledger.complete_host_bind_rank("request:0", owner,
            tp_rank=rank, tp_size=size)
    assert ledger.get("request:0")["state"] == "consumed"
    for manager in sources:
        manager.bridge.cleanup_allowed = True
        assert manager._cleanup_one("request:0", ledger.get("request:0"))
        assert manager.arena.used_bytes == 0
    return owner, claim


@pytest.mark.parametrize("size", [1, 2, 8])
def test_consumed_prune_waits_for_delayed_allrank_workset_handoff(ledger, monkeypatch, size):
    import sglang.srt.disaggregation.agentic_host_staging as host
    owner, claim = _consumed_workset_group(ledger, size)
    clock = [host.time.time() + 600]
    monkeypatch.setattr(host.time, "time", lambda: clock[0])
    # The complete Host copy has already been freed, but its tiny lifecycle
    # record is still required by a delayed scheduler COMMIT on every rank.
    for rank in range(size):
        ledger.prune()
        assert ledger.get("request:0") is not None
        assert ledger.mark_d2p_recovery_phase_rank("request:0", owner,
            tp_rank=rank, tp_size=size, claim_id=claim, lease_id=rank + 1,
            phase="handed")
        clock[0] += 600
    ledger.prune()
    assert ledger.get("request:0") is None


@pytest.mark.parametrize("size", [1, 2, 8])
def test_prune_rechecks_current_handoff_under_entry_lock(ledger, monkeypatch, size):
    import copy
    owner, claim = _consumed_workset_group(ledger, size)
    # A stale scan can claim this entry is reclaimable. Only the authoritative
    # entry read under the deletion lock may decide to remove ownership.
    stale = copy.deepcopy(ledger.get("request:0"))
    stale["updated_at"] = 0
    for item in stale["recovery_claims"].values():
        item["phase"] = "handed"
    monkeypatch.setattr(ledger, "snapshot_entries", lambda **_: {"request:0": stale})
    ledger.prune(consumed_older_than_seconds=0)
    assert ledger.get("request:0") is not None
    for rank in range(size):
        assert ledger.mark_d2p_recovery_phase_rank("request:0", owner,
            tp_rank=rank, tp_size=size, claim_id=claim, lease_id=rank + 1,
            phase="handed")
    ledger.prune(consumed_older_than_seconds=0)
    assert ledger.get("request:0") is None


@pytest.mark.parametrize("terminal", [False, True])
def test_prune_preserves_legacy_and_explicit_terminal_cleanup(ledger, terminal):
    manager = ready_group(ledger)[0]
    if terminal:
        assert ledger.begin_host_eviction("request:0", manager.owner,
            tp_size=1, reason="application_final", terminal_on_release=True)
    else:
        assert ledger.complete_d2p_host_load_rank("request:0", manager.owner,
            tp_rank=0, tp_size=1)
        assert ledger.complete_host_bind_rank("request:0", manager.owner,
            tp_rank=0, tp_size=1)
    manager.bridge.cleanup_allowed = True
    assert manager._cleanup_one("request:0", ledger.get("request:0"))
    assert ledger.get("request:0")["state"] == "consumed"
    ledger.prune(consumed_older_than_seconds=0)
    assert ledger.get("request:0") is None


@pytest.mark.parametrize("size", [1, 2, 8])
@pytest.mark.parametrize("failure", ["host_cleanup", "handoff_ack"])
def test_gate_handoff_survives_prune_ttl_and_cleanup_retry(ledger, monkeypatch, size, failure):
    import threading
    import sglang.srt.disaggregation.agentic_host_staging as host
    owner, claim = _consumed_workset_group(ledger, size)
    clock = [host.time.time() + 600]
    monkeypatch.setattr(host.time, "time", lambda: clock[0])
    request = SimpleNamespace(snapshot_id="request:0")
    managers = []
    original_mark = ledger.mark_d2p_recovery_phase_rank
    fail_ack = {rank for rank in range(size)} if failure == "handoff_ack" else set()

    def mark(*args, **kwargs):
        rank = kwargs["tp_rank"]
        if rank in fail_ack:
            fail_ack.remove(rank)
            raise OSError("transient ledger ACK failure")
        return original_mark(*args, **kwargs)

    monkeypatch.setattr(ledger, "mark_d2p_recovery_phase_rank", mark)
    for rank in range(size):
        req = SimpleNamespace(rid="child", origin_input_ids=list(range(17)), extra_key=None)
        lease = SimpleNamespace(lease_id=rank + 1, owner=claim)
        record = dict(snapshot=object(), offer=dict(token_count=16, byte_size=128), loading="h2d")
        load = dict(record=record, request_generation=request, device_indices=list(range(16)),
                    workset_lease=lease, recovery_claim_id=claim, io_complete=True,
                    io_error=None, radix_bound=True, host_released=False)
        manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
        manager._state_lock = threading.RLock()
        manager.tp_rank, manager.tp_size, manager.owner = rank, size, owner
        manager.ledger = ledger
        manager.loads = {req.rid: load}
        manager.host_ready = {request.snapshot_id: record}
        manager._h2d_lane_reservations = {request.snapshot_id: 0}
        manager.workset_broker = SimpleNamespace(handoff_to_req=Mock())
        manager._complete_shared_host_manifest = lambda _: True
        manager._release_record = Mock(side_effect=[False, True] if failure == "host_cleanup" else [True])
        manager.tp_host_commit_snapshot = request.snapshot_id
        managers.append((manager, req, load))
        if size > 1:
            # Actual BIND gate hands the load descriptor to the live request;
            # neither this transition nor physical source release is COMMIT.
            assert manager.gate_request(req, request) is True
            assert req._agentic_host_rank_loaded

    ledger.prune()
    assert ledger.get(request.snapshot_id) is not None
    for manager, req, load in managers:
        assert manager.gate_request(req, request) is True
        assert ledger.get(request.snapshot_id)["recovery_claims"][str(manager.tp_rank)]["phase"] == "io_inflight"
        clock[0] += 600
        ledger.prune()
        assert ledger.get(request.snapshot_id) is not None
        # Cleanup may already have succeeded when the final ACK failed. The
        # same bound lease must be retried without reopening/reloading Host.
        assert manager.gate_request(req, request) is False
        assert req._agentic_kv_gate_complete
        assert not manager.loads
        assert not manager.host_ready
        assert not manager._h2d_lane_reservations
        assert manager._release_record.call_count == (2 if failure == "host_cleanup" else 1)
    clock[0] += 600
    ledger.prune()
    assert ledger.get(request.snapshot_id) is None


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


@pytest.mark.parametrize('size', [1, 8])
@pytest.mark.parametrize('quiesce_fails', [False, True])
def test_peer_retry_before_h2d_releases_real_broker_lease(ledger, size, quiesce_fails):
    import torch
    from sglang.srt.managers.scheduler import AgenticPWorksetLeaseBroker
    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager.ledger, manager.owner = ledger, 'p-group:p0'
    manager.tp_rank, manager.tp_size = size - 1, size
    manager._remote_host_bridge = SimpleNamespace(cancel_unstarted=Mock())
    manager._h2d_host_copy_pool = SimpleNamespace(submit=Mock())
    ready_group(ledger, size)
    ledger.assign_d2p_recovery_domain('request:0', 0)
    for rank in range(size):
        assert ledger.claim_d2p_recovery_rank('request:0', manager.owner,
            tp_rank=rank, tp_size=size, claim_id='child', recovery_domain=0)
    broker = manager.workset_broker = AgenticPWorksetLeaseBroker(page_size=4)
    allocator = SimpleNamespace(alloc=lambda count: torch.arange(count), free=Mock())
    broker.request('request:0', parent_tokens=16, prompt_tokens=20)
    broker.service(allocator)
    lease = broker.get('request:0')
    assert broker.begin_io_attempt('request:0', lease, 'attempt')
    load = dict(record={'network_host': True, 'loading': False},
        device_indices=lease.device_indices, workset_lease=lease, io_attempt='attempt',
        request_generation=SimpleNamespace(snapshot_id='request:0'), recovery_claim_id='child')
    manager.loads, manager.host_ready = {'child': load}, {}
    mark_phase = ledger.mark_d2p_recovery_phase_rank
    def peer_enters_retry(*args, **kwargs):
        assert lease.state == 'io_inflight'
        assert ledger.request_d2p_retry('request:0', manager.owner, reason='peer_registration_failed')
        for rank in range(size - 1):
            assert ledger.complete_d2p_retry_rank('request:0', manager.owner, tp_rank=rank, tp_size=size)
        return mark_phase(*args, **kwargs)
    ledger.mark_d2p_recovery_phase_rank = peer_enters_retry
    if quiesce_fails:
        broker.mark_io_quiesced = Mock(return_value=False)
    with pytest.raises(RuntimeError, match='lost Host ownership|rollback lost'):
        manager._start_h2d_chunk(load)
    if quiesce_fails:
        assert load['dma_quarantined'] and lease.state == 'io_inflight'
        assert not manager._discard_failed_h2d_load('child', load)
        allocator.free.assert_not_called()
        manager._h2d_host_copy_pool.submit.assert_not_called()
        return
    assert load['io_quiesced'] and lease.state == 'active'
    manager._h2d_host_copy_pool.submit.assert_not_called()
    assert not manager._discard_failed_h2d_load('child', load)
    assert manager.loads['child'] is load
    broker.service(allocator)
    assert manager._discard_failed_h2d_load('child', load)
    assert 'child' not in manager.loads
    assert ledger.get('request:0')['state'] == HostStageState.HOST_READY.value
    assert manager.host_ready['request:0'] is load['record']
    broker.service(allocator)
    allocator.free.assert_called_once()


@pytest.mark.parametrize("size", [2, 8])
def test_prestart_retry_publishes_every_rank_receipt_before_next_epoch(ledger, tmp_path, size):
    from sglang.srt.disaggregation.test_agentic_remote_host_engine import bridge, publish
    from sglang.srt.managers.scheduler import AgenticPWorksetLeaseBroker
    sid = "request:0"
    ready_group(ledger, size)
    ledger.assign_d2p_recovery_domain(sid, 0)
    claim = AgenticPWorksetLeaseBroker.slow_owner(sid, "child")
    targets, grants = [], []
    for rank in range(size):
        src, _ = bridge(tmp_path, rank=rank, size=size, direction="d2p")
        grants.append(publish(src, sid))
        target, agent = bridge(tmp_path, rank=rank, size=size, node="p", direction="d2p")
        targets.append((target, agent))
        assert ledger.claim_d2p_recovery_rank(sid, "p-group:p0", tp_rank=rank,
            tp_size=size, claim_id=claim, recovery_domain=0)
    old = claim + ":epoch:1"
    # One rank reaches connection preparation and fails before READ; the
    # others still have no load/future when the shared group enters RETRY.
    agent = targets[0][1]
    agent.add_remote_agent = Mock(side_effect=RuntimeError("QP creation failed"))
    with pytest.raises(RuntimeError, match="QP creation failed"):
        targets[0][0].load(sid, grants[0], [10, 11, 12], attempt_id=old)
    assert ledger.request_d2p_retry(sid, "p-group:p0", reason="QP failed")
    for rank, (target, _) in enumerate(targets):
        m = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
        m.owner, m.tp_rank, m.tp_size = "p-group:p0", rank, size
        m.ledger, m._remote_host_bridge = ledger, target
        m.workset_broker = AgenticPWorksetLeaseBroker(1)
        m.loads, m.host_ready = {}, {sid: {"network_host": True, "loading": "h2d_reserving"}}
        m._h2d_lane_reservations = {sid: 0}
        m._h2d_resident_reservations = {sid}
        req = SimpleNamespace(rid="child")
        assert m.gate_request(req, SimpleNamespace(snapshot_id=sid)) is True
        assert not m._h2d_lane_reservations
        if rank + 1 < size:
            assert ledger.get(sid)["state"] == "retry_pending"
    assert ledger.get(sid)["state"] == "host_ready"
    receipts = targets[0][0]._receipts(sid, old)
    assert len(receipts) == size and all(r.outcome == "drained" for r in receipts)
    # The next epoch can really enter the bridge; no missing-receipt
    # UnfencedRemoteRead and no destination quarantine from the old failure.
    for rank in range(size):
        ledger.claim_d2p_recovery_rank(sid, "p-group:p0", tp_rank=rank,
            tp_size=size, claim_id=claim, recovery_domain=0)
    assert ledger.get(sid)["remote_read_epoch"] == 2
    targets[-1][0]._claim_attempt(sid, claim + ":epoch:2")


def test_prestart_retry_no_receipt_does_not_ack_or_release(ledger):
    from sglang.srt.managers.scheduler import AgenticPWorksetLeaseBroker
    sid = "request:0"
    ready_group(ledger)
    ledger.assign_d2p_recovery_domain(sid, 0)
    claim = AgenticPWorksetLeaseBroker.slow_owner(sid, "child")
    ledger.claim_d2p_recovery_rank(sid, "p-group:p0", tp_rank=0,
        tp_size=1, claim_id=claim, recovery_domain=0)
    ledger.request_d2p_retry(sid, "p-group:p0", reason="peer")
    m = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    m.owner, m.tp_rank, m.tp_size = "p-group:p0", 0, 1
    m.ledger, m.loads, m.host_ready = ledger, {}, {}
    m._remote_host_bridge = SimpleNamespace(cancel_unstarted=Mock(side_effect=OSError("NFS")))
    m.workset_broker = AgenticPWorksetLeaseBroker(1)
    m.workset_broker.request(sid, 16, 20, owner=claim)
    m._h2d_lane_reservations = {sid: 0}
    req = SimpleNamespace(rid="child")
    assert m.gate_request(req, SimpleNamespace(snapshot_id=sid)) is True
    assert req._agentic_remote_retry_entry["remote_read_epoch"] == 1
    assert ledger.get(sid)["retry_acks"] == []
    assert m.workset_broker.owner_has_unretired_work(sid, owner=claim)
    assert m._h2d_lane_reservations == {sid: 0}


def test_remote_retry_retains_load_until_durable_ack_even_after_ambiguous_commit(ledger):
    import torch
    from sglang.srt.managers.scheduler import AgenticPWorksetLeaseBroker
    sid = "request:0"
    ready_group(ledger)
    ledger.assign_d2p_recovery_domain(sid, 0)
    claim = AgenticPWorksetLeaseBroker.slow_owner(sid, "child")
    ledger.claim_d2p_recovery_rank(sid, "p-group:p0", tp_rank=0,
        tp_size=1, claim_id=claim, recovery_domain=0)
    m = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    m.owner, m.tp_rank, m.tp_size = "p-group:p0", 0, 1
    m.ledger = ledger
    m._remote_host_bridge = SimpleNamespace(cancel_unstarted=Mock())
    broker = m.workset_broker = AgenticPWorksetLeaseBroker(1)
    allocator = SimpleNamespace(alloc=lambda n: torch.arange(n), free=Mock())
    broker.request(sid, 16, 20, owner=claim)
    broker.service(allocator)
    lease = broker.get(sid, owner=claim)
    broker.begin_io_attempt(sid, lease, "read")
    future = Future()
    future.set_exception(RuntimeError("QP failed, drained"))
    load = dict(record={"network_host": True}, request_generation=SimpleNamespace(snapshot_id=sid),
        workset_lease=lease, io_attempt="read", remote_h2d_future=future,
        remote_h2d_attempt=claim + ":epoch:1", io_error=RuntimeError("QP failed"))
    m.loads, m.host_ready = {"child": load}, {}
    m._h2d_lane_reservations = {sid: 0}
    m._h2d_resident_reservations = {sid}
    assert not m._discard_failed_h2d_load("child", load)
    broker.service(allocator)
    original = ledger.complete_d2p_retry_rank
    ledger.complete_d2p_retry_rank = Mock(return_value=False)
    assert not m._discard_failed_h2d_load("child", load)
    assert m.loads["child"] is load and m._h2d_lane_reservations == {sid: 0}
    def commit_then_raise(*args, **kwargs):
        assert original(*args, **kwargs)
        raise OSError("ACK reply lost")
    ledger.complete_d2p_retry_rank = commit_then_raise
    with pytest.raises(OSError, match="ACK reply lost"):
        m._discard_failed_h2d_load("child", load)
    assert ledger.get(sid)["state"] == "host_ready" and m.loads["child"] is load
    ledger.complete_d2p_retry_rank = original
    assert m._discard_failed_h2d_load("child", load)
    assert not m.loads and not m._h2d_lane_reservations
    allocator.free.assert_called_once()


def test_peer_abort_supersedes_local_retry_without_waiting_for_http_abort():
    m = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    load = dict(request_generation=SimpleNamespace(snapshot_id="request:0"),
                io_error=RuntimeError("QP failed"), record={"network_host": True})
    m.loads = {"child": load}
    m._h2d_poisoned = False
    m._remote_host_bridge = object()
    m.ledger = SimpleNamespace(get=lambda sid: dict(state="aborting", h2d_abort_started=True))
    m._discard_failed_h2d_load = Mock(return_value=False)
    m._progress_h2d_loads()
    assert load["abort_requested"] and load["drop_host_on_abort"]
    m._discard_failed_h2d_load.assert_called_once_with("child", load)
    assert m.loads["child"] is load  # only original physical-fence path may free


def test_stale_remote_retry_ack_cannot_ack_successor_epoch(ledger):
    sid = "request:0"
    ready_group(ledger)
    ledger.assign_d2p_recovery_domain(sid, 0)
    claim_args = dict(tp_rank=0, tp_size=1, claim_id="child", recovery_domain=0)
    ledger.claim_d2p_recovery_rank(sid, "p-group:p0", **claim_args)
    ledger.request_d2p_retry(sid, "p-group:p0", reason="first")
    assert ledger.complete_d2p_retry_rank(sid, "p-group:p0", tp_rank=0,
        tp_size=1, remote_read_epoch=1)
    ledger.claim_d2p_recovery_rank(sid, "p-group:p0", **claim_args)
    ledger.request_d2p_retry(sid, "p-group:p0", reason="second")
    before = ledger.get(sid)
    assert before["remote_read_epoch"] == 2 and before["retry_acks"] == []
    assert ledger.complete_d2p_retry_rank(sid, "p-group:p0", tp_rank=0,
        tp_size=1, remote_read_epoch=1)
    assert ledger.get(sid) == before


@pytest.mark.parametrize("size", [2, 8])
def test_remote_retry_rebuilds_real_prepared_load_in_next_epoch(ledger, size):
    import torch
    from sglang.srt.disaggregation.agentic_kv_lifecycle import token_ids_digest
    from sglang.srt.disaggregation.test_agentic_h2d_decoupling import manager
    from sglang.srt.managers.scheduler import AgenticPWorksetLeaseBroker

    sid = "request:0"
    ready_group(ledger, size)
    ledger.assign_d2p_recovery_domain(sid, 0)
    parent = SimpleNamespace(snapshot_id=sid)
    req = SimpleNamespace(rid="child", origin_input_ids=list(range(20)))
    contexts = []
    for rank in range(size):
        m = manager(enabled=False)
        m.h2d_lane_overlap = True
        m.owner, m.tp_rank, m.tp_size, m.arena_domain = "p-group:p0", rank, size, 0
        m.ledger = ledger
        m._remote_host_bridge = SimpleNamespace(cancel_unstarted=Mock())
        m.active, m.aborting = {}, {}
        m.host_ready = {sid: dict(network_host=True, loading=False,
            snapshot=SimpleNamespace(_materialized=True),
            offer=dict(token_count=16, byte_size=128,
                       token_digest=token_ids_digest(req.origin_input_ids[:16])))}
        m.workset_broker = AgenticPWorksetLeaseBroker(1)
        allocator = SimpleNamespace(alloc=lambda n: torch.arange(n), free=Mock())
        assert m._prepare_host_restore(req, parent) is True
        m.workset_broker.service(allocator)
        assert m._prepare_host_restore(req, parent) is True
        assert m.loads[req.rid]["remote_h2d_attempt"].endswith(":epoch:1")
        contexts.append((m, allocator, m.loads[req.rid]["workset_lease"]))
    assert ledger.request_d2p_retry(sid, "p-group:p0", reason="connection_failed")
    for m, allocator, lease in contexts:
        load = m.loads[req.rid]
        load["io_error"] = RuntimeError("peer connection failure before READ")
        assert not m._discard_failed_h2d_load(req.rid, load)
        m.workset_broker.service(allocator)
        assert m._discard_failed_h2d_load(req.rid, load)
        assert "remote_h2d_attempt" not in m.host_ready[sid]
        assert "recovery_claim_id" not in m.host_ready[sid]
    assert ledger.get(sid)["state"] == "host_ready"
    for m, allocator, old_lease in contexts:
        # Exercise actual prepare twice (intent -> granted load), not just
        # bridge._claim_attempt: stale record epochs used to poison this path.
        assert m._prepare_host_restore(req, parent) is True
        m.workset_broker.service(allocator)
        assert m._prepare_host_restore(req, parent) is True
        load = m.loads[req.rid]
        assert load["remote_h2d_attempt"].endswith(":epoch:2")
        assert load["workset_lease"].lease_id != old_lease.lease_id
        assert not load["ledger_prepare_pending"]


@pytest.mark.parametrize("via_retry_context", [False, True])
def test_prestart_retry_abort_clears_req_context_and_reaches_terminal_gate(
    ledger, tmp_path, via_retry_context,
):
    from sglang.srt.disaggregation.test_agentic_h2d_decoupling import manager
    from sglang.srt.disaggregation.test_agentic_remote_host_engine import bridge, publish
    from sglang.srt.managers.scheduler import AgenticPWorksetLeaseBroker
    sid = "request:0"
    ready_group(ledger)
    ledger.assign_d2p_recovery_domain(sid, 0)
    m = manager(enabled=False)
    m.owner, m.tp_rank, m.tp_size, m.arena_domain = "p-group:p0", 0, 1, 0
    m.ledger, m.workset_broker = ledger, AgenticPWorksetLeaseBroker(1)
    src, _ = bridge(tmp_path, direction="d2p")
    publish(src, sid)
    m._remote_host_bridge, _ = bridge(tmp_path, node="p", direction="d2p")
    m.active, m.aborting = {}, {}
    m.host_ready = {sid: dict(network_host=True, remote_host=True, loading="h2d_reserving")}
    m._release_record = Mock(return_value=True)
    m._release_consumed_owned_host = Mock()
    claim = m.workset_broker.slow_owner(sid, "child")
    assert ledger.claim_d2p_recovery_rank(sid, m.owner, tp_rank=0,
        tp_size=1, claim_id=claim, recovery_domain=0)
    assert ledger.request_d2p_retry(sid, m.owner, reason="peer")
    old_attempt = claim + ":epoch:1"
    m._remote_host_bridge._claim_attempt(sid, old_attempt)
    req = SimpleNamespace(rid="child", _agentic_remote_retry_entry=ledger.get(sid),
        _agentic_host_retry_reason="peer", _agentic_tp_host_failed=True)
    assert ledger.request_host_load_failure(sid, m.owner, reason="cancelled")
    parent = SimpleNamespace(snapshot_id=sid)
    if via_retry_context:
        assert m.gate_request(req, parent) is True
    else:
        # Direct cancellation reaches the same prestart worker without the
        # retry-gate helper; it too must publish the source's no-I/O receipt.
        del req._agentic_remote_retry_entry
        del req._agentic_host_retry_reason
        del req._agentic_tp_host_failed
        m.abort_request(req.rid, parent)
    assert not hasattr(req, "_agentic_remote_retry_entry")
    assert not hasattr(req, "_agentic_host_retry_reason")
    assert ledger.get(sid)["state"] == "failed"
    receipts = m._remote_host_bridge._receipts(sid, old_attempt)
    assert len(receipts) == 1 and receipts[0].outcome == "drained"
    assert src.cleanup_source(sid, ledger.get(sid))
    assert m.gate_request(req, parent) is False
    assert req._agentic_kv_gate_complete
    assert req._agentic_kv_fallback == "shared_host_h2d_failed"


def test_prestart_retry_cancels_frozen_owner_not_new_http_rid(ledger):
    from sglang.srt.disaggregation.test_agentic_h2d_decoupling import manager
    from sglang.srt.managers.scheduler import AgenticPWorksetLeaseBroker
    sid = "request:0"
    ready_group(ledger)
    ledger.assign_d2p_recovery_domain(sid, 0)
    m = manager(enabled=False)
    m.owner, m.tp_rank, m.tp_size = "p-group:p0", 0, 1
    m.ledger, m.workset_broker = ledger, AgenticPWorksetLeaseBroker(1)
    m._remote_host_bridge = SimpleNamespace(cancel_unstarted=Mock())
    m.host_ready = {}
    old_owner = m.workset_broker.slow_owner(sid, "old-http-rid")
    m.workset_broker.request(sid, 16, 20, owner=old_owner)
    assert ledger.claim_d2p_recovery_rank(sid, m.owner, tp_rank=0,
        tp_size=1, claim_id=old_owner, recovery_domain=0)
    assert ledger.request_d2p_retry(sid, m.owner, reason="peer")
    assert m.gate_request(SimpleNamespace(rid="new-http-rid"),
                          SimpleNamespace(snapshot_id=sid)) is True
    assert not m.workset_broker.owner_has_unretired_work(sid, owner=old_owner)
    assert ledger.get(sid)["state"] == "host_ready"


@pytest.mark.parametrize("previous_retry", [False, True])
def test_remote_unclaimed_host_abort_does_not_invent_read_attempt(ledger, previous_retry):
    from sglang.srt.disaggregation.test_agentic_h2d_decoupling import manager
    from sglang.srt.managers.scheduler import AgenticPWorksetLeaseBroker
    sid = "request:0"
    ready_group(ledger)
    ledger.assign_d2p_recovery_domain(sid, 0)
    m = manager(enabled=False)
    m.owner, m.tp_rank, m.tp_size, m.arena_domain = "p-group:p0", 0, 1, 0
    m.ledger, m.workset_broker = ledger, AgenticPWorksetLeaseBroker(1)
    m._remote_host_bridge = SimpleNamespace(cancel_unstarted=Mock())
    m.host_ready = {sid: dict(network_host=True, remote_host=True, loading=False)}
    m._release_record = Mock(return_value=True)
    m._release_consumed_owned_host = Mock()
    if previous_retry:
        assert ledger.claim_d2p_recovery_rank(sid, m.owner, tp_rank=0,
            tp_size=1, claim_id="old-child", recovery_domain=0)
        assert ledger.request_d2p_retry(sid, m.owner, reason="old")
        assert ledger.complete_d2p_retry_rank(sid, m.owner, tp_rank=0,
            tp_size=1, remote_read_epoch=1)
    m.abort_request("new-http-rid", SimpleNamespace(snapshot_id=sid))
    assert ledger.get(sid)["state"] == "failed"
    m._remote_host_bridge.cancel_unstarted.assert_not_called()


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
