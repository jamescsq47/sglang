from __future__ import annotations

import mmap
import os
import threading
import time
from types import SimpleNamespace

import pytest

from sglang.srt.disaggregation.agentic_group_protocol import (
    CommandKind,
    GenerationKey,
    GroupCommand,
    Owner,
)
from sglang.srt.disaggregation.agentic_remote_host import (
    HostShard,
    layout_fingerprint,
)
from sglang.srt.disaggregation.agentic_source_host import (
    DrainedSourceHostCopy,
    HostDirection,
    HostExtentPhase,
    SourceHostStoreExecutor,
    SourceHostStorePayload,
    SourceLocalHostArena,
    complete_snapshot_bytes,
    make_source_host_eviction_handler,
    make_source_host_store_handler,
    make_source_host_store_path,
)
from sglang.srt.disaggregation.agentic_transfer_queues import (
    FenceKind,
    PhysicalState,
    TransferAttempt,
    TransferCompletion,
    TransferPath,
)


class DType:
    itemsize = 2


class MHAPool:
    layer_num = 2
    head_num = 3
    head_dim = 4
    store_dtype = DType()


class FakeTensor:
    def __init__(self, shape, itemsize=2):
        self.shape = tuple(shape)
        self._itemsize = itemsize

    def numel(self):
        result = 1
        for value in self.shape:
            result *= value
        return result

    def element_size(self):
        return self._itemsize


class HybridPool:
    def __init__(self):
        self.full_kv_pool = MHAPool()
        cache = SimpleNamespace(
            conv=(FakeTensor((2, 16, 4)), FakeTensor((3, 16, 2))),
            temporal=FakeTensor((2, 16, 5)),
        )
        self.mamba_pool = SimpleNamespace(mamba_cache=cache)


class FakeSnapshot:
    def __init__(self, **values):
        self.__dict__.update(values)
        self.populated = False
        self.closed = False

    def mark_populated(self):
        self.populated = True

    def close(self, *, unlink=False):
        assert not unlink
        self.closed = True


def fake_snapshot_factory(**values):
    return FakeSnapshot(**values)


def test_mha_arena_allocates_complete_extent_and_recycles_memfd_space():
    expected = 2 * 10 * 2 * 3 * 4 * 2
    assert complete_snapshot_bytes(10, MHAPool(), 1) == expected
    arena = SourceLocalHostArena(
        direction=HostDirection.D2P,
        device_pool=MHAPool(),
        capacity_bytes=3 * mmap.ALLOCATIONGRANULARITY,
        snapshot_factory=fake_snapshot_factory,
    )
    first = arena.allocate(snapshot_id="a:0", token_count=10)
    assert first.byte_size == expected
    assert first.snapshot.path == arena.path
    assert arena.path.startswith(f"/proc/{__import__('os').getpid()}/fd/")
    assert arena.used_bytes == mmap.ALLOCATIONGRANULARITY
    assert arena.release(first.extent_id)
    second = arena.allocate(snapshot_id="b:0", token_count=10)
    assert second.offset == first.offset
    assert arena.release(second.extent_id)
    arena.close()


def test_direction_capacity_reads_only_new_multinode_environment(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_MULTINODE_D2P_HOST_GIB", "0.001")
    monkeypatch.setenv("SGLANG_AGENTIC_MULTINODE_P2D_HOST_GIB", "0.002")
    d2p = SourceLocalHostArena(
        direction=HostDirection.D2P,
        device_pool=MHAPool(),
        snapshot_factory=fake_snapshot_factory,
        preallocate=False,
    )
    p2d = SourceLocalHostArena(
        direction=HostDirection.P2D,
        device_pool=MHAPool(),
        snapshot_factory=fake_snapshot_factory,
        preallocate=False,
    )
    alignment = mmap.ALLOCATIONGRANULARITY
    assert d2p.capacity_bytes == int(0.001 * 1024**3) // alignment * alignment
    assert p2d.capacity_bytes == int(0.002 * 1024**3) // alignment * alignment
    assert os.path.exists(d2p.path) and os.path.exists(p2d.path)
    d2p.close()
    p2d.close()


def test_hybrid_extent_contains_attention_and_all_request_owned_state():
    pool = HybridPool()
    attention = complete_snapshot_bytes(7, pool.full_kv_pool, 1)
    state_bytes = (
        (2 * 4 + 3 * 2 + 2 * 5) * 2 * 2
    )  # layers*shape*itemsize*two state slots
    expected = (
        (attention + mmap.ALLOCATIONGRANULARITY - 1)
        // mmap.ALLOCATIONGRANULARITY
        * mmap.ALLOCATIONGRANULARITY
        + state_bytes
    )
    assert complete_snapshot_bytes(7, pool, 2) == expected
    arena = SourceLocalHostArena(
        direction=HostDirection.P2D,
        device_pool=pool,
        capacity_bytes=4 * mmap.ALLOCATIONGRANULARITY,
        snapshot_factory=fake_snapshot_factory,
    )
    extent = arena.allocate(snapshot_id="hybrid:4", token_count=7, state_slots=2)
    assert extent.byte_size == expected
    assert extent.state_slots == 2
    assert arena.release(extent.extent_id)
    arena.close()


class ImmediateCopy:
    def copy(self, extent, payload, cancel):
        assert not cancel.is_set()
        assert payload.extent_id == extent.extent_id


class BlockingReadyFence:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def synchronize(self):
        self.entered.set()
        assert self.release.wait(2)


class RecordingCopy(ImmediateCopy):
    def __init__(self):
        self.entered = threading.Event()

    def copy(self, extent, payload, cancel):
        self.entered.set()
        super().copy(extent, payload, cancel)


class FakeRemoteWorker:
    def __init__(self):
        self.exports = []

    def export_snapshot(self, snapshot_id, snapshot):
        assert snapshot.populated
        shard = HostShard(
            snapshot_id,
            f"export-{len(self.exports)}",
            0,
            1,
            layout_fingerprint({"kind": "fake"}),
            snapshot.token_count,
            4096,
            snapshot.byte_size,
            "",
            "fake-peer",
        )
        self.exports.append(shard)
        return shard

    def discard_unclaimed_export(self, _snapshot_id):
        return True


class FakeEvictionWorker:
    def __init__(self):
        self.reserved = set()
        self.finished = []

    def reserve_export_eviction(self, snapshot_id, eviction_id):
        self.reserved.add((snapshot_id, eviction_id))
        return True

    def cancel_export_eviction(self, snapshot_id, eviction_id):
        self.reserved.discard((snapshot_id, eviction_id))
        return True

    def finish_export_eviction(self, snapshot_id, eviction_id):
        identity = (snapshot_id, eviction_id)
        if identity not in self.reserved:
            return False
        self.reserved.remove(identity)
        self.finished.append(identity)
        return True


def transfer(extent_id):
    return TransferAttempt(
        "request:0",
        "1",
        "lease",
        TransferPath.D2P_HOST,
        SourceHostStorePayload(extent_id, source_indices=tuple(range(10))),
    )


def wait_terminal(executor, handle, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = executor.progress(handle)
        if value.state is not PhysicalState.INFLIGHT:
            return value
        time.sleep(0.001)
    raise AssertionError("Host copy did not complete")


def test_store_executor_exports_only_after_durable_copy_fence():
    arena = SourceLocalHostArena(
        direction=HostDirection.D2P,
        device_pool=MHAPool(),
        capacity_bytes=2 * mmap.ALLOCATIONGRANULARITY,
        snapshot_factory=fake_snapshot_factory,
    )
    extent = arena.allocate(snapshot_id="request:0", token_count=10)
    worker = FakeRemoteWorker()
    executor = SourceHostStoreExecutor(arena, worker, ImmediateCopy())
    notified = threading.Event()
    handle = executor.submit(transfer(extent.extent_id), notified.set)
    assert notified.wait(1.0)
    progress = wait_terminal(executor, handle)
    assert progress.state is PhysicalState.SUCCEEDED
    assert progress.fence is FenceKind.DMA_COMPLETE
    assert arena.get(extent.extent_id).phase is HostExtentPhase.EXPORTED
    assert len(worker.exports) == 1
    executor.close()
    assert arena.release(extent.extent_id)
    arena.close()


def test_group_eviction_fences_then_releases_complete_extent():
    arena = SourceLocalHostArena(
        direction=HostDirection.D2P,
        device_pool=MHAPool(),
        capacity_bytes=2 * mmap.ALLOCATIONGRANULARITY,
        snapshot_factory=fake_snapshot_factory,
    )
    extent = arena.allocate(snapshot_id="evict:0", token_count=10)
    arena.begin_copy(extent.extent_id)
    arena.mark_durable(extent.extent_id)
    arena.mark_exported(extent.extent_id, ("shard",))
    worker = FakeEvictionWorker()
    handler = make_source_host_eviction_handler(arena, worker)
    command = GroupCommand(
        GenerationKey("run", "evict", 0),
        2,
        1,
        CommandKind.PREPARE,
        Owner.D_HOST,
        Owner.NONE,
        "evict:evict:0",
        {
            "agentic_data_plane": {
                "version": 1,
                "path": "d2p_host",
                "operation": "host_evict",
            },
            "transfer": {"kind": "d2p_host_evict"},
        },
    )

    prepared = handler.prepare(command)
    assert arena.get(extent.extent_id).phase is HostExtentPhase.EVICTING
    handler.commit(command, prepared, SimpleNamespace())
    assert arena.used_bytes == 0
    assert worker.finished == [("evict:0", "evict:evict:0")]
    arena.close()


def test_host_store_preserves_forward_fence_and_waits_only_in_io_worker():
    arena = SourceLocalHostArena(
        direction=HostDirection.D2P,
        device_pool=MHAPool(),
        capacity_bytes=2 * mmap.ALLOCATIONGRANULARITY,
        snapshot_factory=fake_snapshot_factory,
    )
    fence = BlockingReadyFence()
    backend = RecordingCopy()
    handler = make_source_host_store_handler(
        arena,
        descriptor=lambda _command: SourceHostStorePayload(
            0, tuple(range(10)), ready_event=fence
        ),
        source_hbm_release=lambda _command, _extent: None,
        discard_durable_export=lambda extent: arena.release(extent.extent_id),
    )
    command = GroupCommand(
        GenerationKey("run", "fenced", 0),
        1,
        1,
        CommandKind.PREPARE,
        Owner.D_GPU,
        Owner.D_HOST,
        "lease",
        {
            "agentic_data_plane": {
                "version": 1,
                "path": "d2p_host",
                "operation": "host_store",
            },
            "transfer": {"token_count": 10, "state_slots": 1},
        },
    )
    prepared = handler.prepare(command)
    assert prepared.transfer_payload.ready_event is fence
    executor = SourceHostStoreExecutor(arena, FakeRemoteWorker(), backend)
    attempt = TransferAttempt(
        "fenced:0",
        "1",
        "lease",
        TransferPath.D2P_HOST,
        prepared.transfer_payload,
    )
    try:
        handle = executor.submit(attempt, lambda: None)
        assert fence.entered.wait(1)
        assert not backend.entered.is_set()
        fence.release.set()
        assert wait_terminal(executor, handle).state is PhysicalState.SUCCEEDED
        assert backend.entered.is_set()
    finally:
        fence.release.set()
        executor.close()
        assert arena.release(prepared.transfer_payload.extent_id)
        arena.close()


def test_path_factory_pairs_direction_with_queue_executor_and_handler():
    arena = SourceLocalHostArena(
        direction=HostDirection.P2D,
        device_pool=MHAPool(),
        capacity_bytes=2 * mmap.ALLOCATIONGRANULARITY,
        snapshot_factory=fake_snapshot_factory,
    )
    bundle = make_source_host_store_path(
        arena,
        FakeRemoteWorker(),
        ImmediateCopy(),
        descriptor=lambda _command: SourceHostStorePayload(0, tuple(range(10))),
        source_hbm_release=lambda _command, _extent: None,
    )
    assert bundle.path is TransferPath.P2D_HOST
    assert isinstance(bundle.executor, SourceHostStoreExecutor)
    bundle.executor.close()
    arena.close()


class CancelledCopy:
    def __init__(self):
        self.entered = threading.Event()

    def copy(self, _extent, _payload, cancel):
        self.entered.set()
        while not cancel.wait(0.001):
            pass
        raise DrainedSourceHostCopy("cancelled after current event drained")


def test_cancel_keeps_extent_until_real_copy_drain():
    arena = SourceLocalHostArena(
        direction=HostDirection.D2P,
        device_pool=MHAPool(),
        capacity_bytes=2 * mmap.ALLOCATIONGRANULARITY,
        snapshot_factory=fake_snapshot_factory,
    )
    extent = arena.allocate(snapshot_id="request:0", token_count=10)
    backend = CancelledCopy()
    executor = SourceHostStoreExecutor(arena, FakeRemoteWorker(), backend)
    notified = threading.Event()
    handle = executor.submit(transfer(extent.extent_id), notified.set)
    assert backend.entered.wait(1.0)
    assert not arena.release(extent.extent_id)
    executor.request_cancel(handle, notified.set)
    assert notified.wait(1.0)
    progress = wait_terminal(executor, handle)
    assert progress.state is PhysicalState.CANCELLED
    assert progress.fence is FenceKind.CANCEL_DRAINED
    assert arena.get(extent.extent_id).phase is HostExtentPhase.ALLOCATED
    assert arena.release(extent.extent_id)
    executor.close()
    arena.close()


def test_handler_releases_source_hbm_only_after_exported_host_commit():
    arena = SourceLocalHostArena(
        direction=HostDirection.D2P,
        device_pool=MHAPool(),
        capacity_bytes=2 * mmap.ALLOCATIONGRANULARITY,
        snapshot_factory=fake_snapshot_factory,
    )
    released = []
    handler = make_source_host_store_handler(
        arena,
        descriptor=lambda _command: SourceHostStorePayload(
            0, tuple(range(10))
        ),
        source_hbm_release=lambda command, extent: released.append(
            (command.attempt, extent.snapshot_id)
        ),
        discard_durable_export=lambda extent: arena.release(extent.extent_id),
    )
    command = GroupCommand(
        GenerationKey("run", "handler", 0),
        1,
        1,
        CommandKind.PREPARE,
        Owner.D_GPU,
        Owner.D_HOST,
        "lease",
        {
            "agentic_data_plane": {
                "version": 1,
                "path": "d2p_host",
                "operation": "host_store",
            },
            "transfer": {"token_count": 10, "state_slots": 1},
        },
    )
    prepared = handler.prepare(command)
    extent = arena.get(prepared.transfer_payload.extent_id)
    with pytest.raises(RuntimeError, match="durability"):
        handler.commit(command, prepared, SimpleNamespace())
    arena.begin_copy(extent.extent_id)
    arena.mark_durable(extent.extent_id)
    arena.mark_exported(extent.extent_id, ("shard",))
    handler.commit(command, prepared, SimpleNamespace())
    assert released == [(1, "handler:0")]
    assert arena.release(extent.extent_id)
    arena.close()


def test_prepare_abort_reclaims_complete_arena_capacity():
    capacity = 2 * mmap.ALLOCATIONGRANULARITY
    arena = SourceLocalHostArena(
        direction=HostDirection.D2P,
        device_pool=MHAPool(),
        capacity_bytes=capacity,
        snapshot_factory=fake_snapshot_factory,
    )
    handler = make_source_host_store_handler(
        arena,
        descriptor=lambda _command: SourceHostStorePayload(0, tuple(range(10))),
        source_hbm_release=lambda _command, _extent: None,
        discard_durable_export=lambda extent: arena.release(extent.extent_id),
    )
    prepare = GroupCommand(
        GenerationKey("run", "abort", 0),
        1,
        1,
        CommandKind.PREPARE,
        Owner.D_GPU,
        Owner.D_HOST,
        "lease",
        {
            "agentic_data_plane": {
                "version": 1,
                "path": "d2p_host",
                "operation": "host_store",
            },
            "transfer": {"token_count": 10},
        },
    )
    prepared = handler.prepare(prepare)
    assert arena.used_bytes == mmap.ALLOCATIONGRANULARITY
    handler.abort(prepare, prepared, None)
    assert arena.used_bytes == 0
    assert arena.capacity_bytes == capacity
    arena.close()


def test_exported_group_abort_discards_unclaimed_export_and_reclaims_extent():
    arena = SourceLocalHostArena(
        direction=HostDirection.D2P,
        device_pool=MHAPool(),
        capacity_bytes=2 * mmap.ALLOCATIONGRANULARITY,
        snapshot_factory=fake_snapshot_factory,
    )
    worker = FakeRemoteWorker()
    bundle = make_source_host_store_path(
        arena,
        worker,
        ImmediateCopy(),
        descriptor=lambda _command: SourceHostStorePayload(0, tuple(range(10))),
        source_hbm_release=lambda _command, _extent: None,
    )
    prepare = GroupCommand(
        GenerationKey("run", "abort-export", 0),
        1,
        1,
        CommandKind.PREPARE,
        Owner.D_GPU,
        Owner.D_HOST,
        "lease",
        {
            "agentic_data_plane": {
                "version": 1,
                "path": "d2p_host",
                "operation": "host_store",
            },
            "transfer": {"token_count": 10},
        },
    )
    prepared = bundle.handler.prepare(prepare)
    attempt = TransferAttempt(
        prepare.key.snapshot_id,
        "1",
        prepare.lease_id,
        TransferPath.D2P_HOST,
        prepared.transfer_payload,
    )
    handle = bundle.executor.submit(attempt, lambda: None)
    assert wait_terminal(bundle.executor, handle).state is PhysicalState.SUCCEEDED
    now = time.monotonic()
    completion = TransferCompletion(
        attempt,
        PhysicalState.SUCCEEDED,
        FenceKind.DMA_COMPLETE,
        None,
        now,
        now,
    )
    bundle.handler.abort(prepare, prepared, completion)
    assert arena.used_bytes == 0
    bundle.executor.close()
    arena.close()
