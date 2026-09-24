"""Source-local physical Host staging for V2 multi-node KV transfers.

Each TP rank owns one anonymous, preallocated memfd arena.  Complete
request-generation snapshots receive immutable extents in that arena.  D2H is
launched on a private CUDA stream and a worker blocks on the real CUDA event;
only then is the extent marked durable and exported by
``RemoteHostRankWorker``.  There is no manifest, marker, directory scan, NFS
access, lifecycle policy, routing, or scheduler dependency in this module.
"""

from __future__ import annotations

import mmap
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Protocol

from sglang.srt.disaggregation.agentic_group_protocol import GroupCommand
from sglang.srt.disaggregation.agentic_group_transfer import (
    CallbackPathHandler,
    PreparedRankTransfer,
)
from sglang.srt.disaggregation.agentic_transfer_queues import (
    FenceKind,
    PhysicalProgress,
    PhysicalState,
    TransferAttempt,
    TransferExecutor,
    TransferPath,
)


_GIB = 1024**3


class HostDirection(str, Enum):
    D2P = "d2p"
    P2D = "p2d"


class HostExtentPhase(str, Enum):
    ALLOCATED = "allocated"
    COPYING = "copying"
    DURABLE = "durable"
    EXPORTED = "exported"
    RELEASED = "released"


class UnfencedSourceHostCopy(RuntimeError):
    """A submitted CUDA operation has no proven terminal fence."""


class DrainedSourceHostCopy(RuntimeError):
    """A failed or cancelled copy is physically quiescent."""


@dataclass(slots=True)
class SourceHostExtent:
    extent_id: int
    snapshot_id: str
    token_count: int
    byte_size: int
    allocation_bytes: int
    offset: int
    state_slots: int
    snapshot: Any
    phase: HostExtentPhase = HostExtentPhase.ALLOCATED
    exported_shard: Any = None


def _mha_bytes(token_count: int, pool) -> int:
    return (
        2
        * int(token_count)
        * int(pool.layer_num)
        * int(pool.head_num)
        * int(pool.head_dim)
        * int(pool.store_dtype.itemsize)
    )


def complete_snapshot_bytes(token_count: int, device_pool, state_slots: int) -> int:
    """Return complete MHA or Qwen3.5 Attention+Mamba wire bytes."""

    token_count = int(token_count)
    state_slots = int(state_slots)
    if token_count <= 0 or state_slots <= 0:
        raise ValueError("positive token and state-slot counts are required")
    if not hasattr(device_pool, "mamba_pool"):
        return _mha_bytes(token_count, device_pool)
    attention = device_pool.full_kv_pool
    attention_bytes = _mha_bytes(token_count, attention)
    state_offset = (
        (attention_bytes + mmap.ALLOCATIONGRANULARITY - 1)
        // mmap.ALLOCATIONGRANULARITY
        * mmap.ALLOCATIONGRANULARITY
    )
    cache = device_pool.mamba_pool.mamba_cache
    tensors = (*cache.conv, cache.temporal)
    state_bytes = 0
    for tensor in tensors:
        if int(tensor.shape[1]) <= 0:
            raise ValueError("hybrid state pool has no request slots")
        bytes_per_slot = int(tensor.numel()) * int(tensor.element_size()) // int(
            tensor.shape[1]
        )
        state_bytes += state_slots * bytes_per_slot
    return state_offset + state_bytes


class SnapshotFactory(Protocol):
    def __call__(
        self,
        *,
        path: str,
        token_count: int,
        device_pool: Any,
        byte_size: int,
        file_offset: int,
        state_slots: int,
    ) -> Any:
        ...


def _native_snapshot_factory(**kwargs):
    from sglang.srt.disaggregation.agentic_hybrid_dma import open_request_snapshot

    return open_request_snapshot(create=False, **kwargs)


class SourceLocalHostArena:
    """Best-fit extent allocator over one source-rank memfd."""

    def __init__(
        self,
        *,
        direction: HostDirection,
        device_pool: Any,
        capacity_bytes: Optional[int] = None,
        snapshot_factory: SnapshotFactory = _native_snapshot_factory,
        preallocate: bool = True,
    ) -> None:
        self.direction = HostDirection(direction)
        self.device_pool = device_pool
        if capacity_bytes is None:
            if self.direction is HostDirection.D2P:
                env_name, default = "SGLANG_AGENTIC_MULTINODE_D2P_HOST_GIB", 32
            else:
                env_name, default = (
                    "SGLANG_AGENTIC_MULTINODE_P2D_HOST_GIB",
                    16,
                )
            capacity_bytes = int(float(os.getenv(env_name, str(default))) * _GIB)
        alignment = mmap.ALLOCATIONGRANULARITY
        self.capacity_bytes = int(capacity_bytes) // alignment * alignment
        if self.capacity_bytes <= 0:
            raise ValueError("Host arena capacity must be positive")
        create = getattr(os, "memfd_create", None)
        if create is None:
            raise RuntimeError("source-local Host staging requires memfd_create")
        self._fd = int(
            create(
                f"sglang-agentic-host-arena-{os.getpid()}",
                getattr(os, "MFD_CLOEXEC", 0x0001),
            )
        )
        try:
            os.ftruncate(self._fd, self.capacity_bytes)
            if preallocate:
                os.posix_fallocate(self._fd, 0, self.capacity_bytes)
        except BaseException:
            os.close(self._fd)
            self._fd = -1
            raise
        self.path = f"/proc/{os.getpid()}/fd/{self._fd}"
        self._snapshot_factory = snapshot_factory
        self._free = [(0, self.capacity_bytes)]
        self._extents: dict[int, SourceHostExtent] = {}
        self._by_snapshot: dict[str, int] = {}
        self._next_id = 1
        self._used = 0
        self._closed = False
        self._lock = threading.RLock()

    @staticmethod
    def _align(value: int) -> int:
        alignment = mmap.ALLOCATIONGRANULARITY
        return (int(value) + alignment - 1) // alignment * alignment

    def allocate(
        self,
        *,
        snapshot_id: str,
        token_count: int,
        state_slots: int = 1,
    ) -> SourceHostExtent:
        if not snapshot_id:
            raise ValueError("snapshot_id is required")
        byte_size = complete_snapshot_bytes(
            token_count, self.device_pool, state_slots
        )
        requested = self._align(byte_size)
        with self._lock:
            if self._closed:
                raise RuntimeError("Host arena is closed")
            if snapshot_id in self._by_snapshot:
                extent_id = self._by_snapshot[snapshot_id]
                if extent_id < 0:
                    raise RuntimeError("snapshot extent allocation is already in progress")
                return self._extents[extent_id]
            candidates = [
                (length, offset, index)
                for index, (offset, length) in enumerate(self._free)
                if length >= requested
            ]
            if not candidates:
                raise MemoryError("source-local Host arena is full")
            _, offset, index = min(candidates)
            free_offset, free_length = self._free.pop(index)
            if free_length > requested:
                self._free.append(
                    (free_offset + requested, free_length - requested)
                )
                self._free.sort()
            extent_id = self._next_id
            self._next_id += 1
            self._by_snapshot[snapshot_id] = -extent_id
        try:
            snapshot = self._snapshot_factory(
                path=self.path,
                token_count=int(token_count),
                device_pool=self.device_pool,
                byte_size=byte_size,
                file_offset=offset,
                state_slots=int(state_slots),
            )
        except BaseException:
            with self._lock:
                self._by_snapshot.pop(snapshot_id, None)
                self._insert_free(offset, requested)
            raise
        extent = SourceHostExtent(
            extent_id,
            str(snapshot_id),
            int(token_count),
            byte_size,
            requested,
            offset,
            int(state_slots),
            snapshot,
        )
        with self._lock:
            self._extents[extent_id] = extent
            self._by_snapshot[extent.snapshot_id] = extent_id
            self._used += requested
        return extent

    def _insert_free(self, offset: int, size: int) -> None:
        self._free.append((int(offset), int(size)))
        self._free.sort()
        merged = []
        for current_offset, current_size in self._free:
            if merged and merged[-1][0] + merged[-1][1] == current_offset:
                old_offset, old_size = merged[-1]
                merged[-1] = (old_offset, old_size + current_size)
            else:
                merged.append((current_offset, current_size))
        self._free = merged

    def get(self, extent_id: int) -> SourceHostExtent:
        with self._lock:
            extent = self._extents.get(int(extent_id))
            if extent is None:
                raise KeyError(f"unknown Host extent {extent_id}")
            return extent

    def begin_copy(self, extent_id: int) -> SourceHostExtent:
        with self._lock:
            extent = self.get(extent_id)
            if extent.phase is not HostExtentPhase.ALLOCATED:
                raise RuntimeError("Host extent is not copy-admissible")
            extent.phase = HostExtentPhase.COPYING
            return extent

    def copy_drained(self, extent_id: int) -> None:
        with self._lock:
            extent = self.get(extent_id)
            if extent.phase is HostExtentPhase.COPYING:
                extent.phase = HostExtentPhase.ALLOCATED

    def mark_durable(self, extent_id: int) -> SourceHostExtent:
        with self._lock:
            extent = self.get(extent_id)
            if extent.phase is not HostExtentPhase.COPYING:
                raise RuntimeError("Host durability requires an active copy")
            extent.phase = HostExtentPhase.DURABLE
            marker = getattr(extent.snapshot, "mark_populated", None)
            if marker is not None:
                marker()
            return extent

    def mark_exported(self, extent_id: int, shard: Any) -> None:
        with self._lock:
            extent = self.get(extent_id)
            if extent.phase is not HostExtentPhase.DURABLE:
                raise RuntimeError("only durable Host bytes may be exported")
            extent.exported_shard = shard
            extent.phase = HostExtentPhase.EXPORTED

    def release(self, extent_id: int) -> bool:
        with self._lock:
            extent = self.get(extent_id)
            if extent.phase is HostExtentPhase.COPYING:
                return False
            closer = getattr(extent.snapshot, "close", None)
            if closer is not None:
                closer(unlink=False)
            extent.phase = HostExtentPhase.RELEASED
            self._extents.pop(extent.extent_id)
            self._by_snapshot.pop(extent.snapshot_id, None)
            self._used -= extent.allocation_bytes
            self._insert_free(extent.offset, extent.allocation_bytes)
            return True

    def release_snapshot(self, snapshot_id: str) -> bool:
        """Release one complete request-generation extent by wire identity."""

        with self._lock:
            extent_id = self._by_snapshot.get(str(snapshot_id))
            if extent_id is None or extent_id < 0:
                return False
            return self.release(extent_id)

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self._used

    def close(self) -> None:
        with self._lock:
            if self._extents:
                raise RuntimeError("cannot close Host arena with live extents")
            if self._closed:
                return
            os.close(self._fd)
            self._fd = -1
            self._closed = True


@dataclass(frozen=True, slots=True)
class SourceHostStorePayload:
    extent_id: int
    source_indices: Any
    state_indices: Any = None
    source_indices_host: Any = None


class SourceHostCopyBackend(Protocol):
    def copy(
        self,
        extent: SourceHostExtent,
        payload: SourceHostStorePayload,
        cancel: threading.Event,
    ) -> None:
        ...


class CudaSourceHostCopyBackend:
    """Chunked D2H using existing pure layout/DMA helpers and real events."""

    def __init__(self, device_pool, *, chunk_tokens: int = 2048) -> None:
        self.device_pool = device_pool
        self.chunk_tokens = max(1, int(chunk_tokens))

    def copy(
        self,
        extent: SourceHostExtent,
        payload: SourceHostStorePayload,
        cancel: threading.Event,
    ) -> None:
        import torch
        from sglang.srt.disaggregation.agentic_host_staging import (
            H2DLaunchFence,
            LayerFirstD2HStaging,
            PinnedMHAHostBounce,
        )

        snapshot = extent.snapshot
        if payload.state_indices is not None:
            snapshot.set_state_indices(payload.state_indices)
        pool = getattr(self.device_pool, "full_kv_pool", self.device_pool)
        stream = torch.cuda.Stream(device=pool.device)
        staging = LayerFirstD2HStaging(pool, self.chunk_tokens)
        bounce = PinnedMHAHostBounce(pool, self.chunk_tokens)
        indices = payload.source_indices
        if len(indices) != extent.token_count:
            raise ValueError("source indices do not cover the complete snapshot")
        for start in range(0, extent.token_count, self.chunk_tokens):
            if cancel.is_set():
                raise DrainedSourceHostCopy("cancelled between D2H chunks")
            count = min(self.chunk_tokens, extent.token_count - start)
            chunk = indices[start : start + count]
            host_chunk = (
                None
                if payload.source_indices_host is None
                else payload.source_indices_host[start : start + count]
            )
            fence = H2DLaunchFence(event=torch.cuda.Event(enable_timing=True))
            try:
                event, _refs = snapshot.start_backup_range_from_device(
                    chunk,
                    destination_start=start,
                    stream=stream,
                    staging=staging,
                    host_bounce=bounce,
                    launch_fence=fence,
                    source_indices_host=host_chunk,
                )
                event.synchronize()
            except BaseException as error:
                if fence.submitted and (fence.unavailable or not fence.armed):
                    raise UnfencedSourceHostCopy(
                        "D2H launch has no usable CUDA fence"
                    ) from error
                if fence.submitted:
                    try:
                        fence.event.synchronize()
                    except BaseException as fence_error:
                        raise UnfencedSourceHostCopy(
                            "D2H failure could not be drained"
                        ) from fence_error
                raise DrainedSourceHostCopy(str(error)) from error
            snapshot.commit_backup_range_from_bounce(
                bounce, destination_start=start, token_count=count
            )


@dataclass(slots=True)
class _StoreHandle:
    future: Future
    cancel: threading.Event
    extent_id: int
    shard: Any = None


class SourceHostStoreExecutor(TransferExecutor):
    """Asynchronous source-HBM→local-DRAM operation for one path queue."""

    def __init__(
        self,
        arena: SourceLocalHostArena,
        remote_worker: Any,
        copy_backend: SourceHostCopyBackend,
        *,
        max_workers: int = 4,
    ) -> None:
        self.arena = arena
        self.remote_worker = remote_worker
        self.copy_backend = copy_backend
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix=f"agentic-{arena.direction.value}-host-store",
        )

    def submit(self, attempt: TransferAttempt, notify: Callable[[], None]):
        payload = attempt.payload
        if not isinstance(payload, SourceHostStorePayload):
            raise TypeError("Host store requires SourceHostStorePayload")
        cancel = threading.Event()

        def run():
            extent = self.arena.begin_copy(payload.extent_id)
            try:
                self.copy_backend.copy(extent, payload, cancel)
            except DrainedSourceHostCopy:
                self.arena.copy_drained(payload.extent_id)
                raise
            extent = self.arena.mark_durable(payload.extent_id)
            shard = self.remote_worker.export_snapshot(
                extent.snapshot_id, extent.snapshot
            )
            self.arena.mark_exported(payload.extent_id, shard)
            return shard

        future = self._pool.submit(run)
        handle = _StoreHandle(future, cancel, payload.extent_id)
        future.add_done_callback(lambda _future: notify())
        return handle

    def progress(self, handle: _StoreHandle) -> PhysicalProgress:
        if not handle.future.done():
            return PhysicalProgress(PhysicalState.INFLIGHT)
        if handle.future.cancelled():
            return PhysicalProgress(PhysicalState.CANCELLED, FenceKind.NOT_POSTED)
        try:
            handle.shard = handle.future.result()
        except UnfencedSourceHostCopy:
            raise
        except BaseException as error:
            cancelled = handle.cancel.is_set()
            return PhysicalProgress(
                PhysicalState.CANCELLED if cancelled else PhysicalState.FAILED,
                FenceKind.CANCEL_DRAINED if cancelled else FenceKind.ERROR_DRAINED,
                str(error),
            )
        shard = handle.shard
        if not hasattr(shard, "to_dict"):
            raise TypeError("Host export must provide a JSON HostShard descriptor")
        return PhysicalProgress(
            PhysicalState.SUCCEEDED,
            FenceKind.DMA_COMPLETE,
            result={"host_shard": shard.to_dict()},
        )

    def request_cancel(self, handle: _StoreHandle, notify: Callable[[], None]) -> None:
        handle.cancel.set()
        if handle.future.cancel():
            notify()

    def discard_durable_export(self, extent: SourceHostExtent) -> bool:
        """Remove an unclaimed export after a group abort, then free its extent."""

        if not self.remote_worker.discard_unclaimed_export(extent.snapshot_id):
            return False
        return self.arena.release(extent.extent_id)

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=False)


def make_source_host_store_handler(
    arena: SourceLocalHostArena,
    *,
    descriptor: Callable[[GroupCommand], SourceHostStorePayload],
    source_hbm_release: Callable[[GroupCommand, SourceHostExtent], None],
    discard_durable_export: Callable[[SourceHostExtent], bool],
) -> CallbackPathHandler:
    """Build the source-side handler for a HOST_STORE group attempt."""

    def prepare(command: GroupCommand) -> PreparedRankTransfer:
        values = command.payload.get("transfer", {})
        extent = arena.allocate(
            snapshot_id=command.key.snapshot_id,
            token_count=int(values["token_count"]),
            state_slots=int(values.get("state_slots", 1)),
        )
        try:
            local = descriptor(command)
            if int(local.extent_id) not in {0, extent.extent_id}:
                raise ValueError("descriptor references a different Host extent")
            local = SourceHostStorePayload(
                extent.extent_id,
                local.source_indices,
                local.state_indices,
                local.source_indices_host,
            )
            return PreparedRankTransfer(local)
        except BaseException:
            # PREPARE never posted DMA, so this extent has no external reader
            # and must not consume Host capacity after a descriptor failure.
            arena.release(extent.extent_id)
            raise

    def commit(
        command: GroupCommand,
        prepared: PreparedRankTransfer,
        _completion,
    ) -> None:
        payload = prepared.transfer_payload
        extent = arena.get(payload.extent_id)
        if extent.phase is not HostExtentPhase.EXPORTED:
            raise RuntimeError("source HBM cannot release before Host durability")
        source_hbm_release(command, extent)

    def abort(command, prepared, completion) -> None:
        del command
        payload = prepared.transfer_payload
        extent = arena.get(payload.extent_id)
        fence = None if completion is None else completion.fence
        safe = completion is None or fence in {
            FenceKind.NOT_POSTED,
            FenceKind.CANCEL_DRAINED,
            FenceKind.ERROR_DRAINED,
        }
        if not safe:
            # Another TP rank may fail after this rank reached durability.
            # Discard only an export that has never been claimed by a reader.
            if extent.phase in {HostExtentPhase.DURABLE, HostExtentPhase.EXPORTED}:
                if not discard_durable_export(extent):
                    return
                return
            return
        if extent.phase in {HostExtentPhase.DURABLE, HostExtentPhase.EXPORTED}:
            discard_durable_export(extent)
        elif extent.phase is not HostExtentPhase.COPYING:
            arena.release(extent.extent_id)

    # ``abort`` is invoked only by the group-wide ABORT_FINALIZE barrier.  At
    # that point every rank has proved its local copy drained, so an unclaimed
    # durable export and its complete extent can be reclaimed safely.  If an
    # export has somehow acquired a reader claim, discard returns false and we
    # deliberately retain the bytes (fail closed).
    return CallbackPathHandler(prepare, commit=commit, abort=abort)


@dataclass(frozen=True, slots=True)
class SourceHostStorePath:
    """Objects installed into the matching queue and rank handler maps."""

    path: TransferPath
    executor: SourceHostStoreExecutor
    handler: CallbackPathHandler


def make_source_host_store_path(
    arena: SourceLocalHostArena,
    remote_worker: Any,
    copy_backend: SourceHostCopyBackend,
    *,
    descriptor: Callable[[GroupCommand], SourceHostStorePayload],
    source_hbm_release: Callable[[GroupCommand, SourceHostExtent], None],
    max_workers: int = 4,
) -> SourceHostStorePath:
    """Create a correctly paired Host queue executor and lifecycle handler."""

    executor = SourceHostStoreExecutor(
        arena,
        remote_worker,
        copy_backend,
        max_workers=max_workers,
    )
    handler = make_source_host_store_handler(
        arena,
        descriptor=descriptor,
        source_hbm_release=source_hbm_release,
        discard_durable_export=executor.discard_durable_export,
    )
    path = (
        TransferPath.D2P_HOST
        if arena.direction is HostDirection.D2P
        else TransferPath.P2D_HOST
    )
    return SourceHostStorePath(path, executor, handler)
