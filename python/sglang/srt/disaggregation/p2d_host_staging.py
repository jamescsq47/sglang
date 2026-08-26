from __future__ import annotations

"""Asynchronous Prefill-HBM -> Host -> Decode-HBM staging.

This is the reverse direction of :mod:`agentic_host_staging`.  It is used only
after Prefill has completed and the late-binding router cannot immediately
find Decode capacity.  The control ledger and byte arena are deliberately
separate from D->P snapshots: pressure in one direction must never consume the
other direction's credit.

GPU allocation and release remain scheduler-owned.  Background workers only
copy immutable, scheduler-pinned pages and publish completion state.
"""

import logging
import mmap
import os
import queue
import threading
import time
from typing import Any, Optional

import torch

from sglang.srt.disaggregation.agentic_host_staging import (
    H2DLaunchFence,
    HostStageState,
    LayerFirstD2HStaging,
    P2D_RELEASE_HOST_OWNED,
    PinnedMHAHostBounce,
    SharedHostStagingLedger,
    SharedMHAHostSnapshot,
)
from sglang.srt.disaggregation.base import KVPoll

logger = logging.getLogger(__name__)

P2D_CUSTOM_SNAPSHOT_ID = "agentic_p2d_host_snapshot_id"
P2D_CUSTOM_PREFILL_DOMAIN = "agentic_p2d_prefill_domain"

# HOST_READY is a transient group-level notification, not a state that every
# TP process is guaranteed to observe.  Once D starts (or finishes) loading,
# the complete P->Host write is still durably committed.  Waiters must use this
# monotonic boundary instead of testing exact equality with HOST_READY.
_P2D_HOST_WRITE_COMMITTED_STATES = {
    HostStageState.HOST_READY.value,
    HostStageState.H2D_LOADING.value,
    HostStageState.CONSUMED.value,
}
_P2D_HOST_TERMINAL_FAILURE_STATES = {
    HostStageState.ABORTING.value,
    HostStageState.REJECTED.value,
    HostStageState.FAILED.value,
}


def _p2d_host_write_committed(entry: Optional[dict[str, Any]]) -> bool:
    return bool(
        entry is not None and entry.get("state") in _P2D_HOST_WRITE_COMMITTED_STATES
    )


def _raise_if_p2d_host_failed(
    snapshot_id: str, entry: Optional[dict[str, Any]]
) -> None:
    if entry is None:
        raise RuntimeError(f"P->D Host ledger entry disappeared for {snapshot_id}")
    state = entry.get("state")
    if state in _P2D_HOST_TERMINAL_FAILURE_STATES:
        reason = entry.get("reason", state)
        raise RuntimeError(
            f"P->D Host snapshot {snapshot_id} terminated in {state}: {reason}"
        )


def p2d_snapshot_id(bootstrap_room: int) -> str:
    """Return a collision-free id in the direction-specific ledger."""

    return f"p2d:{int(bootstrap_room)}"


def p2d_snapshot_from_req(req) -> Optional[str]:
    params = getattr(getattr(req, "sampling_params", None), "custom_params", None)
    if not isinstance(params, dict):
        return None
    value = params.get(P2D_CUSTOM_SNAPSHOT_ID)
    return None if value in (None, "") else str(value)


def _prefill_metadata(req) -> dict[str, Any]:
    """Serialize the small non-KV result normally carried by NIXL metadata."""

    if getattr(req, "return_logprob", False):
        raise ValueError("P->D Host staging V1 does not support return_logprob")
    if not req.output_ids:
        raise ValueError("Prefill result has no sampled output token")
    return {
        "output_id": int(req.output_ids[0]),
        "cached_tokens": int(getattr(req, "cached_tokens", 0) or 0),
        "cached_tokens_device": int(getattr(req, "cached_tokens_device", 0) or 0),
        "cached_tokens_host": int(getattr(req, "cached_tokens_host", 0) or 0),
        "cached_tokens_storage": int(getattr(req, "cached_tokens_storage", 0) or 0),
    }


class _RegisteredP2DHostSnapshot:
    """One logical snapshot view inside a process-lifetime registered arena."""

    def __init__(
        self,
        *,
        arena,
        offset: int,
        allocation_bytes: int,
        token_count: int,
        byte_size: int,
        device_pool,
    ):
        self.arena = arena
        self.path = arena.path
        self.offset = int(offset)
        self.allocation_bytes = int(allocation_bytes)
        self.token_count = int(token_count)
        self.byte_size = int(byte_size)
        self.device_pool = device_pool
        self.layer_num = int(device_pool.layer_num)
        self.head_num = int(device_pool.head_num)
        self.head_dim = int(device_pool.head_dim)
        self.v_head_dim = int(getattr(device_pool, "v_head_dim", self.head_dim))
        if self.v_head_dim != self.head_dim:
            raise ValueError("registered P->D arena requires equal K/V head dimensions")
        self.dtype = device_pool.store_dtype
        self.item_size = self.head_num * self.head_dim * self.dtype.itemsize
        expected = (
            2
            * self.token_count
            * self.layer_num
            * self.head_num
            * self.head_dim
            * self.dtype.itemsize
        )
        if expected != self.byte_size:
            raise ValueError("registered P->D snapshot byte size mismatch")
        raw = arena.raw[self.offset : self.offset + self.byte_size]
        self.kv_buffer = raw.view(self.dtype).view(
            2,
            self.layer_num,
            self.token_count,
            self.head_num,
            self.head_dim,
        )
        self._raw = raw
        self._closed = False
        self._populated = False

    @property
    def k_buffer(self):
        return self.kv_buffer[0]

    @property
    def v_buffer(self):
        return self.kv_buffer[1]

    def materialize(self):
        return self

    def start_backup_range_from_device(
        self,
        source_indices,
        *,
        destination_start: int,
        stream,
        staging,
        host_bounce=None,
        launch_fence: Optional[H2DLaunchFence] = None,
    ):
        """Gather KV and DMA it directly into the registered Shared Arena."""

        del host_bounce
        from sgl_kernel.kvcacheio import transfer_kv_all_layer

        destination_start = int(destination_start)
        if (
            destination_start < 0
            or destination_start + len(source_indices) > self.token_count
        ):
            raise ValueError("P->D chunk falls outside registered Host extent")
        original_source_indices = source_indices
        if launch_fence is None:
            launch_fence = H2DLaunchFence(event=torch.cuda.Event(enable_timing=True))
        event = launch_fence.event
        start_event = torch.cuda.Event(enable_timing=True)
        copy_refs = [source_indices, original_source_indices, staging, self]
        launch_fence.copy_refs = copy_refs
        try:
            with torch.cuda.stream(stream):
                launch_fence.submitted = True
                if not source_indices.is_cuda or source_indices.dtype != torch.int64:
                    source_indices = source_indices.to(
                        device=self.device_pool.device,
                        dtype=torch.int64,
                        non_blocking=True,
                    )
                    copy_refs.append(source_indices)
                start_event.record(stream)
                for start in range(0, len(source_indices), staging.token_capacity):
                    count = min(staging.token_capacity, len(source_indices) - start)
                    source_chunk = source_indices[start : start + count]
                    local_indices = staging.local_indices[:count]
                    transfer_kv_all_layer(
                        src_k_layers=self.device_pool.k_data_ptrs,
                        dst_k_layers=staging.k_data_ptrs,
                        src_v_layers=self.device_pool.v_data_ptrs,
                        dst_v_layers=staging.v_data_ptrs,
                        src_indices=source_chunk,
                        dst_indices=local_indices,
                        item_size=self.item_size,
                        num_layers=self.layer_num,
                        block_quota=8,
                        num_warps_per_block=32,
                    )
                    host_start = destination_start + start
                    host_end = host_start + count
                    for layer_id in range(self.layer_num):
                        self.k_buffer[layer_id, host_start:host_end].copy_(
                            staging.k_buffer[layer_id][:count], non_blocking=True
                        )
                        self.v_buffer[layer_id, host_start:host_end].copy_(
                            staging.v_buffer[layer_id][:count], non_blocking=True
                        )
                event.record(stream)
                launch_fence.armed = True
                source_indices.record_stream(stream)
                if bool(getattr(original_source_indices, "is_cuda", False)):
                    original_source_indices.record_stream(stream)
        except Exception:
            if launch_fence.submitted and not launch_fence.armed:
                try:
                    with torch.cuda.stream(stream):
                        event.record(stream)
                    launch_fence.armed = True
                except Exception:
                    launch_fence.unavailable = True
            raise
        self._last_d2h_start_event = start_event
        copy_refs.append(start_event)
        launch_fence.copy_refs = copy_refs
        return event, tuple(copy_refs)

    def commit_backup_range_from_bounce(self, *args, **kwargs) -> None:
        del args, kwargs

    def mark_populated(self) -> None:
        self._populated = True

    def close(self, *, unlink: bool = False) -> None:
        del unlink
        if self._closed:
            return
        self.kv_buffer = None
        self._raw = None
        self._closed = True


class _RegisteredP2DHostArena:
    """One pre-registered tmpfs arena with request-level suballocation.

    Registration happens once during P startup.  Snapshot D2H then lands
    directly in Shared Arena memory, avoiding the pageable bounce->memcpy path
    and all per-generation cudaHostRegister/unregister calls.
    """

    _ALIGNMENT = mmap.ALLOCATIONGRANULARITY

    def __init__(self, directory: str, capacity_bytes: int, device_pool):
        if not directory.startswith("/dev/shm/"):
            raise ValueError("registered P->D arena must reside in /dev/shm")
        self.directory = directory.rstrip("/")
        self.capacity_bytes = self._align_down(int(capacity_bytes))
        if self.capacity_bytes <= 0:
            raise ValueError("registered P->D arena capacity must be positive")
        self.device_pool = device_pool
        os.makedirs(self.directory, mode=0o700, exist_ok=True)
        self.path = os.path.join(self.directory, "registered-arena.kv")
        self.mapping = None
        self.raw = None
        self._registered = False
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            os.ftruncate(fd, self.capacity_bytes)
            self.mapping = mmap.mmap(
                fd, self.capacity_bytes, access=mmap.ACCESS_WRITE
            )
        finally:
            os.close(fd)
        try:
            self.raw = torch.frombuffer(
                self.mapping, dtype=torch.uint8, count=self.capacity_bytes
            )
            started_at = time.monotonic()
            result = torch.cuda.cudart().cudaHostRegister(
                self.raw.data_ptr(), self.capacity_bytes, 0
            )
            if result != torch.cuda.cudart().cudaError.success:
                raise RuntimeError(f"P->D arena cudaHostRegister failed: {result}")
            self._registered = True
            self.registration_seconds = time.monotonic() - started_at
        except BaseException:
            self.raw = None
            if self.mapping is not None:
                self.mapping.close()
                self.mapping = None
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass
            raise
        self.used_bytes = 0
        self._lock = threading.Lock()
        self._free: list[tuple[int, int]] = [(0, self.capacity_bytes)]
        self._active: dict[int, tuple[_RegisteredP2DHostSnapshot, int, int]] = {}
        self._closed = False

    @classmethod
    def _align_up(cls, value: int) -> int:
        return (int(value) + cls._ALIGNMENT - 1) // cls._ALIGNMENT * cls._ALIGNMENT

    @classmethod
    def _align_down(cls, value: int) -> int:
        return int(value) // cls._ALIGNMENT * cls._ALIGNMENT

    def can_reserve(self, byte_size: int, hard_watermark: float) -> bool:
        requested = self._align_up(byte_size)
        with self._lock:
            if self.used_bytes + requested > int(
                self.capacity_bytes * float(hard_watermark)
            ):
                return False
            return any(length >= requested for _, length in self._free)

    def create(self, snapshot_id: str, token_count: int, device_pool, byte_size: int):
        del snapshot_id
        requested = self._align_up(byte_size)
        with self._lock:
            candidates = [
                (length, offset, index)
                for index, (offset, length) in enumerate(self._free)
                if length >= requested
            ]
            if not candidates:
                raise RuntimeError("registered P->D arena has no contiguous capacity")
            _, offset, index = min(candidates)
            free_offset, free_length = self._free.pop(index)
            if free_length > requested:
                self._free.append(
                    (free_offset + requested, free_length - requested)
                )
                self._free.sort()
            try:
                snapshot = _RegisteredP2DHostSnapshot(
                    arena=self,
                    offset=offset,
                    allocation_bytes=requested,
                    token_count=token_count,
                    byte_size=byte_size,
                    device_pool=device_pool,
                )
            except BaseException:
                # Extent allocation and view construction are one transaction.
                # A bad layout must not silently remove capacity from the pool.
                self._insert_free_locked(offset, requested)
                raise
            self.used_bytes += requested
            self._active[id(snapshot)] = (snapshot, offset, requested)
            return snapshot

    def _insert_free_locked(self, offset: int, allocation_bytes: int) -> None:
        self._free.append((int(offset), int(allocation_bytes)))
        self._free.sort()
        merged: list[tuple[int, int]] = []
        for current_offset, current_length in self._free:
            if merged and merged[-1][0] + merged[-1][1] == current_offset:
                previous_offset, previous_length = merged[-1]
                merged[-1] = (
                    previous_offset,
                    previous_length + current_length,
                )
            else:
                merged.append((current_offset, current_length))
        self._free = merged

    def release(self, snapshot) -> None:
        with self._lock:
            active = self._active.pop(id(snapshot), None)
            if active is None or active[0] is not snapshot:
                return
            _, offset, allocation_bytes = active
            snapshot.close(unlink=False)
            self.used_bytes = max(0, self.used_bytes - allocation_bytes)
            self._insert_free_locked(offset, allocation_bytes)

    def usage(self) -> float:
        with self._lock:
            return self.used_bytes / max(1, self.capacity_bytes)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            for snapshot, _, _ in self._active.values():
                snapshot.close(unlink=False)
            self._active.clear()
            self._free = []
            self.used_bytes = 0
            if self._registered:
                result = torch.cuda.cudart().cudaHostUnregister(self.raw.data_ptr())
                if result != torch.cuda.cudart().cudaError.success:
                    raise RuntimeError(
                        f"P->D arena cudaHostUnregister failed: {result}"
                    )
                self._registered = False
            self.raw = None
            if self.mapping is not None:
                self.mapping.close()
                self.mapping = None
            self._closed = True


class AgenticPToDHostStagingManager:
    """P-side producer for complete request-generation Host snapshots."""

    def __init__(
        self,
        *,
        ledger: SharedHostStagingLedger,
        device_pool,
        page_size: int,
        arena_directory: str,
        arena_capacity_bytes: int,
        prefill_domain: int,
        numa_node: int,
        tp_rank: int = 0,
        tp_size: int = 1,
        hard_watermark: float = 0.90,
    ):
        if not (0.0 < hard_watermark <= 1.0):
            raise ValueError("P->D Host hard watermark must be in (0, 1]")
        self.ledger = ledger
        self.device_pool = device_pool
        self.page_size = int(page_size)
        self.prefill_domain = int(prefill_domain)
        self.numa_node = int(numa_node)
        self.tp_rank = int(tp_rank)
        self.tp_size = int(tp_size)
        self.hard_watermark = float(hard_watermark)
        self.owner = (
            f"p2d-p:{os.getpid()}"
            if self.tp_size == 1
            else f"p2d-p-group:{os.getenv('SGLANG_AGENTIC_KV_ENGINE_ID', 'prefill')}"
        )
        self.arena = _RegisteredP2DHostArena(
            arena_directory, int(arena_capacity_bytes), self.device_pool
        )
        self.chunk_tokens = max(
            self.page_size,
            int(os.getenv("SGLANG_AGENTIC_KV_P2D_D2H_CHUNK_TOKENS", "512")),
        )
        self.chunk_tokens = max(
            self.page_size,
            self.chunk_tokens // self.page_size * self.page_size,
        )
        self.worker_count = max(
            1, int(os.getenv("SGLANG_AGENTIC_KV_P2D_D2H_WORKERS", "4"))
        )
        self._worker_resources = [
            (
                torch.cuda.Stream(
                    device=torch.cuda.current_device(), priority=0
                ),
                LayerFirstD2HStaging(self.device_pool, self.chunk_tokens),
            )
            for _ in range(self.worker_count)
        ]
        self._work: queue.SimpleQueue = queue.SimpleQueue()
        self._lock = threading.RLock()
        self._active: dict[str, dict[str, Any]] = {}
        self._results: dict[str, int] = {}
        self._records: dict[str, dict[str, Any]] = {}
        self._group_pending: dict[str, dict[str, Any]] = {}
        self._group_wakeup = threading.Event()
        self._dma_quarantine: list[tuple[Any, ...]] = []
        # Prefill completion and D admission are deliberately decoupled.  The
        # scheduler registers an immutable page-index vector once; this worker
        # watches the request-level ledger and starts Host staging as soon as D
        # publishes an offer.  No scheduler iteration is needed to discover
        # that offer, so a long subsequent Prefill cannot pin completed KV in
        # P HBM merely because control progress is delayed.
        self._candidates: dict[str, dict[str, Any]] = {}
        self._candidate_wakeup = threading.Event()
        self._stop = threading.Event()
        self._offer_thread = threading.Thread(
            target=self._offer_worker,
            name=f"agentic-p2d-offer-{os.getpid()}",
            daemon=True,
        )
        self._threads = [
            threading.Thread(
                target=self._worker,
                args=(worker_id, *resources),
                name=f"agentic-p2d-spill-{os.getpid()}-{worker_id}",
                daemon=True,
            )
            for worker_id, resources in enumerate(self._worker_resources)
        ]
        self._completion_thread = threading.Thread(
            target=self._completion_worker,
            name=f"agentic-p2d-spill-completion-{os.getpid()}",
            daemon=True,
        )
        self._offer_thread.start()
        for thread in self._threads:
            thread.start()
        self._completion_thread.start()
        logger.info(
            "Agentic P->D Host staging enabled directory=%s capacity_gib=%.1f "
            "P=%d numa=%d chunk_tokens=%d workers=%d registered_arena=true "
            "registration_s=%.3f",
            self.arena.directory,
            self.arena.capacity_bytes / (1024**3),
            self.prefill_domain,
            self.numa_node,
            self.chunk_tokens,
            self.worker_count,
            self.arena.registration_seconds,
        )

    def _byte_size(self, token_count: int) -> int:
        return (
            2
            * int(token_count)
            * int(self.device_pool.layer_num)
            * int(self.device_pool.head_num)
            * int(self.device_pool.head_dim)
            * int(self.device_pool.store_dtype.itemsize)
        )

    def _targets_this_p(self, entry: dict[str, Any]) -> bool:
        return int(entry.get("prefill_domain", -1)) == self.prefill_domain

    def has_offer(self, req) -> bool:
        snapshot_id = p2d_snapshot_id(req.bootstrap_room)
        entry = self.ledger.get(snapshot_id)
        return bool(
            entry is not None
            and entry.get("state")
            in {HostStageState.OFFERED.value, HostStageState.HOST_RESERVED.value}
            and self._targets_this_p(entry)
        )

    def group_claimed(self, req) -> bool:
        """Return whether any TP rank committed this request to Host staging.

        The Router publishes one request-level offer.  Once one rank claims
        it, every peer must keep its Prefill result alive until it has joined
        the same snapshot; otherwise a rank-local native NIXL completion can
        split one logical TP request across two transfer paths.
        """

        if self.tp_size <= 1:
            return False
        entry = self.ledger.get(p2d_snapshot_id(req.bootstrap_room))
        return bool(
            entry is not None
            and self._targets_this_p(entry)
            and entry.get("p_owner") == self.owner
            and entry.get("state")
            in {
                HostStageState.HOST_RESERVED.value,
                HostStageState.HOST_WRITING.value,
                HostStageState.HOST_READY.value,
                HostStageState.H2D_LOADING.value,
            }
        )

    def watch(self, req, source_indices) -> bool:
        """Register completed P KV for scheduler-independent Host admission."""

        snapshot_id = p2d_snapshot_id(req.bootstrap_room)
        with self._lock:
            if (
                snapshot_id in self._active
                or snapshot_id in self._results
                or snapshot_id in self._candidates
                or getattr(req, "_agentic_p2d_host_terminal", False)
            ):
                return True
            source_ready_event = None
            if torch.is_tensor(source_indices) and source_indices.is_cuda:
                source_ready_event = torch.cuda.Event()
                source_ready_event.record(
                    torch.cuda.current_stream(device=source_indices.device)
                )
            self._candidates[snapshot_id] = {
                "req": req,
                "source_indices": source_indices,
                "source_ready_event": source_ready_event,
            }
        self._candidate_wakeup.set()
        return True

    def cancel_watch(self, req) -> bool:
        """Cancel before page release, or return False when Host owns KV."""

        snapshot_id = p2d_snapshot_id(req.bootstrap_room)
        with self._lock:
            local_snapshot = getattr(req, "_agentic_p2d_host_snapshot_id", None)
            if local_snapshot in self._active:
                return False
            if local_snapshot is None:
                ownership = self.ledger.arbitrate_p2d_release(
                    snapshot_id, tp_size=self.tp_size
                )
                if ownership == P2D_RELEASE_HOST_OWNED:
                    return False
            req._agentic_p2d_host_terminal = True
            self._candidates.pop(snapshot_id, None)
            if local_snapshot is not None:
                self._results.pop(local_snapshot, None)
            return True

    def try_submit(self, req, source_indices, source_ready_event=None) -> bool:
        """Atomically claim one Router offer and enqueue immutable D2H work.

        This method is called by the P scheduler.  It performs no Host copy and
        no CUDA synchronization.
        """

        snapshot_id = p2d_snapshot_id(req.bootstrap_room)
        entry = self.ledger.get(snapshot_id)
        if (
            entry is None
            or entry.get("state")
            not in {HostStageState.OFFERED.value, HostStageState.HOST_RESERVED.value}
            or not self._targets_this_p(entry)
        ):
            return False
        token_count = len(req.origin_input_ids)
        if token_count <= 0:
            # Normal NIXL remains the correctness fallback for shapes the V1
            # pageable arena cannot represent atomically.
            return False
        if len(source_indices) != token_count:
            raise RuntimeError("P->D staging source token count mismatch")
        byte_size = self._byte_size(token_count)
        snapshot = None
        claimed = False
        try:
            # ``source_indices`` is cloned by the scheduler on its current
            # CUDA stream.  The D2H worker uses a private stream, so retaining
            # the tensor alone is not a visibility dependency: under load the
            # gather could race the clone (and the preceding Prefill writes).
            # Record, but do not synchronize, at the scheduler boundary.  The
            # worker inserts the corresponding stream wait before touching KV.
            if (
                source_ready_event is None
                and torch.is_tensor(source_indices)
                and source_indices.is_cuda
            ):
                source_ready_event = torch.cuda.Event()
                source_ready_event.record(
                    torch.cuda.current_stream(device=source_indices.device)
                )
            prefill_metadata = _prefill_metadata(req)
            with self._lock:
                # Cancellation, capacity reservation, the ledger claim and
                # active registration form one ownership transaction.  The
                # scheduler cannot release these pages in the middle.
                if getattr(req, "_agentic_p2d_host_terminal", False):
                    return False
                if snapshot_id in self._active or snapshot_id in self._results:
                    return True
                if not self.arena.can_reserve(byte_size, self.hard_watermark):
                    return False
                # Reserve the physical extent before taking ledger ownership.
                # A worker can no longer fail after claim merely because a
                # burst consumed or fragmented the Shared Arena meanwhile.
                snapshot = self.arena.create(
                    snapshot_id, token_count, self.device_pool, byte_size
                )
                grant = {
                    "kind": "shared_host_extent",
                    "arena_path": snapshot.path,
                    "arena_offset": snapshot.offset,
                    "byte_size": int(byte_size),
                    "token_count": int(token_count),
                    "prefill_domain": self.prefill_domain,
                    "arena_numa_node": self.numa_node,
                    "prefill_metadata": prefill_metadata,
                    "tp_rank": self.tp_rank,
                }
                claim = self.ledger.claim_p2d_write_rank(
                    snapshot_id,
                    self.owner,
                    grant,
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                )
                if claim is None:
                    self.arena.release(snapshot)
                    snapshot = None
                    return False
                claimed = True
                record = {
                    "source_indices": source_indices,
                    "source_ready_event": source_ready_event,
                    "token_count": token_count,
                    "byte_size": byte_size,
                    "prefill_metadata": prefill_metadata,
                    "started_at": time.monotonic(),
                    "snapshot": snapshot,
                }
                self._active[snapshot_id] = record
                self._records[snapshot_id] = record
                req._agentic_p2d_host_snapshot_id = snapshot_id
            self._work.put((snapshot_id, record))
            logger.info(
                "AgenticKV p2d_host_d2h_queued snapshot=%s tokens=%d bytes=%d "
                "P=%d numa=%d",
                snapshot_id,
                token_count,
                byte_size,
                self.prefill_domain,
                self.numa_node,
            )
            return True
        except Exception as exc:
            with self._lock:
                record = self._active.pop(snapshot_id, None)
                self._records.pop(snapshot_id, None)
                owned_snapshot = (
                    record.get("snapshot") if record is not None else snapshot
                )
                if owned_snapshot is not None:
                    self.arena.release(owned_snapshot)
            # Before claim, failure leaves the Router offer intact so native
            # Direct remains a valid correctness path.  After claim, Host owns
            # all TP shards and the generation must fail closed.
            if claimed:
                self.ledger.transition(
                    snapshot_id,
                    HostStageState.FAILED,
                    owner=self.owner,
                    reason=f"p2d_submit_failed:{exc}",
                )
            logger.exception("Failed to submit P->D Host staging for %s", snapshot_id)
            return False

    def is_active(self, req) -> bool:
        if getattr(req, "_agentic_p2d_host_terminal", False):
            return True
        snapshot_id = getattr(req, "_agentic_p2d_host_snapshot_id", None)
        if snapshot_id is None:
            return False
        with self._lock:
            return snapshot_id in self._active or snapshot_id in self._results

    def poll(self, req) -> Optional[int]:
        snapshot_id = getattr(req, "_agentic_p2d_host_snapshot_id", None)
        if snapshot_id is None:
            return int(KVPoll.Transferring) if self.group_claimed(req) else None
        with self._lock:
            result = self._results.get(snapshot_id)
            active = snapshot_id in self._active
        if result == int(KVPoll.Success):
            return result
        if result == int(KVPoll.Failed):
            # Host ownership is exclusive once any TP rank claims the offer.
            # Reusing the already-prepared native sender after a Host failure
            # can resurrect stale NIXL metadata and split one logical request
            # across transfer modes.  Fail this generation closed; the caller
            # may retry it as a fresh request-generation.
            return result
        return int(KVPoll.Transferring) if active else None

    def prepare_scheduler_release(self, req) -> bool:
        """Atomically return final page ownership to the scheduler.

        The scheduler calls this before releasing GPU KV.  If Host staging
        won a race with native NIXL completion, the request stays inflight
        until D2H has finished.
        """

        candidate_id = p2d_snapshot_id(req.bootstrap_room)
        with self._lock:
            snapshot_id = getattr(req, "_agentic_p2d_host_snapshot_id", None)
            if snapshot_id is not None:
                if snapshot_id in self._active:
                    return False
                if self._results.get(snapshot_id) != int(KVPoll.Success):
                    return False
                self._results.pop(snapshot_id, None)
            elif candidate_id in self._active or candidate_id in self._results:
                return False
            else:
                ownership = self.ledger.arbitrate_p2d_release(
                    candidate_id, tp_size=self.tp_size
                )
                if ownership == P2D_RELEASE_HOST_OWNED:
                    # Another TP rank won live Host ownership after our last
                    # poll.  Keep this shard until its offer worker joins.  A
                    # failed/aborting peer is different: no local D2H exists,
                    # so this untouched shard may be released immediately.
                    return False
            req._agentic_p2d_host_terminal = True
            self._candidates.pop(candidate_id, None)
            return True

    def mark_scheduler_consumed(self, req) -> None:
        """Compatibility wrapper for callers that already saw success."""

        if not self.prepare_scheduler_release(req):
            raise RuntimeError("P->D Host pages are still owned by staging")

    def close(self) -> None:
        """Stop the background worker before its tmpfs control files vanish."""

        self._stop.set()
        self._candidate_wakeup.set()
        self._group_wakeup.set()
        for _ in self._threads:
            self._work.put(None)
        self._offer_thread.join(timeout=2.0)
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._completion_thread.join(timeout=2.0)
        background_threads = [
            self._offer_thread,
            *self._threads,
            self._completion_thread,
        ]
        if self._dma_quarantine:
            # A submitted CUDA copy without a usable completion fence has no
            # safe reuse or teardown boundary.  Keep the registered mapping,
            # snapshots and CUDA staging objects alive for process lifetime.
            logger.error(
                "P->D Host DMA quarantine is non-empty during shutdown; "
                "retaining registered arena and all physical copy ownership"
            )
        elif any(thread.is_alive() for thread in background_threads):
            logger.warning(
                "P->D Host background work still active during shutdown; "
                "retaining registered arena"
            )
        else:
            self.arena.close()

    def _offer_worker(self) -> None:
        """Claim new D Host offers without waiting for the P scheduler."""

        while not self._stop.is_set():
            self._candidate_wakeup.wait(timeout=0.02)
            self._candidate_wakeup.clear()
            with self._lock:
                candidates = list(self._candidates.items())
            entries = self.ledger.snapshot_entries() if candidates else {}
            for snapshot_id, candidate in candidates:
                if self._stop.is_set():
                    return
                req = candidate["req"]
                if getattr(req, "_agentic_p2d_host_terminal", False):
                    with self._lock:
                        self._candidates.pop(snapshot_id, None)
                    continue
                entry = entries.get(snapshot_id)
                if entry is None:
                    continue
                if entry.get("state") in {
                    HostStageState.CONSUMED.value,
                    HostStageState.FAILED.value,
                    HostStageState.REJECTED.value,
                }:
                    with self._lock:
                        self._candidates.pop(snapshot_id, None)
                    continue
                if entry.get("state") not in {
                    HostStageState.OFFERED.value,
                    HostStageState.HOST_RESERVED.value,
                } or not self._targets_this_p(entry):
                    continue
                # Serialize the final claim against scheduler cleanup.  The
                # lock is re-entrant because try_submit also protects active
                # bookkeeping; once cleanup marks the Req terminal this
                # candidate can never read pages that the scheduler released.
                with self._lock:
                    if self._candidates.get(snapshot_id) is not candidate or getattr(
                        req, "_agentic_p2d_host_terminal", False
                    ):
                        continue
                    if self.try_submit(
                        req,
                        candidate["source_indices"],
                        candidate.get("source_ready_event"),
                    ):
                        self._candidates.pop(snapshot_id, None)

    def _worker(
        self,
        worker_id: int,
        stream: torch.cuda.Stream,
        staging: LayerFirstD2HStaging,
    ) -> None:
        while not self._stop.is_set():
            try:
                work = self._work.get(timeout=0.1)
            except queue.Empty:
                self._cleanup_consumed()
                continue
            if work is None:
                break
            snapshot_id, record = work
            snapshot = record["snapshot"]
            launch_fence = None
            quarantined = False
            try:
                source_indices = record["source_indices"]
                source_ready_event = record.get("source_ready_event")
                if source_ready_event is not None:
                    stream.wait_event(source_ready_event)
                token_count = int(record["token_count"])
                for start in range(0, token_count, self.chunk_tokens):
                    end = min(start + self.chunk_tokens, token_count)
                    launch_fence = H2DLaunchFence(event=torch.cuda.Event())
                    event, _ = snapshot.start_backup_range_from_device(
                        source_indices[start:end],
                        destination_start=start,
                        stream=stream,
                        staging=staging,
                        launch_fence=launch_fence,
                    )
                    event.synchronize()
                    launch_fence = None
                # Only a completely written extent may enter the reusable
                # arena pool.  A failed partial D2H is unlinked on release.
                snapshot.mark_populated()
                if not self.ledger.complete_p2d_host_write_rank(
                    snapshot_id,
                    self.owner,
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                ):
                    raise RuntimeError("P->D HOST_READY publication was rejected")
                record["worker_id"] = worker_id
                current = self.ledger.get(snapshot_id)
                if _p2d_host_write_committed(current):
                    self._finish_d2h_success(snapshot_id, record)
                else:
                    _raise_if_p2d_host_failed(snapshot_id, current)
                    # A DMA lane is local to one TP rank.  Once its rank ACK
                    # is durable it must be free to copy another snapshot;
                    # waiting for peer ranks here can deadlock differently
                    # ordered TP queues.  The independent poller owns only the
                    # group-level completion barrier.
                    with self._lock:
                        self._group_pending[snapshot_id] = record
                    self._group_wakeup.set()
            except Exception as exc:
                if launch_fence is not None and launch_fence.submitted:
                    physically_quiesced = False
                    if launch_fence.armed and not launch_fence.unavailable:
                        try:
                            launch_fence.event.synchronize()
                            physically_quiesced = True
                        except Exception:
                            pass
                    if not physically_quiesced:
                        # Do not publish Failed: the scheduler would recycle
                        # source pages that an unfenced D2H may still read.
                        with self._lock:
                            self._dma_quarantine.append(
                                (
                                    snapshot,
                                    launch_fence,
                                    record,
                                    stream,
                                    staging,
                                )
                            )
                        quarantined = True
                        logger.exception(
                            "P->D Host D2H lost its completion fence for %s; "
                            "quarantining the lane and source pages",
                            snapshot_id,
                        )
                        return
                self.ledger.transition(
                    snapshot_id,
                    HostStageState.FAILED,
                    owner=self.owner,
                    reason=f"p2d_d2h_failed:{exc}",
                )
                with self._lock:
                    self._active.pop(snapshot_id, None)
                    self._results[snapshot_id] = int(KVPoll.Failed)
                logger.exception("P->D Host D2H failed for %s", snapshot_id)
            finally:
                if not quarantined:
                    record["source_indices"] = None
                    record["source_ready_event"] = None
            self._cleanup_consumed()

    def _finish_d2h_success(
        self, snapshot_id: str, record: dict[str, Any]
    ) -> None:
        elapsed = time.monotonic() - float(record["started_at"])
        with self._lock:
            self._group_pending.pop(snapshot_id, None)
            self._active.pop(snapshot_id, None)
            self._results[snapshot_id] = int(KVPoll.Success)
        logger.info(
            "AgenticKV p2d_host_d2h_complete snapshot=%s tokens=%d "
            "elapsed_ms=%.3f gib_per_s=%.3f worker=%d",
            snapshot_id,
            int(record["token_count"]),
            elapsed * 1000.0,
            int(record["byte_size"]) / max(elapsed, 1e-9) / (1024**3),
            int(record.get("worker_id", -1)),
        )

    def _progress_group_completions_once(self) -> int:
        with self._lock:
            pending = list(self._group_pending.items())
        progressed = 0
        for snapshot_id, record in pending:
            current = self.ledger.get(snapshot_id)
            if _p2d_host_write_committed(current):
                self._finish_d2h_success(snapshot_id, record)
                progressed += 1
                continue
            try:
                _raise_if_p2d_host_failed(snapshot_id, current)
            except Exception as exc:
                with self._lock:
                    self._group_pending.pop(snapshot_id, None)
                    self._active.pop(snapshot_id, None)
                    self._results[snapshot_id] = int(KVPoll.Failed)
                logger.error("P->D Host TP write failed for %s: %s", snapshot_id, exc)
                progressed += 1
        return progressed

    def _completion_worker(self) -> None:
        while not self._stop.is_set():
            self._group_wakeup.wait(timeout=0.01)
            self._group_wakeup.clear()
            self._progress_group_completions_once()
            self._cleanup_consumed()

    def _cleanup_consumed(self) -> None:
        with self._lock:
            records = list(self._records.items())
        for snapshot_id, record in records:
            entry = self.ledger.get(snapshot_id)
            if entry is None or entry.get("state") not in {
                HostStageState.CONSUMED.value,
                HostStageState.FAILED.value,
                HostStageState.REJECTED.value,
            }:
                continue
            # Never unlink while this process still has an active D2H event.
            with self._lock:
                if snapshot_id in self._active:
                    continue
                current = self._records.pop(snapshot_id, None)
            if current is not None:
                self.arena.release(current["snapshot"])
                logger.info(
                    "AgenticKV p2d_host_release snapshot=%s state=%s",
                    snapshot_id,
                    entry.get("state"),
                )


class AgenticPToDHostLoadManager:
    """D-side multi-lane, asynchronous Host->HBM loader."""

    def __init__(
        self,
        *,
        ledger: SharedHostStagingLedger,
        device_pool,
        page_size: int,
        decode_domain: int,
        numa_node: int,
        tp_rank: int = 0,
        tp_size: int = 1,
    ):
        self.ledger = ledger
        self.device_pool = device_pool
        self.page_size = int(page_size)
        self.decode_domain = int(decode_domain)
        self.numa_node = int(numa_node)
        self.tp_rank = int(tp_rank)
        self.tp_size = int(tp_size)
        self.chunk_tokens = max(
            self.page_size,
            int(os.getenv("SGLANG_AGENTIC_KV_P2D_H2D_CHUNK_TOKENS", "1024")),
        )
        self.chunk_tokens = max(
            self.page_size,
            self.chunk_tokens // self.page_size * self.page_size,
        )
        self.worker_count = max(
            1, int(os.getenv("SGLANG_AGENTIC_KV_P2D_H2D_WORKERS", "4"))
        )
        self._worker_resources = [
            (
                torch.cuda.Stream(
                    device=torch.cuda.current_device(), priority=0
                ),
                LayerFirstD2HStaging(self.device_pool, self.chunk_tokens),
                PinnedMHAHostBounce(self.device_pool, self.chunk_tokens),
            )
            for _ in range(self.worker_count)
        ]
        self._work: queue.SimpleQueue = queue.SimpleQueue()
        # Entries are retained only when CUDA accepted part of a copy but no
        # completion fence could be established.  They are process-lifetime
        # quarantine, not a retry queue.
        self._dma_quarantine: list[tuple[Any, ...]] = []
        self._dma_poisoned = False
        self._completion_lock = threading.RLock()
        self._group_pending: dict[str, dict[str, Any]] = {}
        self._group_wakeup = threading.Event()
        self._stop = threading.Event()
        self._threads = [
            threading.Thread(
                target=self._worker,
                args=(worker_id, *resources),
                name=f"agentic-p2d-load-{os.getpid()}-{worker_id}",
                daemon=True,
            )
            for worker_id, resources in enumerate(self._worker_resources)
        ]
        for thread in self._threads:
            thread.start()
        self._completion_thread = threading.Thread(
            target=self._completion_worker,
            name=f"agentic-p2d-load-completion-{os.getpid()}",
            daemon=True,
        )
        self._completion_thread.start()
        logger.info(
            "Agentic P->D Host load enabled D_domain=%d numa=%d "
            "chunk_tokens=%d workers=%d",
            self.decode_domain,
            self.numa_node,
            self.chunk_tokens,
            self.worker_count,
        )

    def close(self) -> None:
        self._stop.set()
        self._group_wakeup.set()
        for _ in self._threads:
            self._work.put(None)
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._completion_thread.join(timeout=2.0)

    def submit(self, receiver: "AgenticPToDHostReceiver", device_indices) -> None:
        with receiver._state_lock:
            if receiver._submitted:
                return
            if receiver._abort_pending:
                receiver._terminal = True
                receiver._poll = int(KVPoll.Failed)
                return
            entry = self.ledger.get(receiver.snapshot_id)
            if entry is None or entry.get("state") not in {
                HostStageState.HOST_READY.value,
                HostStageState.H2D_LOADING.value,
            }:
                raise RuntimeError("P->D Host snapshot is not ready")
            grants = entry.get("grants", [])
            matching = [
                grant
                for grant in grants
                if grant.get("kind") == "shared_host_extent"
                and int(grant.get("tp_rank", 0)) == int(getattr(self, "tp_rank", 0))
            ]
            if len(matching) != 1:
                raise RuntimeError(
                    f"P->D Host snapshot has no TP rank "
                    f"{getattr(self, 'tp_rank', 0)} extent"
                )
            grant = matching[0]
            if int(grant.get("arena_numa_node", -1)) != self.numa_node:
                raise RuntimeError(
                    "P->D slow path crossed NUMA: "
                    f"arena={grant.get('arena_numa_node')} D={self.numa_node}"
                )
            if int(grant["token_count"]) != len(device_indices):
                raise RuntimeError("P->D Host destination token count mismatch")
            owner = entry.get("p_owner")
            if not self.ledger.begin_host_load_rank(
                receiver.snapshot_id,
                owner,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
            ):
                raise RuntimeError("P->D H2D ownership transition was rejected")
            receiver._submitted = True
            receiver._poll = int(KVPoll.Transferring)
            receiver._grant = grant
            receiver._owner = owner
            self._work.put((receiver, device_indices))

    def _worker(
        self,
        worker_id: int,
        stream: torch.cuda.Stream,
        staging: LayerFirstD2HStaging,
        bounce: PinnedMHAHostBounce,
    ) -> None:
        while not self._stop.is_set():
            work = self._work.get()
            if work is None:
                break
            receiver, device_indices = work
            started_at = time.monotonic()
            snapshot = None
            launch_fence = None
            try:
                if receiver.abort_pending:
                    raise RuntimeError("P->D Host load aborted before H2D")
                grant = receiver._grant
                snapshot = SharedMHAHostSnapshot(
                    path=str(grant["arena_path"]),
                    token_count=int(grant["token_count"]),
                    device_pool=self.device_pool,
                    byte_size=int(grant["byte_size"]),
                    create=False,
                    file_offset=int(grant.get("arena_offset", 0)),
                )
                for start in range(0, len(device_indices), self.chunk_tokens):
                    end = min(start + self.chunk_tokens, len(device_indices))
                    launch_fence = H2DLaunchFence(
                        event=torch.cuda.Event(enable_timing=True)
                    )
                    event, _ = snapshot.start_load_range_to_device(
                        device_indices[start:end],
                        stream,
                        source_start=start,
                        staging=staging,
                        host_bounce=bounce,
                        launch_fence=launch_fence,
                    )
                    event.synchronize()
                    launch_fence = None
                    if receiver.abort_pending:
                        raise RuntimeError("P->D Host load aborted after H2D fence")
                if not self.ledger.complete_host_load_rank(
                    receiver.snapshot_id,
                    receiver._owner,
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                ):
                    raise RuntimeError("P->D CONSUMED publication was rejected")
                completion = {
                    "receiver": receiver,
                    "started_at": started_at,
                    "token_count": len(device_indices),
                    "byte_size": int(grant["byte_size"]),
                    "worker_id": worker_id,
                }
                current = self.ledger.get(receiver.snapshot_id)
                if (
                    current is not None
                    and current.get("state") == HostStageState.CONSUMED.value
                ):
                    self._finish_h2d_success(completion)
                else:
                    _raise_if_p2d_host_failed(receiver.snapshot_id, current)
                    with self._completion_lock:
                        self._group_pending[receiver.snapshot_id] = completion
                    self._group_wakeup.set()
            except Exception as exc:
                if launch_fence is not None and launch_fence.submitted:
                    physically_quiesced = False
                    if launch_fence.armed and not launch_fence.unavailable:
                        try:
                            launch_fence.event.synchronize()
                            physically_quiesced = True
                        except Exception:
                            pass
                    if not physically_quiesced:
                        # Never publish Failed: Decode would recycle the target
                        # pages while an unfenced H2D may still write them.
                        self._dma_quarantine.append(
                            (snapshot, launch_fence, device_indices, receiver)
                        )
                        self._dma_poisoned = True
                        snapshot = None
                        receiver.mark_quarantined(exc)
                        logger.exception(
                            "P->D Host H2D lost its completion fence for %s; "
                            "quarantining source and destination",
                            receiver.snapshot_id,
                        )
                        # Staging and bounce buffers are shared by this
                        # serialized worker.  With no fence they cannot be
                        # reused safely for any later request.
                        return
                try:
                    self.ledger.transition(
                        receiver.snapshot_id,
                        HostStageState.FAILED,
                        owner=receiver._owner,
                        reason=f"p2d_h2d_failed:{exc}",
                    )
                except Exception:
                    logger.exception("Failed to publish P->D H2D failure")
                receiver.mark_terminal(KVPoll.Failed, error=exc)
                logger.exception("P->D Host H2D failed for %s", receiver.snapshot_id)
            finally:
                if snapshot is not None:
                    snapshot.close(unlink=False)

    def _finish_h2d_success(self, completion: dict[str, Any]) -> None:
        receiver = completion["receiver"]
        with self._completion_lock:
            self._group_pending.pop(receiver.snapshot_id, None)
        receiver.mark_terminal(KVPoll.Success)
        elapsed = time.monotonic() - float(completion["started_at"])
        logger.info(
            "AgenticKV p2d_host_h2d_complete snapshot=%s tokens=%d "
            "elapsed_ms=%.3f gib_per_s=%.3f D_domain=%d numa=%d worker=%d",
            receiver.snapshot_id,
            int(completion["token_count"]),
            elapsed * 1000.0,
            int(completion["byte_size"]) / max(elapsed, 1e-9) / (1024**3),
            self.decode_domain,
            self.numa_node,
            int(completion["worker_id"]),
        )

    def _progress_group_completions_once(self) -> int:
        with self._completion_lock:
            pending = list(self._group_pending.items())
        progressed = 0
        for snapshot_id, completion in pending:
            current = self.ledger.get(snapshot_id)
            if (
                current is not None
                and current.get("state") == HostStageState.CONSUMED.value
            ):
                self._finish_h2d_success(completion)
                progressed += 1
                continue
            try:
                _raise_if_p2d_host_failed(snapshot_id, current)
            except Exception as exc:
                with self._completion_lock:
                    self._group_pending.pop(snapshot_id, None)
                completion["receiver"].mark_terminal(KVPoll.Failed, error=exc)
                logger.error("P->D Host TP load failed for %s: %s", snapshot_id, exc)
                progressed += 1
        return progressed

    def _completion_worker(self) -> None:
        while not self._stop.is_set():
            self._group_wakeup.wait(timeout=0.01)
            self._group_wakeup.clear()
            self._progress_group_completions_once()


class AgenticPToDHostReceiver:
    """Small receiver adapter consumed by DecodeTransferQueue."""

    require_staging = False

    def __init__(self, manager: AgenticPToDHostLoadManager, snapshot_id: str):
        self.manager = manager
        self.snapshot_id = str(snapshot_id)
        self._poll = int(KVPoll.WaitingForInput)
        self._submitted = False
        self._grant: Optional[dict[str, Any]] = None
        self._owner: Optional[str] = None
        self._error: Optional[BaseException] = None
        self._abort_pending = False
        self._terminal = False
        self._quarantined = False
        self._state_lock = threading.RLock()

    @property
    def abort_pending(self) -> bool:
        with self._state_lock:
            return self._abort_pending

    def mark_terminal(
        self, poll: KVPoll, *, error: Optional[BaseException] = None
    ) -> None:
        with self._state_lock:
            self._terminal = True
            if error is not None:
                self._error = error
            if self._abort_pending and poll == KVPoll.Success:
                self._error = RuntimeError("P->D Host load was aborted")
                self._poll = int(KVPoll.Failed)
            else:
                self._poll = int(poll)

    def mark_quarantined(self, error: BaseException) -> None:
        with self._state_lock:
            self._quarantined = True
            self._error = error
            # WaitingForInput is deliberate: DecodeTransferQueue must retain
            # its target pages because no physical completion fence exists.
            self._poll = int(KVPoll.WaitingForInput)

    def init(self, _prefill_dp_rank: int) -> None:
        return

    def bind(self, device_indices) -> None:
        self.manager.submit(self, device_indices)

    def poll(self):
        with self._state_lock:
            return self._poll

    def failure_exception(self):
        if self._error is not None:
            raise self._error

    def clear(self) -> None:
        return

    def abort(self) -> None:
        with self._state_lock:
            self._abort_pending = True
            if not self._submitted or self._terminal:
                self._terminal = True
                self._error = RuntimeError("P->D Host load was aborted")
                self._poll = int(KVPoll.Failed)

    def commit_req(self, req) -> None:
        if self._grant is None:
            raise RuntimeError("P->D Host metadata is missing")
        metadata = self._grant.get("prefill_metadata") or {}
        req.output_ids.append(int(metadata["output_id"]))
        req.cached_tokens = int(metadata.get("cached_tokens", 0))
        req.cached_tokens_device = int(metadata.get("cached_tokens_device", 0))
        req.cached_tokens_host = int(metadata.get("cached_tokens_host", 0))
        req.cached_tokens_storage = int(metadata.get("cached_tokens_storage", 0))
