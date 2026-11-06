"""Complete request-generation Host snapshots for hybrid attention/Mamba models.

The existing agentic shared arena stores attention KV in a request-sized
extent.  Qwen3.5 additionally needs one temporal/conv state slot.  This module
keeps both payloads in one manifest-owned extent while retaining the physical
page-oriented layout used by the arena.
"""

from __future__ import annotations

import mmap
import os
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.disaggregation.agentic_host_staging import (
    SharedMHAHostSnapshot, _validate_shared_host_backing_path, _open_shared_host_backing,
)


def _align_up(value: int, alignment: int = mmap.ALLOCATIONGRANULARITY) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class HybridSnapshotLayout:
    attention_bytes: int
    state_offset: int
    state_bytes: int
    total_bytes: int

    @classmethod
    def from_pools(
        cls, token_count: int, kv_pool, mamba_pool, *, state_slots: int = 1
    ) -> "HybridSnapshotLayout":
        attention_bytes = (
            2
            * int(token_count)
            * int(kv_pool.layer_num)
            * int(kv_pool.head_num)
            * int(kv_pool.head_dim)
            * kv_pool.store_dtype.itemsize
        )
        state_slots = int(state_slots)
        if state_slots <= 0:
            raise ValueError("hybrid snapshot must contain at least one state slot")
        state_bytes = 0
        for tensor in mamba_pool.mamba_cache.conv:
            state_bytes += (
                tensor.shape[0]
                * state_slots
                * int(torch.tensor(tensor.shape[2:]).prod().item())
                * tensor.element_size()
            )
        temporal = mamba_pool.mamba_cache.temporal
        state_bytes += (
            temporal.shape[0]
            * state_slots
            * int(torch.tensor(temporal.shape[2:]).prod().item())
            * temporal.element_size()
        )
        state_offset = _align_up(attention_bytes)
        return cls(
            attention_bytes=attention_bytes,
            state_offset=state_offset,
            state_bytes=state_bytes,
            total_bytes=state_offset + state_bytes,
        )


class SharedMambaHostSnapshot:
    """One Mamba temporal/conv slot stored in a shared-memory sub-extent."""

    def __init__(
        self,
        *,
        path: str,
        mamba_pool,
        byte_size: int,
        file_offset: int,
        state_slots: int = 1,
    ):
        _validate_shared_host_backing_path(path)
        if file_offset % mmap.ALLOCATIONGRANULARITY:
            raise ValueError("Mamba snapshot offset must be mmap-aligned")
        self.path = path
        self.mamba_pool = mamba_pool
        self.byte_size = int(byte_size)
        self.file_offset = int(file_offset)
        self.state_slots = int(state_slots)
        if self.state_slots <= 0:
            raise ValueError("Mamba snapshot must contain at least one state slot")
        fd = _open_shared_host_backing(path, os.O_RDWR)
        try:
            if os.fstat(fd).st_size < self.file_offset + self.byte_size:
                raise ValueError("shared extent is smaller than Mamba payload")
            self.mapping = mmap.mmap(
                fd,
                self.byte_size,
                access=mmap.ACCESS_WRITE,
                offset=self.file_offset,
            )
        finally:
            os.close(fd)

        raw = torch.frombuffer(self.mapping, dtype=torch.uint8, count=self.byte_size)
        cursor = 0
        self.conv = []
        for tensor in mamba_pool.mamba_cache.conv:
            shape = (tensor.shape[0], self.state_slots, *tensor.shape[2:])
            count = int(torch.tensor(shape).prod().item())
            nbytes = count * tensor.element_size()
            view = raw[cursor : cursor + nbytes].view(tensor.dtype).view(shape)
            self.conv.append(view)
            cursor += nbytes
        temporal = mamba_pool.mamba_cache.temporal
        shape = (temporal.shape[0], self.state_slots, *temporal.shape[2:])
        count = int(torch.tensor(shape).prod().item())
        nbytes = count * temporal.element_size()
        self.temporal = raw[cursor : cursor + nbytes].view(temporal.dtype).view(shape)
        cursor += nbytes
        if cursor != self.byte_size:
            raise ValueError(
                f"Mamba layout mismatch expected={self.byte_size} consumed={cursor}"
            )
        self._raw = raw
        self._closed = False

    def _indices(self, value) -> torch.Tensor:
        indices = torch.as_tensor(value, dtype=torch.int64).reshape(-1)
        if indices.numel() != self.state_slots:
            raise ValueError(
                "Mamba state slot count mismatch: "
                f"snapshot={self.state_slots} indices={indices.numel()}"
            )
        return indices

    def backup_from_device(self, source_indices) -> None:
        """CPU-only compatibility helper used by layout unit tests."""

        indices = self._indices(source_indices)
        if self.mamba_pool.mamba_cache.temporal.is_cuda:
            raise RuntimeError("CUDA state backup must use the asynchronous API")
        conv_cpu, temporal_cpu = self.mamba_pool.get_cpu_copy(indices)
        for destination, source in zip(self.conv, conv_cpu):
            destination.copy_(source)
        self.temporal.copy_(temporal_cpu)

    def load_to_device(self, destination_indices) -> None:
        """CPU-only compatibility helper used by layout unit tests."""

        indices = self._indices(destination_indices)
        if self.mamba_pool.mamba_cache.temporal.is_cuda:
            raise RuntimeError("CUDA state load must use the asynchronous API")
        self.mamba_pool.load_cpu_copy((self.conv, self.temporal), indices)

    def start_backup_from_device(self, source_indices, stream, launch_fence=None):
        """Launch state D2H on ``stream`` without a device-wide fence.

        The mmap view is pageable in the reverse Shared-Arena path.  A pinned
        bounce makes the CUDA operation genuinely asynchronous; the progress
        worker commits it to the manifest-owned mmap only after the event.
        """

        host_indices = self._indices(source_indices)
        conv_bounce = [
            torch.empty_like(destination, device="cpu", pin_memory=True)
            for destination in self.conv
        ]
        temporal_bounce = torch.empty_like(self.temporal, device="cpu", pin_memory=True)
        event = (
            torch.cuda.Event(enable_timing=True)
            if launch_fence is None
            else launch_fence.event
        )
        start_event = torch.cuda.Event(enable_timing=True)
        gathered = []
        refs = [host_indices, conv_bounce, temporal_bounce, start_event, self, gathered]
        if launch_fence is not None:
            launch_fence.copy_refs = refs
        try:
            with torch.cuda.stream(stream):
                if launch_fence is not None:
                    launch_fence.submitted = True
                indices = host_indices.to(device=self.mamba_pool.mamba_cache.temporal.device, non_blocking=True)
                refs[0] = indices
                start_event.record(stream)
                for destination, source in zip(
                    conv_bounce, self.mamba_pool.mamba_cache.conv
                ):
                    selected = source[:, indices]
                    gathered.append(selected)
                    destination.copy_(selected, non_blocking=True)
                selected = self.mamba_pool.mamba_cache.temporal[:, indices]
                gathered.append(selected)
                temporal_bounce.copy_(selected, non_blocking=True)
                event.record(stream)
                if launch_fence is not None:
                    launch_fence.armed = True
                if indices.is_cuda:
                    indices.record_stream(stream)
        except Exception:
            if launch_fence is not None and launch_fence.submitted:
                try:
                    with torch.cuda.stream(stream):
                        event.record(stream)
                    launch_fence.armed = True
                except Exception:
                    launch_fence.unavailable = True
            raise
        self._last_d2h_state_start_event = start_event
        return event, refs

    def commit_backup_from_bounce(self, refs) -> None:
        _, conv_bounce, temporal_bounce, _, _ = refs[:5]
        for destination, source in zip(self.conv, conv_bounce):
            destination.copy_(source)
        self.temporal.copy_(temporal_bounce)

    def start_load_to_device(self, destination_indices, stream, launch_fence=None):
        """Launch state H2D on ``stream`` without stalling Decode/Prefill."""

        host_indices = self._indices(destination_indices)
        # H2D uses scalar destination slices, so no device index tensor is needed.
        indices = host_indices
        slot_indices = host_indices.detach().cpu().tolist()
        # Copy mmap -> pinned bounce on CPU before the asynchronous H2D.  This
        # does not synchronize any CUDA stream and keeps the mmap alive in refs.
        conv_bounce = [
            torch.empty_like(source, device="cpu", pin_memory=True)
            for source in self.conv
        ]
        temporal_bounce = torch.empty_like(self.temporal, device="cpu", pin_memory=True)
        for destination, source in zip(conv_bounce, self.conv):
            destination.copy_(source)
        temporal_bounce.copy_(self.temporal)
        event = (
            torch.cuda.Event(enable_timing=True)
            if launch_fence is None
            else launch_fence.event
        )
        start_event = torch.cuda.Event(enable_timing=True)
        refs = (indices, conv_bounce, temporal_bounce, start_event, self)
        if launch_fence is not None:
            launch_fence.copy_refs = refs
        try:
            with torch.cuda.stream(stream):
                if launch_fence is not None:
                    launch_fence.submitted = True
                start_event.record(stream)
                for destination, source in zip(
                    self.mamba_pool.mamba_cache.conv, conv_bounce
                ):
                    # CUDA index_copy_ requires a CUDA source and therefore
                    # cannot ingest the pinned Host bounce directly.  The
                    # state snapshot is normally one slot (and deliberately
                    # remains a tiny fixed number for future layouts), so use
                    # destination views to issue direct asynchronous H2D
                    # copies without allocating a second device-sized staging
                    # tensor.
                    for source_slot, destination_slot in enumerate(slot_indices):
                        destination[
                            :, destination_slot : destination_slot + 1
                        ].copy_(
                            source[:, source_slot : source_slot + 1],
                            non_blocking=True,
                        )
                for source_slot, destination_slot in enumerate(slot_indices):
                    self.mamba_pool.mamba_cache.temporal[
                        :, destination_slot : destination_slot + 1
                    ].copy_(
                        temporal_bounce[:, source_slot : source_slot + 1],
                        non_blocking=True,
                    )
                event.record(stream)
                if launch_fence is not None:
                    launch_fence.armed = True
                if indices.is_cuda:
                    indices.record_stream(stream)
        except Exception:
            if launch_fence is not None and launch_fence.submitted:
                try:
                    with torch.cuda.stream(stream):
                        event.record(stream)
                    launch_fence.armed = True
                except Exception:
                    launch_fence.unavailable = True
            raise
        refs = (indices, conv_bounce, temporal_bounce, start_event, self)
        if launch_fence is not None:
            launch_fence.copy_refs = refs
        self._last_h2d_state_start_event = start_event
        return event, refs

    def close(self) -> None:
        if self._closed:
            return
        self.conv = []
        self.temporal = None
        self._raw = None
        self.mapping.close()
        self._closed = True


class SharedHybridHostSnapshot:
    """One atomic extent containing Attention KV and the matching Mamba state."""

    def __init__(
        self,
        *,
        path: str,
        token_count: int,
        kv_pool,
        mamba_pool,
        create: bool,
        file_offset: int = 0,
        layout: Optional[HybridSnapshotLayout] = None,
    ):
        self.path = path
        self.layout = layout or HybridSnapshotLayout.from_pools(
            token_count, kv_pool, mamba_pool
        )
        if create:
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
            fd = os.open(path, flags, 0o600)
            try:
                os.ftruncate(fd, file_offset + self.layout.total_bytes)
            finally:
                os.close(fd)
        self.attention = SharedMHAHostSnapshot(
            path=path,
            token_count=token_count,
            device_pool=kv_pool,
            byte_size=self.layout.attention_bytes,
            create=False,
            file_offset=file_offset,
        )
        self.mamba = SharedMambaHostSnapshot(
            path=path,
            mamba_pool=mamba_pool,
            byte_size=self.layout.state_bytes,
            file_offset=file_offset + self.layout.state_offset,
        )
        self.byte_size = self.layout.total_bytes
        self._closed = False

    @property
    def token_count(self) -> int:
        return self.attention.token_count

    @property
    def file_offset(self) -> int:
        return self.attention.file_offset

    def start_backup_range_from_device(self, *args, **kwargs):
        return self.attention.start_backup_range_from_device(*args, **kwargs)

    def commit_backup_range_from_bounce(self, *args, **kwargs) -> None:
        self.attention.commit_backup_range_from_bounce(*args, **kwargs)

    def start_load_range_to_device(self, *args, **kwargs):
        return self.attention.start_load_range_to_device(*args, **kwargs)

    def copy_into_hicache(self, *args, **kwargs) -> None:
        self.attention.copy_into_hicache(*args, **kwargs)

    @property
    def _last_d2h_start_event(self):
        return self.attention._last_d2h_start_event

    @property
    def _last_h2d_start_event(self):
        return self.attention._last_h2d_start_event

    def start_backup_state_from_device(
        self, source_indices, stream, launch_fence=None
    ):
        return self.mamba.start_backup_from_device(
            source_indices, stream, launch_fence=launch_fence
        )

    def commit_backup_state_from_bounce(self, refs) -> None:
        self.mamba.commit_backup_from_bounce(refs)

    def start_load_state_to_device(
        self, destination_indices, stream, launch_fence=None
    ):
        return self.mamba.start_load_to_device(
            destination_indices, stream, launch_fence=launch_fence
        )

    def close(self, *, unlink: bool = False) -> None:
        if self._closed:
            return
        self.attention.close(unlink=False)
        self.mamba.close()
        self._closed = True
        if unlink:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass


class LazySharedHybridHostSnapshot:
    """Arena lease that materializes the Attention+Mamba mapping on demand."""

    def __init__(
        self,
        *,
        path: str,
        token_count: int,
        device_pool,
        byte_size: int,
        allocation_bytes: int,
        file_offset: int,
    ):
        self.path = path
        self.token_count = int(token_count)
        self.device_pool = device_pool
        self.layout = HybridSnapshotLayout.from_pools(
            token_count, device_pool.full_kv_pool, device_pool.mamba_pool
        )
        if int(byte_size) != self.layout.total_bytes:
            raise ValueError(
                f"hybrid Host extent size mismatch expected={self.layout.total_bytes} "
                f"actual={byte_size}"
            )
        self.byte_size = int(byte_size)
        self.allocation_bytes = int(allocation_bytes)
        self.file_offset = int(file_offset)
        self.offset = self.file_offset
        self._materialized = None
        self._closed = False

    def materialize(self):
        if self._closed:
            raise RuntimeError("cannot materialize a released hybrid Host extent")
        if self._materialized is None:
            self._materialized = SharedHybridHostSnapshot(
                path=self.path,
                token_count=self.token_count,
                kv_pool=self.device_pool.full_kv_pool,
                mamba_pool=self.device_pool.mamba_pool,
                create=False,
                file_offset=self.file_offset,
                layout=self.layout,
            )
        return self

    def __getattr__(self, name):
        materialized = object.__getattribute__(self, "_materialized")
        if materialized is None:
            raise RuntimeError(
                f"P hybrid Host extent is not materialized; cannot access {name}"
            )
        return getattr(materialized, name)

    def close(self, *, unlink: bool = False) -> None:
        if self._closed:
            return
        if self._materialized is not None:
            self._materialized.close(unlink=unlink)
        self._closed = True
