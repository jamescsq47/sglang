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

from sglang.srt.disaggregation.agentic_host_staging import SharedMHAHostSnapshot


def _align_up(value: int, alignment: int = mmap.ALLOCATIONGRANULARITY) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class HybridSnapshotLayout:
    attention_bytes: int
    state_offset: int
    state_bytes: int
    total_bytes: int

    @classmethod
    def from_pools(cls, token_count: int, kv_pool, mamba_pool) -> "HybridSnapshotLayout":
        attention_bytes = (
            2
            * int(token_count)
            * int(kv_pool.layer_num)
            * int(kv_pool.head_num)
            * int(kv_pool.head_dim)
            * kv_pool.store_dtype.itemsize
        )
        state_bytes = 0
        for tensor in mamba_pool.mamba_cache.conv:
            state_bytes += (
                tensor.shape[0]
                * int(torch.tensor(tensor.shape[2:]).prod().item())
                * tensor.element_size()
            )
        temporal = mamba_pool.mamba_cache.temporal
        state_bytes += (
            temporal.shape[0]
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
    ):
        if not path.startswith("/dev/shm/"):
            raise ValueError("shared Mamba snapshot must reside in /dev/shm")
        if file_offset % mmap.ALLOCATIONGRANULARITY:
            raise ValueError("Mamba snapshot offset must be mmap-aligned")
        self.path = path
        self.mamba_pool = mamba_pool
        self.byte_size = int(byte_size)
        self.file_offset = int(file_offset)
        fd = os.open(path, os.O_RDWR)
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
            shape = (tensor.shape[0], 1, *tensor.shape[2:])
            count = int(torch.tensor(shape).prod().item())
            nbytes = count * tensor.element_size()
            view = raw[cursor : cursor + nbytes].view(tensor.dtype).view(shape)
            self.conv.append(view)
            cursor += nbytes
        temporal = mamba_pool.mamba_cache.temporal
        shape = (temporal.shape[0], 1, *temporal.shape[2:])
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

    def backup_from_device(self, source_index: torch.Tensor | int) -> None:
        index = torch.as_tensor([int(source_index)], dtype=torch.int64)
        conv_cpu, temporal_cpu = self.mamba_pool.get_cpu_copy(index)
        for destination, source in zip(self.conv, conv_cpu):
            destination.copy_(source)
        self.temporal.copy_(temporal_cpu)

    def load_to_device(self, destination_index: torch.Tensor | int) -> None:
        index = torch.as_tensor([int(destination_index)], dtype=torch.int64)
        self.mamba_pool.load_cpu_copy((self.conv, self.temporal), index)

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
