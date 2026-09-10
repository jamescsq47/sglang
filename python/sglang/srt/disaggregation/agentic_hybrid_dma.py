"""Hybrid payload adapter retaining the current registered-window DMA protocol.

Only the final Attention chunk appends recurrent-state copies on the same
stream, re-recording the same physical fence after both components. Thus the
existing chunk commit, retry, cancellation and durable machinery is unchanged.
"""
from __future__ import annotations

import torch

from sglang.srt.disaggregation.agentic_hybrid_snapshot import HybridSnapshotLayout, SharedMambaHostSnapshot
from sglang.srt.disaggregation.agentic_host_staging import (
    SharedMHAHostSnapshot, H2DLaunchFence, _registered_host_arena,
    _cuda_batch_memcpy_async, _cuda_runtime, _REGISTERED_HOST_ARENAS_LOCK,
)


class RegisteredHybridHostSnapshot:
    def __init__(self, *, path, token_count, device_pool, byte_size,
                 file_offset=0, create=False, state_slots=1, **kwargs):
        if create:
            raise ValueError("hybrid snapshots require a preallocated complete extent")
        self.device_pool = device_pool
        self.layout = HybridSnapshotLayout.from_pools(
            token_count, device_pool.full_kv_pool, device_pool.mamba_pool,
            state_slots=state_slots,
        )
        if int(byte_size) != self.layout.total_bytes:
            raise ValueError("incomplete or incompatible hybrid Host extent")
        self.byte_size = int(byte_size)
        self.state_slots = int(state_slots)
        self.attention = SharedMHAHostSnapshot(
            path=path, token_count=token_count, device_pool=device_pool.full_kv_pool,
            byte_size=self.layout.attention_bytes, file_offset=file_offset,
            create=False, **kwargs,
        )
        self.state_indices = None
        self._state_mapping = None
        self._state_windows = ()
        self._state_bounce = None
        self._state_bounce_refs = None
        self._closed = False

    def __getattr__(self, name):
        return getattr(self.attention, name)

    def materialize(self):
        if self._closed:
            raise RuntimeError("cannot materialize a released hybrid extent")
        return self

    def mark_populated(self):
        if self._closed:
            raise RuntimeError("cannot publish a released hybrid extent")
        # The arena already reserved physical backing; callers publish this
        # only after the composite Attention + Mamba fence has completed.
        self.requires_prefault = False

    def set_state_indices(self, indices):
        indices = tuple(int(i) for i in indices)
        if len(indices) != self.state_slots or min(indices) < 0:
            raise ValueError("wrong hybrid state slot vector")
        cache = self.device_pool.mamba_pool.mamba_cache
        for tensor in (*cache.conv, cache.temporal):
            if max(indices) >= tensor.shape[1] or not tensor[0, indices[0]].is_contiguous():
                raise ValueError("invalid hybrid state slot or tensor layout")
        if self.state_indices is not None and self.state_indices != indices:
            raise RuntimeError("live hybrid state addresses cannot change")
        self.state_indices = indices

    def _drop_state_mapping(self):
        mapping = self._state_mapping
        if mapping is None:
            return
        if self._state_windows:
            mapping.release(self._state_windows)
            self._state_windows = ()
        with _REGISTERED_HOST_ARENAS_LOCK:
            if mapping._users <= 0:
                raise RuntimeError("hybrid mapping user underflow")
            mapping._users -= 1
        self._state_mapping = None

    def reset_state_indices_after_quiesce(self):
        """End a failed D2P load attempt; caller has settled its full fence."""
        if self._closed or self.state_slots != 1:
            raise RuntimeError("only live D2P destinations may reset state binding")
        # Mapping/window refs and durable Host bytes belong to the snapshot,
        # not this attempt. Keep them; only its destination addresses expire.
        self.state_indices = None
        self._state_bounce_refs = None

    def _prepare_state_backing(self):
        if self._state_mapping is not None or self._state_bounce is not None:
            return
        mapping = _registered_host_arena(self.path, self.device_pool.device)
        self._state_mapping = mapping
        if self.file_offset + self.layout.total_bytes > mapping.byte_size:
            self._drop_state_mapping()
            raise ValueError("hybrid state extent exceeds physical backing")
        try:
            if not self.attention._registered_dma_enabled:
                raise RuntimeError("registered DMA explicitly disabled")
            self._state_windows = mapping.acquire(
                self.file_offset + self.layout.state_offset,
                self.layout.state_bytes, self.device_pool.device,
            )
        except Exception:
            self._drop_state_mapping()
            self._state_bounce = SharedMambaHostSnapshot(
                path=self.path, mamba_pool=self.device_pool.mamba_pool,
                byte_size=self.layout.state_bytes,
                file_offset=self.file_offset + self.layout.state_offset,
                state_slots=self.state_slots,
            )

    def _append_state(self, *, stream, fence, prior_refs, to_host):
        if self.state_indices is None:
            raise RuntimeError("hybrid DMA missing pinned recurrent slots")
        refs = list(prior_refs)
        fence.copy_refs = refs
        # An Attention event recorded before this function is not a fence for
        # the following state copies. Re-arm even on partially posted failure.
        fence.armed = False
        try:
            self._prepare_state_backing()
            if self._state_bounce is not None:
                indices = torch.tensor(self.state_indices, dtype=torch.int64)
                operation = (self._state_bounce.start_backup_from_device if to_host
                             else self._state_bounce.start_load_to_device)
                inner = H2DLaunchFence(event=torch.cuda.Event(enable_timing=True))
                refs.append(inner)
                try:
                    _, state_refs = operation(indices, stream, launch_fence=inner)
                finally:
                    refs.extend(inner.copy_refs or ())
                refs.extend(state_refs)
                if to_host:
                    self._state_bounce_refs = state_refs
                return fence.event, tuple(refs)
            mapping = self._state_mapping
            cache = self.device_pool.mamba_pool.mamba_cache
            sources, destinations, sizes = [], [], []
            offset = self.file_offset + self.layout.state_offset
            for tensor in (*cache.conv, cache.temporal):
                for layer in range(tensor.shape[0]):
                    for slot in self.state_indices:
                        piece = tensor[layer, slot]
                        if not piece.is_contiguous():
                            raise ValueError("Mamba per-layer slot must be contiguous")
                        device_ptr = piece.data_ptr()
                        remaining = piece.numel() * piece.element_size()
                        while remaining:
                            size = min(remaining, mapping.window_bytes - offset % mapping.window_bytes)
                            host_ptr = mapping.raw.data_ptr() + offset
                            sources.append(device_ptr if to_host else host_ptr)
                            destinations.append(host_ptr if to_host else device_ptr)
                            sizes.append(size)
                            offset += size
                            device_ptr += size
                            remaining -= size
            if offset != self.file_offset + self.layout.total_bytes:
                raise ValueError("Mamba physical layout byte count mismatch")
            refs.extend((mapping, cache))
            with torch.cuda.stream(stream):
                fence.submitted = True
                descriptors = _cuda_batch_memcpy_async(destinations, sources, sizes, stream=stream)
                if descriptors is False:
                    runtime = _cuda_runtime()
                    for destination, source, size in zip(destinations, sources, sizes):
                        error = runtime.cudaMemcpyAsync(destination, source, size, 2 if to_host else 1, int(stream.cuda_stream))
                        if error:
                            raise RuntimeError(f"hybrid cudaMemcpyAsync returned {error}")
                else:
                    refs.append(descriptors)
        finally:
            try:
                fence.event.record(stream)
                fence.armed = True
            except Exception:
                fence.unavailable = True
                raise
        return fence.event, tuple(refs)

    def _finish_composite(self, operation, *, stream, fence, to_host):
        if self.state_indices is None:
            raise RuntimeError("hybrid DMA missing recurrent state indices")
        self._prepare_state_backing()
        inner = H2DLaunchFence(event=torch.cuda.Event(enable_timing=True))
        fence.armed = False
        fence.copy_refs = [self, inner]
        # Publish possible submission before invoking Attention. Never expose
        # its intermediate event as proof of the complete composite transfer.
        fence.submitted = True
        try:
            _, refs = operation(inner)
            return self._append_state(stream=stream, fence=fence,
                                      prior_refs=[self, inner, *refs], to_host=to_host)
        except BaseException:
            if not fence.armed and not fence.unavailable:
                try:
                    fence.event.record(stream)
                    fence.armed = True
                except Exception:
                    fence.unavailable = True
            raise

    def start_backup_range_from_device(self, source_indices, *, destination_start,
                                       stream, launch_fence=None, **kwargs):
        fence = launch_fence or H2DLaunchFence(event=torch.cuda.Event(enable_timing=True))
        if destination_start + len(source_indices) == self.token_count:
            return self._finish_composite(
                lambda inner: self.attention.start_backup_range_from_device(
                    source_indices, destination_start=destination_start, stream=stream,
                    launch_fence=inner, **kwargs), stream=stream, fence=fence, to_host=True,
            )
        return self.attention.start_backup_range_from_device(
            source_indices, destination_start=destination_start, stream=stream,
            launch_fence=fence, **kwargs)

    def commit_backup_range_from_bounce(self, host_bounce, *, destination_start, token_count):
        self.attention.commit_backup_range_from_bounce(
            host_bounce, destination_start=destination_start, token_count=token_count,
        )
        if destination_start + token_count == self.token_count and self._state_bounce is not None:
            self._state_bounce.commit_backup_from_bounce(self._state_bounce_refs)
            self._state_bounce_refs = None

    def start_load_range_from_bounce_to_device(self, device_indices, stream, *,
                                               source_start, launch_fence=None, **kwargs):
        fence = launch_fence or H2DLaunchFence(event=torch.cuda.Event(enable_timing=True))
        if source_start + len(device_indices) == self.token_count:
            return self._finish_composite(
                lambda inner: self.attention.start_load_range_from_bounce_to_device(
                    device_indices, stream, source_start=source_start, launch_fence=inner, **kwargs),
                stream=stream, fence=fence, to_host=False,
            )
        return self.attention.start_load_range_from_bounce_to_device(
            device_indices, stream, source_start=source_start, launch_fence=fence, **kwargs)

    def start_load_range_to_device(self, device_indices, stream, *, source_start,
                                  host_bounce=None, **kwargs):
        self.attention.prepare_load_range_to_bounce(
            source_start=source_start, token_count=len(device_indices), host_bounce=host_bounce,
        )
        return self.start_load_range_from_bounce_to_device(
            device_indices, stream, source_start=source_start, host_bounce=host_bounce, **kwargs,
        )

    def close(self, *, unlink=False):
        if self._closed:
            return
        self.attention.close(unlink=unlink)
        self._drop_state_mapping()
        if self._state_bounce is not None:
            self._state_bounce.close()
        self._closed = True


def open_request_snapshot(*, device_pool, state_slots=1, **kwargs):
    if hasattr(device_pool, "mamba_pool"):
        return RegisteredHybridHostSnapshot(device_pool=device_pool, state_slots=state_slots, **kwargs)
    return SharedMHAHostSnapshot(device_pool=device_pool, **kwargs)
