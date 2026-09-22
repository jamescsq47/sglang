"""Materialize immutable CPU grants without a CUDA free-list round trip.

Not a runtime allocator: the caller must already exclusively own every address
in the plan. It must keep that ownership even if preparation fails. This module
neither calls native alloc/free nor starts KV transfer, and cannot reclaim pages.
"""

from dataclasses import dataclass

import numpy as np
import torch

from sglang.srt.disaggregation.agentic_workset_ledger import WorksetPlan


def _expand(runs):
    if not runs:
        return np.empty(0, dtype=np.int64)
    return np.concatenate([
        np.arange(run.start, run.end, dtype=np.int64) for run in runs
    ])


def _validate_runs(runs, capacity, name):
    previous = 0
    for run in sorted(runs, key=lambda run: run.start):
        if run.start < 1 or run.start < previous or run.end > capacity + 1:
            raise ValueError(f"invalid or overlapping {name} addresses")
        previous = run.end


@dataclass(frozen=True)
class PreparedWorkset:
    plan: WorksetPlan
    device_indices: torch.Tensor
    parent_page_indices: np.ndarray
    checkpoint_indices: torch.Tensor
    runtime_indices: torch.Tensor
    ready_event: object
    # Keep H2D source buffers alive until their own event completes.
    _host_indices: tuple

    def is_ready(self):
        return self.ready_event is None or self.ready_event.query()

    def wait_on(self, stream):
        """Insert a stream dependency, never synchronize the model thread."""
        if self.ready_event is not None:
            stream.wait_event(self.ready_event)


class WorksetPreparationFailure(RuntimeError):
    """A failed preparation is NOT a ready grant or an automatic free.

    Keep partial buffers and the actual preparation fence for cancellation.
    If CUDA cannot record a fence, quiescence is unknown and ownership stays
    quarantined. The controller must still close the attempt and collect the
    other ranks' physical/reference receipts before returning any addresses.
    """

    def __init__(self, plan, cause, *, ready_event=None, keepalive=(), cpu_only=False):
        super().__init__(f"workset preparation failed: {cause}")
        self.plan = plan
        self.ready_event = ready_event
        self.keepalive = keepalive
        self.cpu_only = cpu_only

    def is_quiescent(self):
        if self.cpu_only:
            return True
        if self.ready_event is None:
            return False
        try:
            return bool(self.ready_event.query())
        except Exception:
            return False


def prepare_workset(plan, *, device, page_capacity, mamba_pool=None, stream=None):
    """Prepare an exact grant on a caller-owned background CUDA stream.

    CPU execution exists for fault tests. CUDA requires an explicit stream;
    the controller must supply its dedicated preparation stream, not a Forward
    stream (this helper cannot infer an arbitrary stream's role). No
    `.cpu()` copies or global CUDA synchronization are used. Return only grants
    already validated by the ownership authority, not untrusted wire payloads.
    """
    if not isinstance(plan, WorksetPlan) or plan.page_size <= 0:
        raise ValueError("a validated workset plan is required")
    _validate_runs(plan.parent_pages + plan.suffix_pages, page_capacity, "KV page")
    expected_parent = (plan.parent_tokens + plan.page_size - 1) // plan.page_size
    expected_suffix = (plan.prompt_tokens - plan.parent_tokens + plan.page_size - 1) // plan.page_size
    if (plan.parent_tokens < 0 or plan.prompt_tokens <= 0
            or plan.prompt_tokens < plan.parent_tokens
            or sum(r.count for r in plan.parent_pages) != expected_parent
            or sum(r.count for r in plan.suffix_pages) != expected_suffix):
        raise ValueError("plan does not cover the complete parent and suffix")
    state_runs = plan.checkpoint_slots + plan.runtime_slots
    if state_runs and mamba_pool is None:
        raise ValueError("Mamba grant requires its state pool")
    if mamba_pool is not None:
        _validate_runs(state_runs, mamba_pool.size, "Mamba slot")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("workset preparation supports CPU tests or CUDA")
    if device.type == "cuda":
        if stream is None:
            raise ValueError("an explicit preparation stream on the grant device is required")
        if device.index is None:
            device = torch.device(stream.device)
        if torch.device(stream.device) != device:
            raise ValueError("preparation stream belongs to a different device")
    if mamba_pool is not None:
        pool_device = torch.device(mamba_pool.mamba_cache.temporal.device)
        if pool_device != device:
            raise ValueError("Mamba pool belongs to a different device")
    parent = _expand(plan.parent_pages)
    all_pages = _expand(plan.parent_pages + plan.suffix_pages)
    tokens = (all_pages[:, None] * plan.page_size
              + np.arange(plan.page_size, dtype=np.int64)).reshape(-1)
    state = _expand(state_runs)
    checkpoint_count = sum(r.count for r in plan.checkpoint_slots)
    parent.setflags(write=False)
    if device.type == "cpu":
        indices, state_indices = torch.from_numpy(tokens), torch.from_numpy(state)
        try:
            if mamba_pool is not None:
                mamba_pool.initialize_slots(state_indices)
        except Exception as error:
            raise WorksetPreparationFailure(
                plan, error, keepalive=(indices, state_indices), cpu_only=True
            ) from error
        return PreparedWorkset(plan, indices, parent, state_indices[:checkpoint_count],
                               state_indices[checkpoint_count:], None, (tokens, state))

    # Host buffer allocation/CPU descriptor work happens only on this background
    # preparation caller. A later cancellation still has to wait for the event.
    with torch.cuda.device(device), torch.cuda.stream(stream):
        keepalive = []
        try:
            token_host = torch.from_numpy(tokens).pin_memory()
            keepalive.append(token_host)
            state_host = torch.from_numpy(state).pin_memory()
            keepalive.append(state_host)
            indices = token_host.to(device, non_blocking=True)
            keepalive.append(indices)
            state_indices = state_host.to(device, non_blocking=True)
            keepalive.append(state_indices)
            if mamba_pool is not None:
                mamba_pool.initialize_slots(state_indices)
            event = torch.cuda.Event()
            event.record(stream)
        except Exception as error:
            failure_fence = None
            try:
                failure_fence = torch.cuda.Event()
                failure_fence.record(stream)
            except Exception:
                failure_fence = None
            raise WorksetPreparationFailure(
                plan, error, ready_event=failure_fence, keepalive=tuple(keepalive)
            ) from error
    return PreparedWorkset(plan, indices, parent, state_indices[:checkpoint_count],
                           state_indices[checkpoint_count:], event, (token_host, state_host))
