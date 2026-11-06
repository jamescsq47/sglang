"""Scheduler-only, physical Mamba admission for request-owned PD Prefill.

Reserve runtime and the next Radix checkpoint before COW/Forward. The output
checkpoint must survive overlapped workset allocation until cache insertion.
This module never changes a parent snapshot's owner or transmission fence.
"""
from sglang.srt.mem_cache.base_prefix_cache import EvictParams


def prefill_state_admission_enabled(server_args, req_pool):
    from sglang.srt.disaggregation.agentic_hybrid_transfer import request_owned_mamba_enabled
    # init_running_status precedes init_disaggregation: use launch arguments,
    # not Scheduler.disaggregation_mode, which does not exist at this point.
    return (server_args.disaggregation_mode == "prefill"
            and getattr(req_pool, "mamba_pool", None) is not None
            and request_owned_mamba_enabled())


def missing_runtime_slots(req, req_pool):
    count = int(getattr(req, "mamba_pool_idx", None) is None)
    if getattr(req_pool, "enable_mamba_extra_buffer", False):
        if getattr(req, "mamba_ping_pong_track_buffer", None) is None:
            count += int(req_pool.mamba_ping_pong_track_buffer_size)
    return count


def required_prefill_checkpoint_slots(*, enable_overlap, chunked):
    """Return the checkpoint reservation depth required before Forward.

    The overlap scheduler can admit the next chunk before it processes the
    previous chunk's result.  One checkpoint may therefore still be promised
    to the previous result while the newly admitted Forward needs another.
    Keeping two slots for every overlapped chunk continuation makes that
    pipeline reservation explicit instead of relying on a best-effort refill
    after Forward, when it is already too late to apply backpressure.
    """

    return 2 if enable_overlap and chunked else 1


class PrefillStateReservation:
    def __init__(self, req, pool, indices, fields):
        self.req, self.pool, self.indices, self.fields = req, pool, indices, fields

    def finish(self, admitted):
        if admitted:
            self.req._agentic_prefill_mamba_admitted = True
        if self.indices is None:
            return
        if not admitted:
            for field, old_value in self.fields:
                setattr(self.req, field, old_value)
            self.pool.free(self.indices)
        self.indices = None


def reserve_prefill_state(req, req_pool, tree_cache, *, checkpoint_slots=1):
    """Return a rollback handle, or None if no complete reservation fits.

    Existing runtime/checkpoint fields are owned elsewhere and not charged or
    freed again. Call finish(False) on a rejected candidate; finish(True) moves
    newly allocated fields to Req's ordinary cancellation/completion cleanup.
    """
    pool = req_pool.mamba_pool
    need = missing_runtime_slots(req, req_pool)
    checkpoint = getattr(req, "_agentic_mamba_prefill_checkpoint", None)
    extra = max(0, checkpoint_slots - (0 if checkpoint is None else checkpoint.numel()))
    need += extra
    if not need:
        return PrefillStateReservation(req, pool, None, [])
    if pool.available_size() < need and tree_cache.supports_mamba():
        tree_cache.evict(EvictParams(num_tokens=0, mamba_num=need - pool.available_size()))
    indices = pool.alloc(need)
    if indices is None:
        return None
    fields, offset = [], 0

    def assign(name, value):
        fields.append((name, getattr(req, name, None)))
        setattr(req, name, value)

    if getattr(req, "mamba_pool_idx", None) is None:
        assign("mamba_pool_idx", indices[offset])
        offset += 1
    if (getattr(req_pool, "enable_mamba_extra_buffer", False)
            and getattr(req, "mamba_ping_pong_track_buffer", None) is None):
        size = int(req_pool.mamba_ping_pong_track_buffer_size)
        assign("mamba_ping_pong_track_buffer", indices[offset:offset + size])
        assign("mamba_next_track_idx", 0)
        offset += size
    if extra:
        slots = indices[offset:offset + extra]
        if checkpoint is not None:
            import torch
            slots = torch.cat((checkpoint, slots))
        assign("_agentic_mamba_prefill_checkpoint", slots)
    return PrefillStateReservation(req, pool, indices, fields)


def release_prefill_checkpoint(req, pool):
    indices = getattr(req, "_agentic_mamba_prefill_checkpoint", None)
    if indices is not None:
        pool.free(indices)
        req._agentic_mamba_prefill_checkpoint = None


def fork_prefill_checkpoint(req, pool, source):
    indices = getattr(req, "_agentic_mamba_prefill_checkpoint", None)
    if indices is None:
        return pool.fork_from(source)
    target = indices[:1]
    pool.copy_from(source, target)
    req._agentic_mamba_prefill_checkpoint = indices[1:] if indices.numel() > 1 else None
    return target


def claim_prefill_batch_checkpoints(batch):
    """Move one reserved checkpoint per admitted request onto this Forward.

    The overlap scheduler may schedule the next chunk before processing this
    batch's result.  Keeping the in-flight generation's checkpoint on ``Req``
    lets later scheduling or cleanup steal it.  A claimed checkpoint instead
    follows the immutable batch/result pair until result processing.
    """

    claimed = []
    for req in batch.reqs:
        indices = getattr(req, "_agentic_mamba_prefill_checkpoint", None)
        if not getattr(req, "_agentic_prefill_mamba_admitted", False):
            claimed.append(None)
            continue
        if indices is None or indices.numel() == 0:
            raise RuntimeError(
                f"Prefill Forward has no reserved Mamba checkpoint: rid={req.rid}"
            )
        claimed.append(indices[:1])
        req._agentic_mamba_prefill_checkpoint = (
            indices[1:] if indices.numel() > 1 else None
        )
    return claimed


def restore_prefill_batch_checkpoint(batch, req, index):
    """Return one batch-owned checkpoint immediately before cache insertion."""

    claimed = getattr(batch, "_agentic_mamba_prefill_checkpoints", None)
    if claimed is None:
        return
    owned = claimed[index]
    if owned is None:
        return
    current = getattr(req, "_agentic_mamba_prefill_checkpoint", None)
    if current is None:
        req._agentic_mamba_prefill_checkpoint = owned
    else:
        import torch

        # fork_prefill_checkpoint consumes the first entry, so the result of
        # this batch cannot consume a reservation belonging to a later chunk.
        req._agentic_mamba_prefill_checkpoint = torch.cat((owned, current))
    claimed[index] = None
