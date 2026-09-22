# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""P workset ownership and request handoff.

This module is deliberately independent of the Scheduler class.  The legacy
service entry point still requires a scheduler-safe allocator until every
allocation and release site is migrated to the single-owner controller.
Moving this code is not, on its own, asynchronous allocation.
"""

import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from sglang.srt.disaggregation.utils import kv_to_page_indices

@dataclass
class AgenticPWorksetLease:
    """Physical P-HBM ownership for one complete next-turn prompt.

    The parent slice is the destination of Direct or Slow restore.  The suffix
    slice stays unavailable to every other request until this exact request is
    admitted for incremental Prefill.  All allocator mutations are performed
    by the model scheduler; I/O workers consume only immutable page ids.
    """

    snapshot_id: str
    lease_id: int
    owner: str
    parent_tokens: int
    parent_allocated_tokens: int
    prompt_tokens: int
    allocated_tokens: int
    device_indices: torch.Tensor
    parent_page_indices: np.ndarray
    state_device_indices: Tuple[torch.Tensor, ...] = ()
    runtime_state_device_indices: Tuple[torch.Tensor, ...] = ()
    runtime_state_req: Any = None
    parent_bound: bool = False
    state: str = "active"
    suffix_cursor: int = 0
    io_attempt: Optional[str] = None
    state_cpu_indices: Optional[Tuple[Tuple[int, ...], ...]] = None
    intent_at: Optional[float] = None
    grant_at: Optional[float] = None
    plan_at: Optional[float] = None
    service_at: Optional[float] = None
    receipt_submitted_at: Optional[float] = None
    receipt_observed_at: Optional[float] = None
    # Exact immutable controller provenance. Legacy scheduler allocations leave
    # this unset; transport callbacks must never reconstruct it from addresses.
    controller_plan: Any = None
    prepared_descriptor: Any = None
    cached_prefix_tokens: int = 0

    @property
    def parent_indices(self) -> torch.Tensor:
        return self.device_indices[: self.parent_allocated_tokens]

    @property
    def suffix_indices(self) -> torch.Tensor:
        return self.device_indices[self.parent_allocated_tokens :]

    @property
    def suffix_allocated_tokens(self) -> int:
        return self.allocated_tokens - self.parent_allocated_tokens

    @property
    def remaining_suffix_indices(self) -> torch.Tensor:
        if self.state == "consumed":
            return self.suffix_indices[:0]
        return self.suffix_indices[self.suffix_cursor :]


class AgenticPWorksetLeaseBroker:
    """Thread-safe intent queue with scheduler-owned physical allocation."""

    def __init__(self, page_size: int, *, state_allocators=(), mamba_req_to_token_pool=None,
                 reserve_mamba_checkpoint=False):
        self.page_size = int(page_size)
        self._state_allocators = tuple(state_allocators)
        self._mamba_req_to_token_pool = mamba_req_to_token_pool
        self._runtime_state_slots = 1
        if mamba_req_to_token_pool is not None and getattr(
            mamba_req_to_token_pool, "enable_mamba_extra_buffer", False
        ):
            self._runtime_state_slots += int(mamba_req_to_token_pool.mamba_ping_pong_track_buffer_size)
        self._reserve_mamba_checkpoint = bool(reserve_mamba_checkpoint)
        self._runtime_state_slots += int(self._reserve_mamba_checkpoint)
        self._intents: Dict[str, Tuple[str, int, int]] = {}
        self._intent_requested_at: Dict[str, float] = {}
        self._leases: Dict[str, AgenticPWorksetLease] = {}
        self._release_requested: Dict[str, int] = {}
        self._grant_events: Deque[str] = deque()
        self._next_lease_id = 1
        self._grants = 0
        self._allocation_failures = 0
        # TP ranks must allocate the same request-generation worksets in the
        # same order.  A scheduler-broadcast plan is frozen for one native TP
        # control epoch; background Direct/Slow workers may publish intents or
        # request cancellation, but cannot mutate pages named by that epoch.
        # TP=1 never installs a plan and retains the original eager behavior.
        self._tp_plan_epoch = -1
        self._tp_plan_at = None
        self._tp_plan: Tuple[Tuple[str, str, int, int], ...] = ()
        self._tp_authoritative_retirements: frozenset[str] = frozenset()
        self._tp_cancel_pending: Dict[str, Optional[str]] = {}
        self._tp_release_pending: Dict[str, int] = {}
        self._tp_retire_requested: set[str] = set()
        self._tp_retired_in_epoch: set[str] = set()
        # A final suffix consume transfers page ownership from this broker to
        # the native Req; it is not an allocator release.  TP0 publishes that
        # ownership commit so follower ranks can drop only their broker
        # metadata without freeing the KV pages now owned by the same Req.
        self._tp_handoff_committed: set[str] = set()
        # Monotonic lifecycle boundaries can make one asynchronous producer
        # obsolete before its marker is observed.  Remember that fact so a
        # late marker cannot recreate the cancelled owner.
        self._superseded_owners: set[tuple[str, str]] = set()
        self._controller_owned = False
        self._controller_retired = set()
        self._lock = threading.RLock()

    def install_prepared(self, prepared) -> AgenticPWorksetLease:
        """Install an already-owned, device-ready exact controller grant.

        This is the existing I/O broker's handoff boundary, not another
        allocator. The TP executor must authorize the same grant on all ranks
        before callers expose it to transport. No CUDA readback or alloc/free
        takes place here. Failed installation retains controller ownership.
        """
        from sglang.srt.disaggregation.agentic_workset_device import PreparedWorkset

        if not isinstance(prepared, PreparedWorkset) or not prepared.is_ready():
            raise RuntimeError("controller descriptor is not device-ready")
        plan = prepared.plan
        if plan.page_size != self.page_size:
            raise ValueError("controller/broker page size mismatch")
        snapshot_id = plan.key.snapshot_id
        with self._lock:
            if self._tp_plan_epoch >= 0:
                raise RuntimeError("native TP allocation plan still owns this broker")
            if plan.key in self._controller_retired:
                raise RuntimeError("controller grant was already handed off or retired")
            current = self._leases.get(snapshot_id)
            if current is not None:
                if current.controller_plan != plan:
                    raise RuntimeError("controller grant conflicts with live broker owner")
                return current
            expected = (plan.owner, plan.parent_tokens, plan.prompt_tokens)
            pending = self._intents.get(snapshot_id)
            if pending is not None and pending != expected:
                raise RuntimeError("controller grant disagrees with pending intent")
            if (snapshot_id in self._release_requested
                    or (snapshot_id, plan.owner) in self._superseded_owners):
                raise RuntimeError("controller grant reached a closed broker owner")
            expected_checkpoint = int(bool(plan.parent_tokens)) if self._state_allocators else 0
            expected_runtime = self._runtime_state_slots if self._state_allocators else 0
            if self._state_allocators:
                expected_runtime += 2 - int(self._reserve_mamba_checkpoint)
            if (len(self._state_allocators) > 1
                    or prepared.checkpoint_indices.numel() != expected_checkpoint
                    or prepared.runtime_indices.numel() != expected_runtime
                    or prepared.device_indices.numel() != plan.allocated_tokens):
                raise ValueError("controller grant has an incompatible complete workset")
            states = (prepared.checkpoint_indices,) if expected_checkpoint else ()
            runtime = (prepared.runtime_indices,) if expected_runtime else ()
            state_cpu = tuple(
                index for run in plan.checkpoint_slots
                for index in range(run.start, run.end)
            )
            lease = AgenticPWorksetLease(
                snapshot_id=snapshot_id, lease_id=plan.key.version,
                owner=plan.owner, parent_tokens=plan.parent_tokens,
                parent_allocated_tokens=sum(r.count for r in plan.parent_pages) * self.page_size,
                prompt_tokens=plan.prompt_tokens, allocated_tokens=plan.allocated_tokens,
                device_indices=prepared.device_indices,
                parent_page_indices=prepared.parent_page_indices,
                state_device_indices=states, runtime_state_device_indices=runtime,
                state_cpu_indices=(state_cpu,) if states else None,
                intent_at=self._intent_requested_at.pop(snapshot_id, None),
                grant_at=time.monotonic(), controller_plan=plan,
                prepared_descriptor=prepared,
            )
            self._leases[snapshot_id] = lease
            self._controller_owned = True
            self._intents.pop(snapshot_id, None)
            self._grant_events.append(snapshot_id)
            self._grants += 1
            return lease

    def adopt_fresh_prefix(self, req, lease, *, prefix_tokens, pinned, return_prefix):
        """Consume a native, pinned prefix without allocating its suffix again.

        Native TP compute admission must select one identical prefix cut.
        ``return_prefix`` queues an exact private-prefix retirement and must
        either accept it completely or raise without accepting. Its eventual
        all-rank/device fence, not this cursor update, makes pages reusable.
        The original grant remains immutable. Call after fresh handoff and
        before the first suffix consumption, never for an external parent.
        """
        with self._lock:
            current = self._leases.get(lease.snapshot_id)
            if (current is not lease or lease.controller_plan is None
                    or lease.parent_tokens or lease.state != "handed"
                    or getattr(req, "_agentic_p_workset_lease", None) is not lease):
                raise RuntimeError("cached prefix requires this request's fresh controller lease")
            if (type(prefix_tokens) is not int or prefix_tokens < 0
                    or prefix_tokens >= lease.prompt_tokens
                    or prefix_tokens % self.page_size
                    or len(req.prefix_indices) != prefix_tokens or pinned is not True):
                raise ValueError("cached prefix must be pinned, page-aligned and leave a suffix")
            if lease.cached_prefix_tokens == prefix_tokens and lease.suffix_cursor == prefix_tokens:
                return
            if lease.suffix_cursor or lease.cached_prefix_tokens:
                raise RuntimeError("cannot change a prefix after suffix consumption")
            if prefix_tokens:
                return_prefix(lease.controller_plan, prefix_tokens)
                lease.cached_prefix_tokens = prefix_tokens
                lease.suffix_cursor = prefix_tokens
                req._agentic_workset_suffix_indices = lease.remaining_suffix_indices

    @staticmethod
    def direct_owner(snapshot_id: str) -> str:
        return f"direct:{snapshot_id}"

    @staticmethod
    def slow_owner(snapshot_id: str, rid: str) -> str:
        return f"slow:{snapshot_id}:{rid}"

    def request(
        self,
        snapshot_id: str,
        parent_tokens: int,
        prompt_tokens: int,
        *,
        owner: str = "legacy",
    ) -> bool:
        parent_tokens = int(parent_tokens)
        prompt_tokens = int(prompt_tokens)
        if parent_tokens < 0 or prompt_tokens <= 0 or prompt_tokens < parent_tokens:
            raise ValueError(
                f"invalid workset shape parent={parent_tokens} prompt={prompt_tokens}"
            )
        with self._lock:
            if (snapshot_id, owner) in self._superseded_owners:
                return False
            # TP retirement is snapshot-scoped.  Until every rank commits the
            # old owner's terminal transition, accepting a successor intent
            # would let that pending commit delete or release the new owner.
            if (
                snapshot_id in self._tp_retire_requested
                or snapshot_id in self._tp_cancel_pending
                or snapshot_id in self._tp_release_pending
            ):
                return False
            current = self._leases.get(snapshot_id)
            if current is not None:
                return (
                    current.owner == owner
                    and current.parent_tokens == parent_tokens
                    and current.prompt_tokens == prompt_tokens
                    and current.state not in {"releasing", "consumed"}
                )
            pending = self._intents.get(snapshot_id)
            if pending is not None:
                return pending == (owner, parent_tokens, prompt_tokens)
            if snapshot_id in self._release_requested:
                return False
            self._intents[snapshot_id] = (owner, parent_tokens, prompt_tokens)
            self._intent_requested_at[snapshot_id] = time.monotonic()
            return True

    def get(
        self, snapshot_id: str, *, owner: Optional[str] = None
    ) -> Optional[AgenticPWorksetLease]:
        with self._lock:
            lease = self._leases.get(snapshot_id)
            if lease is not None and (owner is None or lease.owner == owner):
                return lease
            return None

    def direct_admission_snapshot_ids(self) -> frozenset[str]:
        """Count selected work that can still occupy a Direct transport lane.

        A cancelled, unposted owner retains its HBM until native TP retirement,
        but cannot start I/O and must not consume a new transport admission.
        Active grants/receivers are counted separately by the scheduler; local
        cancellation here is never a substitute for their group DMA fence.
        """
        with self._lock:
            def retiring(snapshot_id):
                # Exactly the tombstones checked by begin_io_attempt under
                # this same lock. A stale frozen plan may still materialize
                # pages, but cannot authorize this owner to publish I/O.
                return (
                    snapshot_id in self._tp_cancel_pending
                    or snapshot_id in self._tp_release_pending
                    or snapshot_id in self._tp_retire_requested
                )

            selected = {
                snapshot_id
                for snapshot_id, (owner, _, _) in self._intents.items()
                if owner == self.direct_owner(snapshot_id)
                and not retiring(snapshot_id)
            }
            selected.update(
                snapshot_id
                for snapshot_id, lease in self._leases.items()
                if lease.owner == self.direct_owner(snapshot_id)
                and lease.state in {"active", "io_reserved", "io_inflight"}
                and not (
                    lease.state == "active"
                    and lease.io_attempt is None
                    and retiring(snapshot_id)
                )
            )
            return frozenset(selected)

    def owner_has_unretired_work(self, snapshot_id: str, *, owner: str) -> bool:
        """Return whether an owner can still materialize or owns P pages.

        In TP mode an immutable allocation plan is physical work even before
        ``service()`` has created its rank-local lease.  Callers that gate a
        successor owner must therefore wait for pending intents, live leases,
        *and* a frozen plan entry to retire group-wide.
        """

        with self._lock:
            # Although this query is owner-scoped, TP retirement ACK is not.
            # A successor owner may not start in the window between the plan
            # dropping the old owner and all ranks committing that snapshot-
            # scoped retirement.
            if (
                snapshot_id in self._tp_retire_requested
                or snapshot_id in self._tp_cancel_pending
                or snapshot_id in self._tp_release_pending
            ):
                return True
            pending = self._intents.get(snapshot_id)
            if pending is not None and pending[0] == owner:
                return True
            lease = self._leases.get(snapshot_id)
            if lease is not None and lease.owner == owner:
                return True
            return snapshot_id not in self._tp_retired_in_epoch and any(
                entry[0] == snapshot_id and entry[1] == owner
                for entry in self._tp_plan
            )

    def request_release(
        self,
        snapshot_id: str,
        lease: Optional[AgenticPWorksetLease] = None,
        *,
        owner: Optional[str] = None,
        io_attempt: Optional[str] = None,
    ) -> bool:
        """Schedule release of the exact lease owned by the caller.

        A caller without a lease identity cannot release anything; intent
        cancellation is a separate owner-scoped operation.  This prevents a
        delayed Direct callback from cancelling a newer Slow owner for the
        same request-generation.
        """

        with self._lock:
            if lease is None:
                return False
            current = self._leases.get(snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                return False
            if owner is not None and current.owner != owner:
                return False
            if (
                current.state in {"active", "retire_ready"}
                and any(entry[0] == snapshot_id for entry in self._tp_plan)
            ):
                # The current TP allocation epoch is immutable.  Record the
                # exact lease terminal now; only all-rank retirement permits
                # TP0 to omit it and release at a scheduler-safe boundary.
                # retire_ready is local quiescence, not group permission: a
                # repeated receiver cleanup must not turn it into a local free.
                # Releasing eagerly here would let the still-frozen plan
                # recreate the snapshot with a new lease id.
                self._tp_release_pending[snapshot_id] = current.lease_id
                self._tp_retire_requested.add(snapshot_id)
                return True
            pending = self._intents.get(snapshot_id)
            if pending is not None and pending[0] == current.owner:
                self._intents.pop(snapshot_id, None)
                self._intent_requested_at.pop(snapshot_id, None)
            if current.state in {"io_reserved", "io_inflight"}:
                if current.io_attempt != io_attempt:
                    return False
                planned = any(entry[0] == snapshot_id for entry in self._tp_plan)
                # A TP retirement is intent, not proof of posted I/O. Keep
                # an unposted attempt cancellable after its claim ACK drains.
                # TP1 retains its existing eager release-pending semantics.
                if current.state != "io_reserved" or not planned:
                    current.state = "release_pending"
                if planned:
                    self._tp_release_pending[snapshot_id] = current.lease_id
                    self._tp_retire_requested.add(snapshot_id)
                return False
            if current.state in {
                "release_pending",
                "binding",
                "handed",
                "consumed",
                "releasing",
            }:
                return False
            current.state = "releasing"
            self._release_requested[snapshot_id] = current.lease_id
            return True

    def release_handed(
        self,
        snapshot_id: str,
        lease: AgenticPWorksetLease,
        *,
        req,
    ) -> bool:
        """Release a workset after scheduler ownership was handed to ``req``.

        Transport callbacks deliberately cannot release a handed lease.  Only
        the live request that owns the suffix may return it on cancellation.
        This separates request lifetime from stale Direct/Slow completions.
        """

        with self._lock:
            current = self._leases.get(snapshot_id)
            if current is None:
                # The complete suffix may already belong to the native Req.
                # Native request cleanup owns those pages; treating this as a
                # broker retirement would double-free them on a TP follower.
                return False
            if current.lease_id != lease.lease_id:
                return False
            if current.state != "handed":
                return False
            if getattr(req, "_agentic_p_workset_lease", None) is not current:
                return False
            if self._tp_plan_epoch >= 0:
                # A handed lease has deliberately left the allocator plan,
                # but cancellation still has to retire its remaining local
                # suffix on every TP rank as one group transaction.
                current.state = "retire_ready"
                self._tp_release_pending[snapshot_id] = current.lease_id
                self._tp_retire_requested.add(snapshot_id)
                return True
            current.state = "releasing"
            self._release_requested[snapshot_id] = current.lease_id
            return True

    def _release_hybrid_lease_slots(self, lease):
        for pool, indices in zip(self._state_allocators, lease.state_device_indices):
            pool.free(indices)
        for pool, indices in zip(self._state_allocators, lease.runtime_state_device_indices):
            pool.free(indices)
        lease.state_device_indices = ()
        lease.runtime_state_device_indices = ()

    def service(
        self,
        allocator,
        *,
        reserve_tokens: int = 0,
    ) -> None:
        """Allocate/free only at a scheduler-safe boundary.

        ``reserve_tokens`` protects the unfinished suffix of the one native
        chunked-Prefill request.  That request was admitted only after the
        scheduler proved that its complete prompt fitted, but its later
        chunks are allocated lazily.  Background Direct/Slow intents must not
        consume that already-promised capacity between chunks.
        """

        reserve_tokens = max(0, int(reserve_tokens))
        service_at = time.monotonic()

        with self._lock:
            if self._controller_owned:
                raise RuntimeError("controller-owned broker cannot service the native allocator")
            releases = tuple(self._release_requested.items())
            self._release_requested.clear()
            for snapshot_id, expected_id in releases:
                lease = self._leases.get(snapshot_id)
                if (
                    lease is not None
                    and lease.lease_id == expected_id
                    and lease.state == "releasing"
                ):
                    self._leases.pop(snapshot_id, None)
                    allocator.free(
                        lease.remaining_suffix_indices
                        if lease.parent_bound or lease.parent_tokens == 0
                        else lease.device_indices
                    )
                    self._release_hybrid_lease_slots(lease)
                    if self._tp_plan_epoch >= 0:
                        self._tp_retired_in_epoch.add(snapshot_id)

            allocation_plan = self._tp_plan if self._tp_plan_epoch >= 0 else None
            if allocation_plan is None:
                candidates = tuple(
                    (snapshot_id, *intent)
                    for snapshot_id, intent in self._intents.items()
                )
            else:
                planned_ids = {entry[0] for entry in allocation_plan}
                # Rank 0 omits a cancelled/released active lease from the next
                # epoch.  Followers mirror that removal here.  I/O-owned or
                # scheduler-owned leases may never disappear by reconciliation:
                # reaching this branch would indicate an earlier TP split and
                # must fail closed before another model collective.
                for snapshot_id, lease in tuple(self._leases.items()):
                    if snapshot_id in planned_ids:
                        continue
                    if lease.state != "active":
                        raise RuntimeError(
                            "TP workset plan removed a non-cancellable lease "
                            f"{snapshot_id} state={lease.state}"
                        )
                    self._leases.pop(snapshot_id, None)
                    allocator.free(lease.device_indices)
                    self._release_hybrid_lease_slots(lease)
                    self._tp_release_pending.pop(snapshot_id, None)
                for snapshot_id, cancel_owner in tuple(
                    self._tp_cancel_pending.items()
                ):
                    if snapshot_id not in planned_ids:
                        pending = self._intents.get(snapshot_id)
                        if pending is not None and (
                            cancel_owner is None or pending[0] == cancel_owner
                        ):
                            self._intents.pop(snapshot_id, None)
                            self._intent_requested_at.pop(snapshot_id, None)
                        self._tp_cancel_pending.pop(snapshot_id, None)
                candidates = tuple(
                    (
                        str(snapshot_id),
                        str(owner),
                        int(parent_tokens),
                        int(prompt_tokens),
                    )
                    for snapshot_id, owner, parent_tokens, prompt_tokens in (
                        allocation_plan
                    )
                )

            for snapshot_id, owner, parent_tokens, prompt_tokens in candidates:
                if snapshot_id in self._tp_retired_in_epoch:
                    continue
                existing = self._leases.get(snapshot_id)
                if existing is not None:
                    if (
                        existing.owner != owner
                        or existing.parent_tokens != parent_tokens
                        or existing.prompt_tokens != prompt_tokens
                    ):
                        raise RuntimeError(
                            "TP workset allocation plan disagrees with an existing "
                            f"lease for {snapshot_id}"
                        )
                    continue
                local_intent = self._intents.get(snapshot_id)
                if local_intent is None:
                    # TP0 is the sole logical decision maker.  A follower can
                    # observe the filesystem/HTTP marker one tick later, so
                    # install the immutable group intent from TP0 rather than
                    # allowing rank-local observation order to choose pages.
                    if allocation_plan is None:
                        continue
                    self._intents[snapshot_id] = (
                        owner,
                        parent_tokens,
                        prompt_tokens,
                    )
                    self._intent_requested_at[snapshot_id] = time.monotonic()
                    local_intent = self._intents[snapshot_id]
                if local_intent != (owner, parent_tokens, prompt_tokens):
                    local_owner = local_intent[0]
                    direct_to_slow_race = (
                        allocation_plan is not None
                        and owner == self.direct_owner(snapshot_id)
                        and (snapshot_id, owner) in self._superseded_owners
                        and local_owner.startswith(f"slow:{snapshot_id}:")
                        and local_intent[1:] == (parent_tokens, prompt_tokens)
                    )
                    if not direct_to_slow_race:
                        raise RuntimeError(
                            "TP workset allocation plan disagrees with local intent "
                            f"for {snapshot_id}: "
                            f"plan={(owner, parent_tokens, prompt_tokens)} "
                            f"local={local_intent}"
                        )
                    # The plan is already a TP0-broadcast group decision, but
                    # one rank may observe HOST_READY and enqueue Slow before
                    # another.  Both ranks must still allocate the old Direct
                    # entry in this epoch so every following page allocation
                    # remains identical.  TP0's persistent tombstone retires
                    # it group-wide in the next control epoch; Host recovery
                    # then retries the Slow intent.
                    local_intent = (owner, parent_tokens, prompt_tokens)
                # Parent and suffix have distinct ownership transitions: the
                # parent is filled by Direct/Slow I/O, while the suffix is
                # filled by incremental Prefill.  Round each slice
                # independently so an unaligned parent can never steal the
                # first page required by the suffix.
                parent_allocated = (
                    (parent_tokens + self.page_size - 1) // self.page_size
                ) * self.page_size
                suffix_tokens = prompt_tokens - parent_tokens
                suffix_allocated = (
                    (suffix_tokens + self.page_size - 1) // self.page_size
                ) * self.page_size
                allocated_tokens = parent_allocated + suffix_allocated
                if reserve_tokens and (
                    allocator.available_size() - reserve_tokens < allocated_tokens
                ):
                    self._allocation_failures += 1
                    if allocation_plan is not None:
                        break
                    continue
                device_indices = allocator.alloc(allocated_tokens)
                if device_indices is None:
                    self._allocation_failures += 1
                    if allocation_plan is not None:
                        break
                    continue
                state_device_indices = []
                runtime_state_device_indices = []
                for state_allocator in self._state_allocators:
                    # Fresh/recompute has no restored checkpoint. Its complete
                    # Attention suffix and Mamba runtime are still one grant.
                    state_indices = state_allocator.alloc(1) if parent_tokens else None
                    runtime_slots = self._runtime_state_slots
                    if parent_tokens == 0:
                        runtime_slots += 2 - int(self._reserve_mamba_checkpoint)
                    runtime_indices = state_allocator.alloc(runtime_slots)
                    if (parent_tokens and state_indices is None) or runtime_indices is None:
                        if state_indices is not None:
                            state_allocator.free(state_indices)
                        if runtime_indices is not None:
                            state_allocator.free(runtime_indices)
                        for pool, indices in zip(self._state_allocators, state_device_indices):
                            pool.free(indices)
                        for pool, indices in zip(self._state_allocators, runtime_state_device_indices):
                            pool.free(indices)
                        allocator.free(device_indices)
                        device_indices = None
                        break
                    if state_indices is not None:
                        state_device_indices.append(state_indices)
                    runtime_state_device_indices.append(runtime_indices)
                if device_indices is None:
                    self._allocation_failures += 1
                    if allocation_plan is not None:
                        break
                    continue
                parent_indices = device_indices[:parent_allocated]
                state_cpu_indices = None
                if (self._tp_plan_epoch >= 0 and state_device_indices
                        and os.getenv("SGLANG_AGENTIC_CONTROL_ENDPOINT")):
                    from sglang.srt.disaggregation.agentic_hybrid_transfer import (
                        prepare_workset_transfer_indices,
                    )
                    try:
                        page_indices, state_cpu_indices = prepare_workset_transfer_indices(
                            parent_indices, state_device_indices, self.page_size
                        )
                    except Exception:
                        # No lease/grant is visible until the blocking index
                        # fence and both CPU descriptors have completed.
                        for pool, indices in zip(self._state_allocators, state_device_indices):
                            pool.free(indices)
                        for pool, indices in zip(self._state_allocators, runtime_state_device_indices):
                            pool.free(indices)
                        allocator.free(device_indices)
                        raise
                else:
                    page_indices = kv_to_page_indices(
                        parent_indices.cpu().numpy(), self.page_size
                    )
                self._leases[snapshot_id] = AgenticPWorksetLease(
                    snapshot_id=snapshot_id,
                    lease_id=self._next_lease_id,
                    owner=owner,
                    parent_tokens=parent_tokens,
                    parent_allocated_tokens=parent_allocated,
                    prompt_tokens=prompt_tokens,
                    allocated_tokens=allocated_tokens,
                    device_indices=device_indices,
                    parent_page_indices=page_indices,
                    state_device_indices=tuple(state_device_indices),
                    runtime_state_device_indices=tuple(runtime_state_device_indices),
                    state_cpu_indices=state_cpu_indices,
                    intent_at=self._intent_requested_at.pop(snapshot_id, None),
                    grant_at=time.monotonic(),
                    plan_at=self._tp_plan_at,
                    service_at=service_at,
                )
                self._next_lease_id += 1
                self._grants += 1
                self._intents.pop(snapshot_id, None)
                self._grant_events.append(snapshot_id)

    def install_tp_plan(
        self,
        epoch: int,
        plan: Sequence[Tuple[str, str, int, int]],
        *,
        retiring_ids: Sequence[str] = (),
    ) -> None:
        """Freeze one TP0-authored allocator transaction until the next epoch."""

        normalized = tuple(
            (
                str(snapshot_id),
                str(owner),
                int(parent_tokens),
                int(prompt_tokens),
            )
            for snapshot_id, owner, parent_tokens, prompt_tokens in plan
        )
        if len({entry[0] for entry in normalized}) != len(normalized):
            raise RuntimeError(f"duplicate snapshot in TP workset epoch {epoch}")
        authoritative_retirements = frozenset(str(item) for item in retiring_ids)
        with self._lock:
            if self._controller_owned:
                raise RuntimeError("controller-owned broker cannot install a native TP allocation plan")
            epoch = int(epoch)
            if epoch == self._tp_plan_epoch:
                if (
                    normalized == self._tp_plan
                    and authoritative_retirements
                    == self._tp_authoritative_retirements
                ):
                    return
                raise RuntimeError(
                    f"TP workset epoch {epoch} was replayed with new content"
                )
            if epoch < self._tp_plan_epoch:
                raise RuntimeError(
                    f"stale TP workset epoch {epoch} < {self._tp_plan_epoch}"
                )
            for snapshot_id, owner, parent_tokens, prompt_tokens in normalized:
                current = self._leases.get(snapshot_id)
                if current is not None and (
                    current.owner != owner
                    or current.parent_tokens != parent_tokens
                    or current.prompt_tokens != prompt_tokens
                ):
                    raise RuntimeError(
                        "TP workset epoch changes a live lease shape for "
                        f"{snapshot_id}"
                    )
                pending = self._intents.get(snapshot_id)
                if pending is not None and pending != (
                    owner,
                    parent_tokens,
                    prompt_tokens,
                ):
                    pending_owner = pending[0]
                    direct_to_slow_race = (
                        owner == self.direct_owner(snapshot_id)
                        and (snapshot_id, owner) in self._superseded_owners
                        and pending_owner.startswith(f"slow:{snapshot_id}:")
                        and pending[1:] == (parent_tokens, prompt_tokens)
                    )
                    if not direct_to_slow_race:
                        raise RuntimeError(
                            "TP workset epoch disagrees with local intent for "
                            f"{snapshot_id}: "
                            f"plan={(owner, parent_tokens, prompt_tokens)} "
                            f"local={pending}"
                        )
                if snapshot_id not in authoritative_retirements:
                    # Rank-local marker expiry is not a TP group decision.
                    # TP0's live plan clears only unstarted/active local
                    # tombstones.  A shard with a posted or quiesced failed
                    # DMA keeps its terminal state until the ordinary group
                    # failure path publishes an authoritative retire command.
                    self._tp_cancel_pending.pop(snapshot_id, None)
                    if current is None or current.state == "active":
                        self._tp_release_pending.pop(snapshot_id, None)
                        self._tp_retire_requested.discard(snapshot_id)
            self._tp_plan_epoch = epoch
            self._tp_plan_at = time.monotonic()
            self._tp_plan = normalized
            self._tp_authoritative_retirements = authoritative_retirements
            self._tp_retired_in_epoch.clear()
            for snapshot_id in authoritative_retirements:
                self._prepare_tp_retire_locked(snapshot_id)

    def prepare_tp_plan(
        self,
        epoch: int,
        *,
        retiring_ids: Optional[Sequence[str]] = None,
    ) -> tuple[tuple[str, str, int, int], ...]:
        """Build and freeze TP0's plan before it enters the broadcast.

        Freezing at construction, rather than when TP0 later consumes its own
        broadcast, closes the interval in which a background timeout could
        delete a newly planned intent without leaving a deferred-cancel record.
        """

        with self._lock:
            plan = []
            seen = set()
            for snapshot_id, lease in self._leases.items():
                if (
                    lease.state in {"consumed", "releasing"}
                    or snapshot_id in self._release_requested
                ):
                    # A consumed lease has already moved to the native Req on
                    # this rank.  Native TP scheduling must perform that same
                    # ownership move on every rank before TP0 publishes the
                    # separate group handoff acknowledgement.
                    continue
                plan.append(
                    (
                        snapshot_id,
                        lease.owner,
                        int(lease.parent_tokens),
                        int(lease.prompt_tokens),
                    )
                )
                seen.add(snapshot_id)
            for snapshot_id, (
                owner,
                parent_tokens,
                prompt_tokens,
            ) in self._intents.items():
                if (
                    snapshot_id not in seen
                ):
                    plan.append(
                        (
                            snapshot_id,
                            owner,
                            int(parent_tokens),
                            int(prompt_tokens),
                        )
                    )
            frozen = tuple(plan)
            self.install_tp_plan(
                epoch,
                frozen,
                retiring_ids=(
                    self.tp_retire_candidates
                    if retiring_ids is None
                    else retiring_ids
                ),
            )
            return frozen

    def prepare_tp_control(
        self,
        epoch: int,
        *,
        retiring_ids: Sequence[str] = (),
    ) -> tuple[
        tuple[tuple[str, str, int, int], ...],
        tuple[str, ...],
        tuple[str, ...],
    ]:
        """Atomically freeze one TP allocation-and-retirement transaction.

        TP0 must not sample retire candidates and freeze its allocation plan
        under separate broker locks: an asynchronous I/O completion can ask
        to retire a lease in that interval.  Folding every candidate visible
        at the freeze boundary into the authoritative retirement set ensures
        that the native TP broadcast carries exactly the terminal decisions
        used when the plan was installed.
        """

        with self._lock:
            authoritative_retirements = set(str(item) for item in retiring_ids)
            authoritative_retirements.update(self._tp_retire_requested)
            # HOST_READY can supersede Direct after TP0 has already frozen the
            # current allocation epoch but before that epoch is installed on
            # every rank.  Such an old plan is still installed verbatim for
            # rank consistency; the next TP0 epoch must then derive a group
            # retirement from the persistent owner tombstone.  Relying only
            # on the one-shot cancel flag loses this race when no local intent
            # or lease existed at the instant HOST_READY was observed.
            for snapshot_id, owner in self._superseded_owners:
                if owner != self.direct_owner(snapshot_id):
                    continue
                lease = self._leases.get(snapshot_id)
                intent = self._intents.get(snapshot_id)
                planned = any(
                    entry[0] == snapshot_id and entry[1] == owner
                    for entry in self._tp_plan
                )
                # Retirement is snapshot-scoped, while supersession is
                # owner-scoped.  Once Slow has taken over the same snapshot,
                # an older Direct plan must not retire the new owner.  The old
                # plan is evidence only while there is no newer live owner.
                live_owner = (
                    lease.owner
                    if lease is not None
                    else (intent[0] if intent is not None else None)
                )
                if live_owner == owner or (live_owner is None and planned):
                    authoritative_retirements.add(snapshot_id)
                    self._tp_retire_requested.add(snapshot_id)
            frozen_retirements = tuple(sorted(authoritative_retirements))
            plan = self.prepare_tp_plan(
                epoch,
                retiring_ids=frozen_retirements,
            )
            return (
                plan,
                frozen_retirements,
                tuple(sorted(self._tp_handoff_committed)),
            )

    def drain_grant_events(self) -> tuple[str, ...]:
        """Return newly allocated worksets without scanning broker state."""

        with self._lock:
            events = tuple(self._grant_events)
            self._grant_events.clear()
            return events

    def attach_runtime_state_for_bind(
        self, snapshot_id: str, req, lease: AgenticPWorksetLease
    ) -> None:
        """Attach pre-reserved active/tracking slots before native Mamba COW."""

        with self._lock:
            current = self._leases.get(snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                raise RuntimeError(f"workset lease disappeared for {snapshot_id}")
            if current.state != "binding" or not current.parent_bound:
                raise RuntimeError(
                    f"cannot attach runtime state from {current.state} workset"
                )
            self._attach_runtime_state_to_req(current, req)

    def _attach_runtime_state_to_req(self, current, req) -> None:
        """Caller holds the ownership lock and validated its handoff boundary."""

        if not current.runtime_state_device_indices:
            return
        if current.runtime_state_req is not None:
            if current.runtime_state_req is req:
                return
            raise RuntimeError("runtime Mamba reservation has another owner")
        if getattr(req, "mamba_pool_idx", None) is not None:
            raise RuntimeError("request already owns active Mamba state")
        if len(current.runtime_state_device_indices) != 1:
            raise RuntimeError("V1 supports exactly one Mamba state component")
        reserved = current.runtime_state_device_indices[0]
        pool = self._mamba_req_to_token_pool
        buffer = None
        if pool is not None and getattr(pool, "enable_mamba_extra_buffer", False):
            buffer_size = int(pool.mamba_ping_pong_track_buffer_size)
            checkpoint_slots = self._checkpoint_slots(current)
            ping_pong = reserved[1:-checkpoint_slots] if checkpoint_slots else reserved[1:]
            if current.controller_plan is not None:
                if ping_pong.numel() != buffer_size:
                    raise RuntimeError("prepared workset lacks exact Mamba tracking slots")
                # The actor already materialized these exact initialized slot
                # indices. Native Req ownership needs only a view, not another
                # CUDA allocation/fill/copy on the Forward thread.
                buffer = ping_pong
            else:
                buffer = torch.full(
                    (buffer_size,),
                    -1,
                    dtype=reserved.dtype,
                    device=reserved.device,
                )
                if ping_pong.numel() > buffer_size:
                    raise RuntimeError("too many reserved Mamba tracking slots")
                buffer[: ping_pong.numel()] = ping_pong
        # All tensor materialization/validation precedes publishing any Req
        # references. A construction failure must leave one owner, the lease.
        active = reserved[0]
        checkpoint_slots = self._checkpoint_slots(current)
        checkpoint = reserved[-checkpoint_slots:] if checkpoint_slots else None
        req.mamba_pool_idx = active
        req.mamba_needs_clear = False
        if buffer is not None:
            req.mamba_ping_pong_track_buffer = buffer
            req.mamba_next_track_idx = 0
        if checkpoint is not None:
            req._agentic_mamba_prefill_checkpoint = checkpoint
        current.runtime_state_req = req

    def _checkpoint_slots(self, lease) -> int:
        # Fresh chunked Prefill starts with two output checkpoints, matching
        # native admission. They are a rolling budget, NOT one slot per chunk:
        # the future controller integration must replenish only after Radix
        # returns the retired checkpoint. This API alone does not migrate the
        # native cache_unfinished_req replenishment allocation. If another
        # request still pins that checkpoint, two slots alone are NOT a
        # completion guarantee: integration must prove it reclaimable or hold
        # a real replacement credit, never fall back to unreserved allocation.
        return 2 if lease.parent_tokens == 0 or lease.controller_plan is not None else int(self._reserve_mamba_checkpoint)

    def handoff_fresh_to_req(
        self, snapshot_id: str, req, lease: AgenticPWorksetLease
    ) -> None:
        """Hand an untransferred complete prompt to Req, without a fake bind.

        parent_tokens=0 means there is no external or Radix parent to restore.
        The caller must not already own a cached prefix or runtime slots.
        Attention chunks consume this lease; transferred suffix/runtime belong
        to native Req cleanup, while unused suffix stays broker-owned.
        """
        with self._lock:
            current = self._leases.get(snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                raise RuntimeError(f"workset lease disappeared for {snapshot_id}")
            if current.parent_tokens != 0:
                raise RuntimeError("fresh handoff cannot bypass parent restore")
            if current.state == "handed":
                if getattr(req, "_agentic_p_workset_lease", None) is current:
                    return
                raise RuntimeError("fresh workset was handed to another request")
            if (current.state != "active" or current.io_attempt is not None
                    or current.parent_bound or current.state_device_indices
                    or snapshot_id in self._tp_cancel_pending
                    or snapshot_id in self._tp_release_pending
                    or snapshot_id in self._tp_retire_requested):
                raise RuntimeError("fresh workset is not available for handoff")
            if len(req.origin_input_ids) != current.prompt_tokens:
                raise RuntimeError("fresh workset prompt changed")
            if (getattr(req, "req_pool_idx", None) is not None
                    or len(getattr(req, "prefix_indices", ()))
                    or getattr(req, "mamba_pool_idx", None) is not None
                    or getattr(req, "mamba_ping_pong_track_buffer", None) is not None
                    or getattr(req, "_agentic_mamba_prefill_checkpoint", None) is not None):
                raise RuntimeError("fresh request already owns KV or runtime state")
            self._attach_runtime_state_to_req(current, req)
            self._finish_handoff_to_req(current, req)

    def handoff_to_req(
        self, snapshot_id: str, req, lease: AgenticPWorksetLease
    ) -> None:
        """Move the complete lease from broker ownership to one live Req."""

        with self._lock:
            current = self._leases.get(snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                raise RuntimeError(f"workset lease disappeared for {snapshot_id}")
            # Handoff is the ownership commit boundary.  Cleanup after that
            # boundary (Host extent release, TP bookkeeping, lane return) may
            # itself need a retry, so accepting the exact same Req/lease again
            # makes the commit safely idempotent.
            if current.state == "handed":
                if getattr(req, "_agentic_p_workset_lease", None) is current:
                    return
                raise RuntimeError(
                    f"workset lease was handed to another request for {snapshot_id}"
                )
            if current.state != "binding":
                raise RuntimeError(
                    f"workset lease is {current.state} for {snapshot_id}"
                )
            if not current.parent_bound:
                raise RuntimeError(f"workset parent is not bound for {snapshot_id}")
            actual_prompt_tokens = len(req.origin_input_ids)
            if actual_prompt_tokens != current.prompt_tokens:
                raise RuntimeError(
                    f"workset prompt changed for {snapshot_id}: "
                    f"reserved={current.prompt_tokens} actual={actual_prompt_tokens}"
                )
            if current.state_device_indices:
                raise RuntimeError("Radix checkpoint ownership was not committed")
            self._finish_handoff_to_req(current, req)

    def _finish_handoff_to_req(self, current, req) -> None:
        """Commit validated ownership; no allocator or transport operations."""
        snapshot_id = current.snapshot_id
        if current.runtime_state_device_indices:
            if current.runtime_state_req is not req or req.mamba_pool_idx is None:
                raise RuntimeError("runtime Mamba reservation was not attached")
            current.runtime_state_device_indices = ()
            current.runtime_state_req = None
        current.state = "handed"
        self._intents.pop(snapshot_id, None)
        self._intent_requested_at.pop(snapshot_id, None)
        # This marker is the scheduler-visible ownership contract. The request
        # owns its full workset until it leaves P HBM (P->D or P->D Host).
        req._agentic_workset_backed = True
        req._agentic_p_workset_lease = current
        req._agentic_p_workset_broker = self
        req._agentic_workset_suffix_indices = current.remaining_suffix_indices
        if getattr(req, "mamba_pool_idx", None) is not None:
            req._agentic_mamba_runtime_reserved = True
            req.mamba_last_track_seqlen = current.parent_tokens

    def consume_suffix(
        self,
        lease: AgenticPWorksetLease,
        extend_tokens: int,
        *,
        final_prompt_chunk: bool,
    ) -> torch.Tensor:
        """Transfer suffix slots to one Prefill chunk.

        The returned tensor has the length expected by SGLang's extend
        batch.  On the final logical prompt chunk, ownership of the whole
        final KV page (including unused padding slots) moves to the request,
        so the broker drops the lease without freeing that padding.
        """

        extend_tokens = int(extend_tokens)
        if extend_tokens < 0:
            raise ValueError("extend_tokens must be non-negative")
        with self._lock:
            current = self._leases.get(lease.snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                raise RuntimeError(f"workset lease disappeared for {lease.snapshot_id}")
            if current.state != "handed":
                raise RuntimeError(
                    f"workset lease is {current.state} for {lease.snapshot_id}"
                )
            start = current.suffix_cursor
            end = start + extend_tokens
            logical_suffix_tokens = current.prompt_tokens - current.parent_tokens
            physical_suffix_tokens = current.suffix_allocated_tokens
            if end > physical_suffix_tokens:
                raise RuntimeError(
                    f"workset suffix over-consumed for {lease.snapshot_id}: "
                    f"end={end} physical={physical_suffix_tokens}"
                )
            if final_prompt_chunk and end < logical_suffix_tokens:
                raise RuntimeError(
                    f"final workset chunk is incomplete for {lease.snapshot_id}: "
                    f"end={end} logical={logical_suffix_tokens}"
                )
            if (
                not final_prompt_chunk
                and end < physical_suffix_tokens
                and end % self.page_size
            ):
                raise RuntimeError(
                    "non-final chunked Prefill must end on a KV page boundary"
                )
            indices = current.suffix_indices[start:end]
            current.suffix_cursor = end
            if final_prompt_chunk or end == physical_suffix_tokens:
                current.state = "consumed"
                if current.controller_plan is not None:
                    self._controller_retired.add(current.controller_plan.key)
                self._leases.pop(lease.snapshot_id, None)
                if self._tp_plan_epoch >= 0:
                    self._tp_handoff_committed.add(lease.snapshot_id)
            return indices

    def commit_tp_handoff(self, snapshot_id: str) -> bool:
        """Finish a TP-wide broker-to-Req ownership transfer.

        This intentionally does not call the allocator.  Native TP scheduling
        must consume the same final suffix on every rank before this command;
        therefore a remaining local lease is a rank split and fails closed.
        """

        with self._lock:
            current = self._leases.get(snapshot_id)
            if current is not None:
                return False
            self._intents.pop(snapshot_id, None)
            self._intent_requested_at.pop(snapshot_id, None)
            self._tp_cancel_pending.pop(snapshot_id, None)
            self._tp_release_pending.pop(snapshot_id, None)
            self._tp_handoff_committed.discard(snapshot_id)
            return True

    def begin_bind(self, snapshot_id: str, lease: AgenticPWorksetLease) -> bool:
        """Atomically transfer a completed I/O lease to scheduler binding."""

        with self._lock:
            if (
                snapshot_id in self._tp_cancel_pending
                or snapshot_id in self._tp_release_pending
                or snapshot_id in self._tp_retire_requested
            ):
                return False
            current = self._leases.get(snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                return False
            if current.parent_tokens == 0:
                return False
            if current.state != "active":
                return current.state == "binding"
            current.state = "binding"
            return True

    def begin_io_attempt(
        self,
        snapshot_id: str,
        lease: AgenticPWorksetLease,
        attempt: str,
    ) -> bool:
        """Exclusively reserve one lease for one concrete Direct session."""

        with self._lock:
            if (
                snapshot_id in self._tp_cancel_pending
                or snapshot_id in self._tp_release_pending
                or snapshot_id in self._tp_retire_requested
            ):
                return False
            current = self._leases.get(snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                return False
            if current.parent_tokens == 0:
                return False
            if current.state != "active":
                return False
            current.state = "io_reserved"
            current.io_attempt = str(attempt)
            return True

    def mark_io_inflight(
        self,
        snapshot_id: str,
        lease: AgenticPWorksetLease,
        attempt: str,
    ) -> None:
        """Fence allocator reuse after this attempt publishes destinations."""

        if not self.try_mark_io_inflight(snapshot_id, lease, attempt):
            raise RuntimeError(f"cannot publish I/O for retiring workset {snapshot_id}")

    def try_mark_io_inflight(
        self,
        snapshot_id: str,
        lease: AgenticPWorksetLease,
        attempt: str,
    ) -> bool:
        """Atomically arbitrate submission versus retirement of this attempt.

        False means cancellation won, not an I/O failure. Once submission wins,
        retirement must retain the lease until this attempt's physical fence.
        No transport operation runs under the allocator lock.
        """

        with self._lock:
            current = self._leases.get(snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                raise RuntimeError(f"workset lease disappeared for {snapshot_id}")
            if current.state != "io_reserved" or current.io_attempt != attempt:
                raise RuntimeError(
                    f"cannot start attempt={attempt} on {current.state} "
                    f"workset {snapshot_id} owned_by={current.io_attempt}"
                )
            if (
                snapshot_id in self._tp_cancel_pending
                or snapshot_id in self._tp_release_pending
                or snapshot_id in self._tp_retire_requested
            ):
                return False
            current.state = "io_inflight"
            return True

    def cancel_io_attempt(
        self,
        snapshot_id: str,
        lease: AgenticPWorksetLease,
        attempt: str,
    ) -> bool:
        """Drop an exclusive attempt before any remote write can begin."""

        with self._lock:
            current = self._leases.get(snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                return False
            if current.state != "io_reserved" or current.io_attempt != attempt:
                return False
            current.state = "active"
            current.io_attempt = None
            return True

    def mark_io_quiesced(
        self,
        snapshot_id: str,
        lease: AgenticPWorksetLease,
        attempt: str,
    ) -> bool:
        """Publish a definitive transport terminal state to the allocator."""

        with self._lock:
            current = self._leases.get(snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                return False
            if current.io_attempt != attempt:
                return False
            if current.state == "io_inflight":
                current.state = "active"
                current.io_attempt = None
                return True
            if current.state == "release_pending":
                current.io_attempt = None
                if snapshot_id in self._tp_retire_requested:
                    current.state = "retire_ready"
                    self._tp_release_pending[snapshot_id] = current.lease_id
                else:
                    current.state = "releasing"
                    self._release_requested[snapshot_id] = current.lease_id
                return True
            return False

    def commit_parent_bound(
        self, snapshot_id: str, lease: AgenticPWorksetLease, *,
        state_donated_to_radix: bool = False, state_duplicate: bool = False,
    ) -> None:
        if state_donated_to_radix and state_duplicate:
            raise ValueError("Mamba state cannot be both donated and duplicate")
        with self._lock:
            current = self._leases.get(snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                raise RuntimeError(f"workset lease disappeared for {snapshot_id}")
            if current.state != "binding":
                raise RuntimeError(
                    f"cannot bind {current.state} workset lease for {snapshot_id}"
                )
            if current.state_device_indices:
                if not (state_donated_to_radix or state_duplicate):
                    raise RuntimeError("hybrid parent bind did not settle state ownership")
                if state_duplicate:
                    for pool, indices in zip(self._state_allocators, current.state_device_indices):
                        pool.free(indices)
                current.state_device_indices = ()
            current.parent_bound = True

    def abort_bind(
        self,
        snapshot_id: str,
        lease: AgenticPWorksetLease,
        *,
        parent_bound: bool,
    ) -> bool:
        """Release a scheduler-owned bind after its Radix mutation is undone."""

        with self._lock:
            current = self._leases.get(snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                return False
            if current.state != "binding":
                return False
            attached_req = current.runtime_state_req
            if attached_req is not None:
                attached_req.mamba_pool_idx = None
                attached_req.mamba_ping_pong_track_buffer = None
                attached_req.mamba_next_track_idx = None
                attached_req.mamba_cow_src_index = None
                attached_req._agentic_mamba_prefill_checkpoint = None
                if hasattr(attached_req, "_agentic_mamba_cow_dst_indices"):
                    delattr(attached_req, "_agentic_mamba_cow_dst_indices")
                current.runtime_state_req = None
            if self._tp_plan_epoch >= 0:
                # A bind failure is a TP-group retirement.  It must be
                # broadcast and committed at one scheduler-safe boundary;
                # no rank rolls back pages independently.
                current.parent_bound = bool(parent_bound)
                current.state = "retire_ready"
                self._tp_release_pending[snapshot_id] = current.lease_id
                self._tp_retire_requested.add(snapshot_id)
                return True
            current.parent_bound = bool(parent_bound)
            current.state = "releasing"
            self._release_requested[snapshot_id] = current.lease_id
            return True

    def cancel_unstarted(
        self, snapshot_id: str, *, owner: Optional[str] = None
    ) -> bool:
        """Atomically cancel work that has not started physical I/O.

        The scheduler can grant an intent between two progress-worker passes.
        Cancellation must therefore cover both representations: a pending
        intent and an ``active`` lease.  Once I/O, binding, or request handoff
        begins, the corresponding owner-specific terminal path is solely
        responsible for release.
        """

        with self._lock:
            cancelled = False
            pending = self._intents.get(snapshot_id)
            lease = self._leases.get(snapshot_id)
            planned = any(entry[0] == snapshot_id for entry in self._tp_plan)
            planned_owner_matches = (
                snapshot_id not in self._tp_retired_in_epoch
                and any(
                    entry[0] == snapshot_id
                    and (owner is None or entry[1] == owner)
                    for entry in self._tp_plan
                )
            )
            if planned and (
                planned_owner_matches
                or (
                    pending is not None
                    and (owner is None or pending[0] == owner)
                )
                or (
                    lease is not None
                    and lease.state == "active"
                    and (owner is None or lease.owner == owner)
                )
            ):
                # The current TP epoch is immutable.  Rank0 will omit this
                # entry from its next plan; followers learn the same removal
                # from the native broadcast.  Until then all ranks retain the
                # exact same physical ownership.
                self._tp_cancel_pending[snapshot_id] = owner
                self._tp_retire_requested.add(snapshot_id)
                return True
            if pending is not None and (owner is None or pending[0] == owner):
                self._intents.pop(snapshot_id, None)
                self._intent_requested_at.pop(snapshot_id, None)
                cancelled = True
            if (
                lease is not None
                and lease.state == "active"
                and (owner is None or lease.owner == owner)
            ):
                lease.state = "releasing"
                self._release_requested[snapshot_id] = lease.lease_id
                cancelled = True
            return cancelled

    def supersede_unstarted(self, snapshot_id: str, *, owner: str) -> bool:
        """Permanently supersede an unstarted owner for this generation.

        Unlike a one-shot cancellation, this also rejects a late intent from
        the same owner.  Posted I/O is deliberately left to its transport
        terminal path.
        """

        with self._lock:
            self._superseded_owners.add((snapshot_id, owner))
            return self.cancel_unstarted(snapshot_id, owner=owner)

    def owner_is_superseded(self, snapshot_id: str, *, owner: str) -> bool:
        with self._lock:
            return (snapshot_id, owner) in self._superseded_owners

    def cancel_direct_before_bind(
        self, snapshot_id: str, lease: Optional[AgenticPWorksetLease],
        *, require_quiesced: bool = False,
    ) -> bool:
        """Close an exact unbound Direct owner, without releasing its pages.

        This shares the begin_bind/begin_io_attempt lock: either native bind
        wins and only native rollback may finish, or this permanent tombstone
        wins and there cannot be any Radix work to roll back. A reserved claim
        and posted DMA still need their original control/physical fences.
        """
        with self._lock:
            owner = self.direct_owner(snapshot_id)
            if snapshot_id in self._tp_handoff_committed:
                return False
            current = self._leases.get(snapshot_id)
            if current is not lease or (current is not None and current.owner != owner):
                return False
            pending = self._intents.get(snapshot_id)
            if pending is not None and pending[0] != owner:
                return False
            if any(sid == snapshot_id and planned_owner != owner
                   for sid, planned_owner, *_ in self._tp_plan):
                return False
            if current is not None and current.state not in {
                "active", "io_reserved", "io_inflight", "release_pending",
                "retire_ready", "releasing",
            }:
                return False
            self._superseded_owners.add((snapshot_id, owner))
            self.cancel_unstarted(snapshot_id, owner=owner)
            # Keep allocator retirement on the original native all-rank path.
            # A completed commit must not reopen the old epoch's tombstone.
            if snapshot_id not in self._tp_retired_in_epoch:
                self._prepare_tp_retire_locked(snapshot_id)
            return not require_quiesced or current is None or (
                current.io_attempt is None
                and current.state in {"active", "retire_ready", "releasing"}
            )

    @property
    def tp_retire_candidates(self) -> tuple[str, ...]:
        """Return TP0's locally requested group terminal transitions."""

        with self._lock:
            return tuple(self._tp_retire_requested)

    def prepare_tp_retire(self, snapshot_id: str) -> bool:
        """Install a group tombstone and report whether local pages are safe."""

        with self._lock:
            return self._prepare_tp_retire_locked(snapshot_id)

    def _prepare_tp_retire_locked(self, snapshot_id: str) -> bool:
        """Materialize one TP tombstone while ``self._lock`` is held."""

        self._tp_retire_requested.add(snapshot_id)
        lease = self._leases.get(snapshot_id)
        if lease is None:
            return True
        if lease.state in {"io_reserved", "io_inflight"}:
            # Preserve reserved vs posted: only the latter needs a physical
            # transport fence. The exact unposted owner still drains its RPC
            # before cancel_io_attempt can make this rank retirement-ready.
            if lease.state == "io_inflight":
                lease.state = "release_pending"
            self._tp_release_pending[snapshot_id] = lease.lease_id
            return False
        return lease.state in {"active", "retire_ready", "releasing"}

    def tp_retire_ready(
        self, snapshot_id: str, *, lease_id: Optional[int] = None
    ) -> bool:
        with self._lock:
            lease = self._leases.get(snapshot_id)
            return lease is None or (
                lease_id is not None and lease.lease_id != lease_id
            ) or lease.state in {
                "active",
                "retire_ready",
                "releasing",
            }

    def commit_tp_retire(self, snapshot_id: str) -> bool:
        """Commit one all-rank-safe terminal at the scheduler boundary."""

        with self._lock:
            if snapshot_id not in self._tp_retire_requested:
                return False
            lease = self._leases.get(snapshot_id)
            if lease is not None:
                if lease.state not in {"active", "retire_ready", "releasing"}:
                    return False
                if lease.state != "releasing":
                    lease.state = "releasing"
                    lease.io_attempt = None
                    self._release_requested[snapshot_id] = lease.lease_id
            # All-rank commit is terminal for this frozen plan, whether its
            # lease is awaiting physical free or was never materialized. An
            # abort later in this same control envelope must not re-open its
            # tombstone, nor may service recreate it before the next epoch.
            self._tp_retired_in_epoch.add(snapshot_id)
            self._intents.pop(snapshot_id, None)
            self._intent_requested_at.pop(snapshot_id, None)
            self._tp_cancel_pending.pop(snapshot_id, None)
            self._tp_release_pending.pop(snapshot_id, None)
            self._tp_retire_requested.discard(snapshot_id)
            return True

    @property
    def leased_tokens(self) -> int:
        with self._lock:
            return sum(lease.allocated_tokens for lease in self._leases.values())

    @property
    def stats(self) -> Tuple[int, int, int]:
        """Return pending intents, grants, and allocation misses."""

        with self._lock:
            return len(self._intents), self._grants, self._allocation_failures

    @property
    def lease_state_summary(self) -> str:
        """Compact count/token ownership summary for progress diagnostics."""

        with self._lock:
            counts: Dict[str, int] = {}
            tokens: Dict[str, int] = {}
            for lease in self._leases.values():
                counts[lease.state] = counts.get(lease.state, 0) + 1
                tokens[lease.state] = tokens.get(lease.state, 0) + int(
                    lease.allocated_tokens
                )
            return ",".join(
                f"{state}:{counts[state]}/{tokens[state]}"
                for state in sorted(counts)
            ) or "empty"

    @property
    def active_lease_summary(self) -> str:
        """Identify unstarted leases when progress stops making forward progress."""

        with self._lock:
            active = [
                f"{lease.snapshot_id}@{lease.owner}/{lease.allocated_tokens}"
                for lease in self._leases.values()
                if lease.state == "active"
            ]
            return ";".join(active) or "empty"

    def eviction_blocker(self, snapshot_id: str) -> Optional[str]:
        """Describe live ownership that makes Host eviction illegal.

        A request-generation may be evicted only while Host is its sole
        owner.  Pending intents own no physical pages and releasing leases are
        already fenced for allocator cleanup; every other lease state means
        the snapshot is claimed by recovery, I/O, Radix binding, or a live
        Prefill request and must remain invisible to the evictor.
        """

        with self._lock:
            lease = self._leases.get(snapshot_id)
            if lease is None or lease.state in {"releasing", "consumed"}:
                return None
            return (
                f"id={lease.lease_id} owner={lease.owner} "
                f"state={lease.state} tokens={lease.allocated_tokens}"
            )

    @property
    def unaccounted_tokens(self) -> int:
        """Lease pages not already represented by a bound Radix parent."""

        with self._lock:
            return sum(
                (
                    lease.remaining_suffix_indices.numel()
                    if lease.parent_bound or lease.parent_tokens == 0
                    else lease.allocated_tokens
                )
                for lease in self._leases.values()
            )
