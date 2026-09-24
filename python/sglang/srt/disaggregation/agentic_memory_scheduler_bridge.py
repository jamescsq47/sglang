from __future__ import annotations

"""Minimal P scheduler bridge for the multi-node V2 memory authority.

Transport controllers reserve and fill a complete workset, then perform a
short Radix bind through this bridge.  The P scheduler only drains
``prefill-ready`` events and adopts the immutable lease; it never polls a
network endpoint or allocates the same workset again.

The module is opt-in and has no import-time effect on the pd_mamba path.
"""

import os
import threading
import itertools
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Optional, Sequence, Tuple

from sglang.srt.disaggregation.agentic_memory_authority import (
    AgenticMemoryAuthority,
    MemoryReadyEvent,
    PhysicalMemoryLease,
    RequestGenerationAttempt,
)


def memory_authority_v2_enabled() -> bool:
    return os.getenv("SGLANG_AGENTIC_MEMORY_AUTHORITY_V2", "0").lower() in {
        "1",
        "true",
    }


@dataclass(frozen=True)
class PrefillReadyItem:
    ready_sequence: int
    lease: PhysicalMemoryLease
    req: Any
    binding: Any


@dataclass(frozen=True)
class PrefillCompleteItem:
    completion_sequence: int
    lease: PhysicalMemoryLease
    req: Any


@dataclass
class _BoundWorkset:
    req: Any
    binding: Any
    release_handler: Callable[[PhysicalMemoryLease], None]


class AgenticPMemorySchedulerBridge:
    """Connect a P memory controller to a compute-only scheduler queue."""

    READY_QUEUE = "prefill-ready"

    def __init__(
        self,
        authority: AgenticMemoryAuthority,
        *,
        default_state_slot_counts: Sequence[int] = (),
    ) -> None:
        self.authority = authority
        self.default_state_slot_counts = tuple(
            int(value) for value in default_state_slot_counts
        )
        self._lock = threading.RLock()
        self._complete_condition = threading.Condition(self._lock)
        self._bound: Dict[int, _BoundWorkset] = {}
        self._complete: Deque[PrefillCompleteItem] = deque()
        self._complete_by_lease: Dict[int, PrefillCompleteItem] = {}
        self._next_complete_sequence = itertools.count(1)
        self._completion_sink: Optional[Callable[[PrefillCompleteItem], None]] = None
        self._final_release_sink: Optional[
            Callable[[RequestGenerationAttempt, str], None]
        ] = None

    def install_completion_sink(
        self, sink: Optional[Callable[[PrefillCompleteItem], None]]
    ) -> None:
        """Install the controller's edge sink; never a scheduler poll hook."""

        if sink is not None and not callable(sink):
            raise TypeError("completion sink must be callable")
        with self._complete_condition:
            if (
                sink is not None
                and self._completion_sink is not None
                and self._completion_sink is not sink
            ):
                raise RuntimeError("a Prefill completion sink is already installed")
            self._completion_sink = sink

    def install_final_release_sink(
        self,
        sink: Optional[Callable[[RequestGenerationAttempt, str], None]],
    ) -> None:
        if sink is not None and not callable(sink):
            raise TypeError("final release sink must be callable")
        with self._lock:
            self._final_release_sink = sink

    def native_guard(self, owner: str):
        return self.authority.native_guard(owner)

    def reserve_workset(
        self,
        key: RequestGenerationAttempt,
        *,
        owner: str,
        parent_tokens: int,
        prompt_tokens: int,
        state_slot_counts: Optional[Sequence[int]] = None,
    ) -> Optional[PhysicalMemoryLease]:
        counts = (
            self.default_state_slot_counts
            if state_slot_counts is None
            else tuple(int(value) for value in state_slot_counts)
        )
        return self.authority.reserve_prefill_workset(
            key,
            owner=owner,
            parent_tokens=parent_tokens,
            prompt_tokens=prompt_tokens,
            state_slot_counts=counts,
        )

    def begin_ingress(self, lease: PhysicalMemoryLease, io_attempt: str) -> bool:
        return self.authority.begin_io(lease.lease_id, io_attempt)

    def complete_ingress(
        self, lease: PhysicalMemoryLease, io_attempt: str, *, success: bool
    ) -> bool:
        return self.authority.complete_io(
            lease.lease_id, io_attempt, success=success
        )

    def bind_and_publish(
        self,
        lease: PhysicalMemoryLease,
        req: Any,
        *,
        bind_parent: Callable[[PhysicalMemoryLease, Any], Any],
        release_bound: Callable[[PhysicalMemoryLease, Any, Any], None],
    ) -> Optional[MemoryReadyEvent]:
        """Bind one completed parent and publish exactly one ready event.

        ``bind_parent`` and installation of its cleanup handler are one local
        memory transaction.  The callback may mutate Radix and request-pool
        metadata, but may not wait for transport or TP acknowledgements.
        Rank 0 calls this only after all ranks completed the physical ingress.
        """

        self.bind_workset(
            lease,
            req,
            bind_parent=bind_parent,
            release_bound=release_bound,
        )
        return self.publish_bound(lease, req)

    def bind_workset(
        self,
        lease: PhysicalMemoryLease,
        req: Any,
        *,
        bind_parent: Callable[[PhysicalMemoryLease, Any], Any],
        release_bound: Callable[[PhysicalMemoryLease, Any, Any], None],
        release_unbound: Optional[
            Callable[[PhysicalMemoryLease, Any, Any], None]
        ] = None,
    ) -> Any:
        """Prepare local Req/Radix ownership without publishing scheduler ready."""

        with self._lock:
            current = self._bound.get(lease.lease_id)
            if current is not None:
                if current.req is not req:
                    raise RuntimeError("workset was bound to another request")
                return current.binding

            def transaction():
                binding_holder = {"value": None}

                def release_handler(bound_lease: PhysicalMemoryLease) -> None:
                    binding = binding_holder["value"]
                    adopted = bool(
                        getattr(req, "_agentic_workset_backed", False)
                        or getattr(req, "req_pool_idx", None) is not None
                    )
                    if adopted or release_unbound is None:
                        release_bound(bound_lease, req, binding)
                    else:
                        release_unbound(bound_lease, req, binding)

                # Install cleanup before the first native Req/Radix mutation.
                # A failed bind is then retired by the ordinary group-abort
                # fence through the exact same lease identity.
                if not self.authority.install_release_handler(
                    lease.lease_id, release_handler
                ):
                    raise RuntimeError("failed to install bound-workset cleanup")
                binding = bind_parent(lease, req)
                binding_holder["value"] = binding
                return binding, release_handler

            binding, release_handler = self.authority.run_memory_transaction(
                transaction
            )
            self._bound[lease.lease_id] = _BoundWorkset(
                req=req,
                binding=binding,
                release_handler=release_handler,
            )
            return binding

    def publish_bound(
        self, lease: PhysicalMemoryLease, req: Any
    ) -> Optional[MemoryReadyEvent]:
        """Publish only after the TP group HANDOFF barrier."""

        with self._lock:
            current = self._bound.get(lease.lease_id)
            if current is None or current.req is not req:
                raise RuntimeError("cannot publish an unbound P workset")
        return self.authority.publish_ready(lease.lease_id, self.READY_QUEUE)

    def take_prefill_ready(
        self, *, max_items: int = 64, timeout: Optional[float] = None
    ) -> Tuple[PrefillReadyItem, ...]:
        events = self.authority.take_ready(
            self.READY_QUEUE, max_items=max_items, timeout=timeout
        )
        out = []
        with self._lock:
            for event in events:
                bound = self._bound.get(event.lease.lease_id)
                if bound is None:
                    raise RuntimeError("ready workset has no bound request")
                out.append(
                    PrefillReadyItem(
                        ready_sequence=event.ready_sequence,
                        lease=event.lease,
                        req=bound.req,
                        binding=bound.binding,
                    )
                )
        return tuple(out)

    def take_prefill_ready_attempts(
        self, attempts: Sequence[RequestGenerationAttempt]
    ) -> Tuple[PrefillReadyItem, ...]:
        """Resolve TP0's ordered activation ticket against local staging."""

        events = self.authority.take_ready_attempts(self.READY_QUEUE, attempts)
        out = []
        with self._lock:
            for event in events:
                bound = self._bound.get(event.lease.lease_id)
                if bound is None:
                    raise RuntimeError("activated workset has no bound request")
                out.append(
                    PrefillReadyItem(
                        ready_sequence=event.ready_sequence,
                        lease=event.lease,
                        req=bound.req,
                        binding=bound.binding,
                    )
                )
        return tuple(out)

    def drain_prefill_ready(self, *, max_items: int = 64) -> Tuple[Any, ...]:
        """Return compute-ready requests without polling transport state.

        This is the only operation called by the P scheduler.  Allocation,
        ingress completion and Radix binding have already committed before an
        item reaches this edge-triggered queue.
        """

        reqs = []
        for item in self.take_prefill_ready(max_items=max_items, timeout=None):
            if not self.adopt_for_prefill(item):
                raise RuntimeError(
                    f"failed to adopt ready workset lease={item.lease.lease_id}"
                )
            reqs.append(item.req)
        return tuple(reqs)

    def activate_prefill_attempts(
        self, attempts: Sequence[RequestGenerationAttempt]
    ) -> Tuple[Any, ...]:
        """Adopt only the exact attempts broadcast by this endpoint's TP0."""

        reqs = []
        for item in self.take_prefill_ready_attempts(attempts):
            if not self.adopt_for_prefill(item):
                raise RuntimeError(
                    f"failed to adopt activated workset lease={item.lease.lease_id}"
                )
            reqs.append(item.req)
        return tuple(reqs)

    def adopt_for_prefill(self, item: PrefillReadyItem) -> bool:
        """Attach an already-owned workset to Req without capacity admission."""

        lease = item.lease
        if not self.authority.adopt_ready(lease.lease_id):
            return False
        req = item.req
        req._agentic_workset_backed = True
        req._agentic_p_workset_lease = lease
        req._agentic_p_workset_broker = self
        req._agentic_workset_suffix_indices = (
            self.authority.remaining_prefill_suffix(lease.lease_id)
        )
        req._agentic_memory_ready_sequence = item.ready_sequence
        return True

    # Compatibility with mem_cache.common.alloc_for_extend.  This method does
    # not allocate; it only consumes the suffix named by the immutable lease.
    def consume_suffix(
        self,
        lease: PhysicalMemoryLease,
        extend_tokens: int,
        *,
        final_prompt_chunk: bool,
    ):
        return self.authority.consume_prefill_suffix(
            lease.lease_id,
            extend_tokens,
            final_prompt_chunk=final_prompt_chunk,
        )

    def remaining_suffix_indices(self, lease: PhysicalMemoryLease):
        return self.authority.remaining_prefill_suffix(lease.lease_id)

    def finish_prefill(self, req: Any) -> bool:
        lease = getattr(req, "_agentic_p_workset_lease", None)
        if lease is None:
            return False
        return self.authority.finish_compute(lease.lease_id)

    def publish_prefill_complete(
        self, req: Any
    ) -> Optional[PrefillCompleteItem]:
        """Hand one completed P workset to the independent P->D controller."""

        lease = getattr(req, "_agentic_p_workset_lease", None)
        if lease is None or getattr(req, "_agentic_p_workset_broker", None) is not self:
            return None
        with self._complete_condition:
            existing = self._complete_by_lease.get(lease.lease_id)
            if existing is not None:
                if existing.req is not req or existing.lease.key != lease.key:
                    raise RuntimeError("prefill completion attempt identity changed")
                return existing
            with self.authority.native_guard("p-scheduler-complete"):
                if not self.authority.finish_compute(lease.lease_id):
                    return None
            event = PrefillCompleteItem(
                completion_sequence=next(self._next_complete_sequence),
                lease=lease,
                req=req,
            )
            self._complete_by_lease[lease.lease_id] = event
            sink = self._completion_sink
            if sink is None:
                self._complete.append(event)
            self._complete_condition.notify_all()
        if sink is not None:
            sink(event)
        return event

    def take_prefill_complete(
        self, *, max_items: int = 64, timeout: Optional[float] = None
    ) -> Tuple[PrefillCompleteItem, ...]:
        max_items = int(max_items)
        if max_items <= 0:
            return ()
        with self._complete_condition:
            if not self._complete and timeout is not None and timeout > 0:
                self._complete_condition.wait_for(
                    lambda: bool(self._complete), timeout=timeout
                )
            out = []
            while self._complete and len(out) < max_items:
                out.append(self._complete.popleft())
            return tuple(out)

    def drain_prefill_complete(
        self, *, max_items: int = 64
    ) -> Tuple[PrefillCompleteItem, ...]:
        return self.take_prefill_complete(max_items=max_items, timeout=None)

    def release_abort(self, req: Any, *, reason: str) -> bool:
        """Abort one compute-owned workset through the sole memory authority."""

        lease = getattr(req, "_agentic_p_workset_lease", None)
        if lease is None or getattr(req, "_agentic_p_workset_broker", None) is not self:
            return False
        with self.authority.native_guard("p-scheduler-abort-release"):
            if not self.authority.finish_compute(lease.lease_id):
                return False
            if not self.authority.request_release(lease.lease_id):
                return False
            released = self.authority.commit_release(
                lease.lease_id, reason=str(reason)
            )
        if released:
            with self._lock:
                self._bound.pop(lease.lease_id, None)
                self._complete_by_lease.pop(lease.lease_id, None)
                sink = self._final_release_sink
            for name in (
                "_agentic_p_workset_lease",
                "_agentic_p_workset_broker",
                "_agentic_workset_backed",
                "_agentic_workset_suffix_indices",
            ):
                if hasattr(req, name):
                    delattr(req, name)
            if sink is not None:
                sink(lease.key, str(reason))
        return released

    def release_after_group_fence(self, lease_id: int, *, reason: str) -> bool:
        """Release only after rank 0 observed every rank's terminal fence."""

        if not self.authority.request_release(lease_id):
            return False
        released = self.authority.commit_release(lease_id, reason=reason)
        if released:
            with self._lock:
                self._bound.pop(int(lease_id), None)
                self._complete_by_lease.pop(int(lease_id), None)
        return released
