from __future__ import annotations

"""Minimal D scheduler bridge for the multi-node V2 memory authority.

The controller reserves and imports a complete decode input before publishing
``decode-ready``.  The scheduler only adopts ready requests and materializes
future decode pages through the same authority.  No transport polling or
network protocol is implemented here.
"""

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


@dataclass(frozen=True)
class DecodeReadyItem:
    ready_sequence: int
    lease: PhysicalMemoryLease
    req: Any
    binding: Any


@dataclass(frozen=True)
class DecodeCompleteItem:
    completion_sequence: int
    lease: PhysicalMemoryLease
    req: Any


@dataclass
class _BoundDecode:
    req: Any
    binding: Any
    release_handler: Callable[[PhysicalMemoryLease], None]


class AgenticDMemorySchedulerBridge:
    """One rank-local D controller/scheduler ownership bridge."""

    READY_QUEUE = "decode-ready"

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
        self._bound: Dict[int, _BoundDecode] = {}
        self._complete: Deque[DecodeCompleteItem] = deque()
        self._complete_by_lease: Dict[int, DecodeCompleteItem] = {}
        self._next_complete_sequence = itertools.count(1)
        self._completion_sink: Optional[Callable[[DecodeCompleteItem], None]] = None
        self._final_release_sink: Optional[
            Callable[[RequestGenerationAttempt, str], None]
        ] = None
        self._forward_fences: Dict[int, Any] = {}
        self._deferred_final_releases: Dict[
            int, Tuple[Any, str, RequestGenerationAttempt]
        ] = {}

    def install_completion_sink(
        self, sink: Optional[Callable[[DecodeCompleteItem], None]]
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
                raise RuntimeError("a Decode completion sink is already installed")
            self._completion_sink = sink

    def install_final_release_sink(
        self,
        sink: Optional[Callable[[RequestGenerationAttempt, str], None]],
    ) -> None:
        """Install the lifecycle-GC edge for true terminal generations."""

        if sink is not None and not callable(sink):
            raise TypeError("final release sink must be callable")
        with self._lock:
            self._final_release_sink = sink

    def native_guard(self, owner: str):
        return self.authority.native_guard(owner)

    def record_forward_fence(self, reqs: Sequence[Any], event: Any) -> None:
        """Associate a submitted overlapped Decode Forward with its leases."""

        if event is None:
            return
        with self._lock:
            for req in reqs:
                if getattr(req, "_agentic_d_memory_bridge", None) is not self:
                    continue
                lease = getattr(req, "_agentic_d_memory_lease", None)
                if lease is not None:
                    self._forward_fences[int(lease.lease_id)] = event

    def wait_forward_fence(self, lease_id: int) -> None:
        """Wait in the path worker before reading source pages."""

        event = self.forward_fence(lease_id)
        if event is not None:
            event.synchronize()

    def forward_fence(self, lease_id: int) -> Any:
        """Return the immutable CUDA fence without blocking TP control."""

        with self._lock:
            return self._forward_fences.get(int(lease_id))

    @staticmethod
    def _event_complete(event: Any) -> bool:
        if event is None:
            return True
        query = getattr(event, "query", None)
        return bool(query()) if callable(query) else False

    def progress_forward_releases(self, *, max_items: int = 64) -> int:
        """Finish application-terminal releases without blocking overlap."""

        ready = []
        with self._lock:
            for lease_id, item in self._deferred_final_releases.items():
                if len(ready) >= int(max_items):
                    break
                if self._event_complete(self._forward_fences.get(lease_id)):
                    ready.append((lease_id, *item))
        released = 0
        for lease_id, req, reason, key in ready:
            if not self.authority.commit_release(lease_id, reason=reason):
                continue
            with self._lock:
                self._deferred_final_releases.pop(lease_id, None)
                self._forward_fences.pop(lease_id, None)
                self._bound.pop(lease_id, None)
                sink = self._final_release_sink
            for name in ("_agentic_d_memory_lease", "_agentic_d_memory_bridge"):
                if hasattr(req, name):
                    delattr(req, name)
            if sink is not None:
                sink(key, reason)
            released += 1
        return released

    def reserve_decode(
        self,
        key: RequestGenerationAttempt,
        *,
        owner: str,
        prompt_tokens: int,
        decode_growth_tokens: int,
        state_slot_counts: Optional[Sequence[int]] = None,
    ) -> Optional[PhysicalMemoryLease]:
        counts = (
            self.default_state_slot_counts
            if state_slot_counts is None
            else tuple(int(value) for value in state_slot_counts)
        )
        return self.authority.reserve_decode(
            key,
            owner=owner,
            prompt_tokens=prompt_tokens,
            decode_growth_tokens=decode_growth_tokens,
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
        bind_prompt: Callable[[PhysicalMemoryLease, Any], Any],
        release_bound: Callable[[PhysicalMemoryLease, Any, Any], None],
    ) -> Optional[MemoryReadyEvent]:
        """Commit imported pages to one Req, then publish a ready edge."""

        self.bind_decode(
            lease,
            req,
            bind_prompt=bind_prompt,
            release_bound=release_bound,
        )
        return self.publish_bound(lease, req)

    def bind_decode(
        self,
        lease: PhysicalMemoryLease,
        req: Any,
        *,
        bind_prompt: Callable[[PhysicalMemoryLease, Any], Any],
        release_bound: Callable[[PhysicalMemoryLease, Any, Any], None],
        release_unbound: Optional[
            Callable[[PhysicalMemoryLease, Any, Any], None]
        ] = None,
    ) -> Any:
        """Prepare local imported request without exposing decode-ready."""

        with self._lock:
            current = self._bound.get(lease.lease_id)
            if current is not None:
                if current.req is not req:
                    raise RuntimeError("decode lease was bound to another request")
                return current.binding

            def transaction():
                binding_holder = {"value": None}

                def release_handler(bound_lease: PhysicalMemoryLease) -> None:
                    binding = binding_holder["value"]
                    if binding is None and release_unbound is not None:
                        release_unbound(bound_lease, req, binding)
                    else:
                        release_bound(bound_lease, req, binding)

                if not self.authority.install_release_handler(
                    lease.lease_id, release_handler
                ):
                    raise RuntimeError("failed to install decode cleanup")
                binding = bind_prompt(lease, req)
                binding_holder["value"] = binding
                return binding, release_handler

            binding, release_handler = self.authority.run_memory_transaction(
                transaction
            )
            self._bound[lease.lease_id] = _BoundDecode(
                req=req,
                binding=binding,
                release_handler=release_handler,
            )
            return binding

    def publish_bound(
        self, lease: PhysicalMemoryLease, req: Any
    ) -> Optional[MemoryReadyEvent]:
        """Publish only after all TP ranks reached the HANDOFF barrier."""

        with self._lock:
            current = self._bound.get(lease.lease_id)
            if current is None or current.req is not req:
                raise RuntimeError("cannot publish an unbound Decode request")
        return self.authority.publish_ready(lease.lease_id, self.READY_QUEUE)

    def take_decode_ready(
        self, *, max_items: int = 64, timeout: Optional[float] = None
    ) -> Tuple[DecodeReadyItem, ...]:
        events = self.authority.take_ready(
            self.READY_QUEUE, max_items=max_items, timeout=timeout
        )
        out = []
        with self._lock:
            for event in events:
                bound = self._bound.get(event.lease.lease_id)
                if bound is None:
                    raise RuntimeError("ready decode lease has no bound request")
                out.append(
                    DecodeReadyItem(
                        ready_sequence=event.ready_sequence,
                        lease=event.lease,
                        req=bound.req,
                        binding=bound.binding,
                    )
                )
        return tuple(out)

    def take_decode_ready_attempts(
        self, attempts: Sequence[RequestGenerationAttempt]
    ) -> Tuple[DecodeReadyItem, ...]:
        """Resolve TP0's ordered activation ticket against local staging."""

        events = self.authority.take_ready_attempts(self.READY_QUEUE, attempts)
        out = []
        with self._lock:
            for event in events:
                bound = self._bound.get(event.lease.lease_id)
                if bound is None:
                    raise RuntimeError("activated decode lease has no bound request")
                out.append(
                    DecodeReadyItem(
                        ready_sequence=event.ready_sequence,
                        lease=event.lease,
                        req=bound.req,
                        binding=bound.binding,
                    )
                )
        return tuple(out)

    def adopt_for_decode(self, item: DecodeReadyItem) -> bool:
        if not self.authority.adopt_ready(item.lease.lease_id):
            return False
        req = item.req
        req._agentic_d_memory_lease = item.lease
        req._agentic_d_memory_bridge = self
        req._agentic_memory_ready_sequence = item.ready_sequence
        return True

    def drain_decode_ready(self, *, max_items: int = 64) -> Tuple[Any, ...]:
        """The sole non-blocking D scheduler operation."""

        reqs = []
        for item in self.take_decode_ready(max_items=max_items, timeout=None):
            if not self.adopt_for_decode(item):
                raise RuntimeError(
                    f"failed to adopt decode lease={item.lease.lease_id}"
                )
            reqs.append(item.req)
        return tuple(reqs)

    def activate_decode_attempts(
        self, attempts: Sequence[RequestGenerationAttempt]
    ) -> Tuple[Any, ...]:
        """Adopt only the exact attempts broadcast by this endpoint's TP0."""

        reqs = []
        for item in self.take_decode_ready_attempts(attempts):
            if not self.adopt_for_decode(item):
                raise RuntimeError(
                    f"failed to adopt activated decode lease={item.lease.lease_id}"
                )
            reqs.append(item.req)
        return tuple(reqs)

    def allocate_decode_growth(
        self,
        reqs: Sequence[Any],
        *,
        seq_lens_next,
        allocate: Callable[[Any], Any],
    ) -> Any:
        """Run one native paged-decode allocation under this authority."""

        charges = self._decode_growth_charges(reqs, seq_lens_next)
        return self.authority.run_decode_growth_batch(charges, allocate)

    def ensure_decode_growth_headroom(
        self,
        reqs: Sequence[Any],
        *,
        seq_lens_next,
    ) -> bool:
        """Renew imminent decode pages before native OOM/retraction checks."""

        charges = self._decode_growth_charges(reqs, seq_lens_next)
        return self.authority.ensure_decode_growth_credit(charges)

    def _decode_growth_charges(
        self,
        reqs: Sequence[Any],
        seq_lens_next,
    ) -> Dict[int, int]:
        if len(reqs) != len(seq_lens_next):
            raise ValueError("decode request and sequence length counts differ")
        charges: Dict[int, int] = {}
        page_size = self.authority.page_size
        for req, seq_len_next in zip(reqs, seq_lens_next):
            if getattr(req, "_agentic_d_memory_bridge", None) is not self:
                raise RuntimeError("mixed or foreign decode memory authority")
            lease = getattr(req, "_agentic_d_memory_lease", None)
            if lease is None:
                raise RuntimeError("decode request is missing its memory lease")
            if lease.lease_id in charges:
                raise RuntimeError("decode batch contains a duplicate memory lease")
            # This matches paged allocator get_num_new_pages(..., decode=True).
            charge = page_size if int(seq_len_next) % page_size == 1 else 0
            charges[lease.lease_id] = charge
        return charges

    def finish_decode(self, req: Any) -> bool:
        lease = getattr(req, "_agentic_d_memory_lease", None)
        if lease is None:
            return False
        return self.authority.finish_compute(lease.lease_id)

    def publish_decode_complete(self, req: Any) -> Optional[DecodeCompleteItem]:
        """Hand one non-terminal generation to the independent D->P controller."""

        lease = getattr(req, "_agentic_d_memory_lease", None)
        if lease is None or getattr(req, "_agentic_d_memory_bridge", None) is not self:
            return None
        with self._complete_condition:
            existing = self._complete_by_lease.get(lease.lease_id)
            if existing is not None:
                if existing.req is not req or existing.lease.key != lease.key:
                    raise RuntimeError("decode completion attempt identity changed")
                return existing
            with self.authority.native_guard("d-scheduler-complete"):
                if not self.authority.finish_compute(lease.lease_id):
                    return None
            event = DecodeCompleteItem(
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

    def take_decode_complete(
        self, *, max_items: int = 64, timeout: Optional[float] = None
    ) -> Tuple[DecodeCompleteItem, ...]:
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

    def drain_decode_complete(
        self, *, max_items: int = 64
    ) -> Tuple[DecodeCompleteItem, ...]:
        return self.take_decode_complete(max_items=max_items, timeout=None)

    def release_final(self, req: Any, *, reason: str = "application_final") -> bool:
        """Release a true final/abort without publishing it to D->P."""

        lease = getattr(req, "_agentic_d_memory_lease", None)
        if lease is None or getattr(req, "_agentic_d_memory_bridge", None) is not self:
            return False
        with self.authority.native_guard("d-scheduler-final-release"):
            if not self.authority.finish_compute(lease.lease_id):
                return False
            if not self.authority.request_release(lease.lease_id):
                return False
            with self._lock:
                fence = self._forward_fences.get(int(lease.lease_id))
            if not self._event_complete(fence):
                with self._lock:
                    self._deferred_final_releases[int(lease.lease_id)] = (
                        req,
                        str(reason),
                        lease.key,
                    )
                return True
            released = self.authority.commit_release(
                lease.lease_id, reason=reason
            )
        if released:
            with self._lock:
                self._bound.pop(lease.lease_id, None)
                self._complete_by_lease.pop(lease.lease_id, None)
                self._forward_fences.pop(int(lease.lease_id), None)
                sink = self._final_release_sink
            for name in ("_agentic_d_memory_lease", "_agentic_d_memory_bridge"):
                if hasattr(req, name):
                    delattr(req, name)
            if sink is not None:
                sink(lease.key, str(reason))
        return released

    def begin_egress(self, req: Any, io_attempt: str) -> bool:
        lease = getattr(req, "_agentic_d_memory_lease", None)
        return bool(
            lease is not None
            and self.authority.begin_io(lease.lease_id, io_attempt)
        )

    def complete_egress(
        self, req: Any, io_attempt: str, *, success: bool
    ) -> bool:
        lease = getattr(req, "_agentic_d_memory_lease", None)
        return bool(
            lease is not None
            and self.authority.complete_io(
                lease.lease_id, io_attempt, success=success
            )
        )

    def release_after_group_fence(self, lease_id: int, *, reason: str) -> bool:
        """Release only after rank 0 observes every rank's egress terminal."""

        if not self.authority.request_release(lease_id):
            return False
        released = self.authority.commit_release(lease_id, reason=reason)
        if released:
            with self._lock:
                self._bound.pop(int(lease_id), None)
                self._complete_by_lease.pop(int(lease_id), None)
                self._forward_fences.pop(int(lease_id), None)
        return released
