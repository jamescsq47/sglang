from __future__ import annotations

"""Rank-local physical memory authority for agentic PD.

This module deliberately contains no transport, scheduler, or TP policy.  A TP
rank-0 coordinator decides which request-generation attempt is allowed to
proceed; every rank then applies that command to its own authority.  The
authority is the only adapter that mutates the existing SGLang token allocator
and optional recurrent-state pools.

The adapter does *not* maintain another free list.  Physical indices returned
by SGLang remain owned by the underlying allocator and are referenced by one
immutable lease until that exact lease is released.  I/O workers receive only
immutable leases and report completions through :meth:`complete_io`; they never
allocate or free pages themselves.
"""

import enum
import itertools
import threading
from collections import OrderedDict, defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Iterable, Optional, Sequence, Tuple


@dataclass(frozen=True, order=True)
class RequestGenerationAttempt:
    """Identity carried by every memory and transport command."""

    request_id: str
    generation: int
    attempt: int


class LeaseKind(enum.Enum):
    PREFILL_WORKSET = "prefill_workset"
    DECODE_RESERVATION = "decode_reservation"


class LeasePhase(enum.Enum):
    RESERVED = "reserved"
    IO_INFLIGHT = "io_inflight"
    IO_COMPLETE = "io_complete"
    READY = "ready"
    COMPUTE = "compute"
    COMPUTE_COMPLETE = "compute_complete"
    RELEASE_PENDING = "release_pending"


@dataclass(frozen=True)
class PhysicalMemoryLease:
    """Immutable description of pages owned by one local TP shard.

    ``device_indices`` includes the complete P workset for a prefill lease and
    only the imported prompt for a decode lease.  Decode growth is backed by
    one allocator-wide reserve rather than multiplying headroom by the number
    of requests; pages are materialized later through
    :meth:`run_decode_growth`.
    """

    lease_id: int
    key: RequestGenerationAttempt
    owner: str
    kind: LeaseKind
    page_size: int
    parent_tokens: int
    parent_allocated_tokens: int
    prompt_tokens: int
    prompt_allocated_tokens: int
    growth_reserved_tokens: int
    device_indices: Any
    state_indices: Tuple[Any, ...] = ()

    @property
    def parent_indices(self):
        return self.device_indices[: self.parent_allocated_tokens]

    @property
    def suffix_indices(self):
        return self.device_indices[
            self.parent_allocated_tokens : self.prompt_allocated_tokens
        ]


@dataclass(frozen=True)
class MemoryReadyEvent:
    queue_name: str
    ready_sequence: int
    lease: PhysicalMemoryLease


@dataclass
class _LeaseRecord:
    lease: PhysicalMemoryLease
    phase: LeasePhase = LeasePhase.RESERVED
    io_attempt: Optional[str] = None
    io_resume_phase: Optional[LeasePhase] = None
    release_requested: bool = False
    suffix_cursor: int = 0
    ready_event: Optional[MemoryReadyEvent] = None
    release_handler: Optional[Callable[[PhysicalMemoryLease], None]] = None


class AgenticMemoryAuthority:
    """Serialize every agentic mutation of one rank's physical memory pools.

    The class is intentionally synchronous and small.  A future controller can
    own it in a dedicated actor thread, while the native scheduler reaches the
    same underlying allocator only through ``run_native_transaction``.  This
    keeps one free list and one mutation lock without making scheduler progress
    responsible for transport progress.
    """

    def __init__(
        self,
        token_allocator,
        *,
        state_allocators: Sequence[Any] = (),
        decode_growth_reserve_tokens: int = 0,
        terminal_history: int = 4096,
    ) -> None:
        self._token_allocator = token_allocator
        self._state_allocators = tuple(state_allocators)
        self._page_size = int(token_allocator.page_size)
        if self._page_size <= 0:
            raise ValueError("allocator page_size must be positive")
        self._lock = threading.RLock()
        self._guard_local = threading.local()
        self._ready_condition = threading.Condition(self._lock)
        self._leases: Dict[int, _LeaseRecord] = {}
        self._lease_by_key: Dict[RequestGenerationAttempt, int] = {}
        self._ready: Dict[str, Deque[MemoryReadyEvent]] = defaultdict(deque)
        self._next_lease_id = itertools.count(1)
        self._next_ready_sequence = itertools.count(1)
        # Decode growth is protected once per D allocator, not once per
        # request.  Per-request credits made a lightly occupied D appear full
        # after only a handful of P->D admissions (N * max_new_tokens), even
        # though those pages had not been materialized.  The native decode
        # allocator consumes this shared floor as sequences grow; new ingress
        # work may not consume it.
        self._decode_growth_reserve = self._round_tokens(
            int(decode_growth_reserve_tokens)
        )
        self._capacity_available_sink: Optional[Callable[[int], None]] = None
        self._terminal_history_limit = max(0, int(terminal_history))
        self._terminal: OrderedDict[int, str] = OrderedDict()

    @property
    def page_size(self) -> int:
        return self._page_size

    @property
    def token_allocator(self):
        """Underlying allocator for wiring only; callers must not mutate it."""

        return self._token_allocator

    def _round_tokens(self, tokens: int) -> int:
        tokens = int(tokens)
        if tokens < 0:
            raise ValueError("token count must be non-negative")
        if tokens == 0:
            return 0
        return ((tokens + self._page_size - 1) // self._page_size) * self._page_size

    def available_tokens(self) -> int:
        """Capacity visible to new work after the group growth reserve."""

        with self._lock:
            return max(
                0,
                int(self._token_allocator.available_size())
                - self._decode_growth_reserve,
            )

    def install_capacity_available_sink(
        self, sink: Optional[Callable[[int], None]]
    ) -> None:
        """Install one edge sink notified after a real lease release.

        The callback runs after the allocator lock is dropped.  It is a
        best-effort wake-up hint and must not mutate ownership or block the
        allocator transaction that produced the edge.
        """

        if sink is not None and not callable(sink):
            raise TypeError("capacity sink must be callable")
        with self._lock:
            current = self._capacity_available_sink
            if sink is not None and current is not None and current is not sink:
                raise RuntimeError("a capacity-available sink is already installed")
            self._capacity_available_sink = sink

    def _remember_terminal(self, lease_id: int, reason: str) -> None:
        if self._terminal_history_limit == 0:
            return
        self._terminal[int(lease_id)] = str(reason)
        self._terminal.move_to_end(int(lease_id))
        while len(self._terminal) > self._terminal_history_limit:
            self._terminal.popitem(last=False)

    def _allocate_state(self, counts: Sequence[int]) -> Optional[Tuple[Any, ...]]:
        if len(counts) != len(self._state_allocators):
            raise ValueError("state slot counts do not match state allocators")
        allocated = []
        for pool, count in zip(self._state_allocators, counts):
            count = int(count)
            if count < 0:
                raise ValueError("state slot count must be non-negative")
            indices = pool.alloc(count) if count else pool.alloc(0)
            if indices is None:
                for old_pool, old_indices in zip(
                    self._state_allocators, allocated
                ):
                    old_pool.free(old_indices)
                return None
            allocated.append(indices)
        return tuple(allocated)

    def _state_counts(
        self, state_slot_counts: Optional[Sequence[int]]
    ) -> Tuple[int, ...]:
        if state_slot_counts is None:
            if self._state_allocators:
                raise ValueError(
                    "state_slot_counts are required when state allocators exist"
                )
            return ()
        return tuple(int(x) for x in state_slot_counts)

    def _free_state(self, state_indices: Iterable[Any]) -> None:
        for pool, indices in zip(self._state_allocators, state_indices):
            pool.free(indices)

    def reserve_prefill_workset(
        self,
        key: RequestGenerationAttempt,
        *,
        owner: str,
        parent_tokens: int,
        prompt_tokens: int,
        state_slot_counts: Optional[Sequence[int]] = None,
    ) -> Optional[PhysicalMemoryLease]:
        """Atomically reserve parent plus suffix pages for one P request."""

        parent_tokens = int(parent_tokens)
        prompt_tokens = int(prompt_tokens)
        if parent_tokens < 0 or prompt_tokens < parent_tokens:
            raise ValueError("invalid parent/prompt token counts")
        parent_allocated = self._round_tokens(parent_tokens)
        suffix_allocated = self._round_tokens(prompt_tokens - parent_tokens)
        allocated_tokens = parent_allocated + suffix_allocated
        counts = self._state_counts(state_slot_counts)
        with self._lock:
            existing_id = self._lease_by_key.get(key)
            if existing_id is not None:
                existing = self._leases[existing_id].lease
                if (
                    existing.kind is LeaseKind.PREFILL_WORKSET
                    and existing.owner == owner
                    and existing.parent_tokens == parent_tokens
                    and existing.prompt_tokens == prompt_tokens
                ):
                    return existing
                raise RuntimeError(f"attempt already owns another lease: {key}")
            if self.available_tokens() < allocated_tokens:
                return None
            device_indices = self._token_allocator.alloc(allocated_tokens)
            if device_indices is None:
                return None
            state_indices = self._allocate_state(counts)
            if state_indices is None:
                self._token_allocator.free(device_indices)
                return None
            lease = PhysicalMemoryLease(
                lease_id=next(self._next_lease_id),
                key=key,
                owner=str(owner),
                kind=LeaseKind.PREFILL_WORKSET,
                page_size=self._page_size,
                parent_tokens=parent_tokens,
                parent_allocated_tokens=parent_allocated,
                prompt_tokens=prompt_tokens,
                prompt_allocated_tokens=allocated_tokens,
                growth_reserved_tokens=0,
                device_indices=device_indices,
                state_indices=state_indices,
            )
            self._leases[lease.lease_id] = _LeaseRecord(lease=lease)
            self._lease_by_key[key] = lease.lease_id
            return lease

    def reserve_decode(
        self,
        key: RequestGenerationAttempt,
        *,
        owner: str,
        prompt_tokens: int,
        decode_growth_tokens: int,
        state_slot_counts: Optional[Sequence[int]] = None,
    ) -> Optional[PhysicalMemoryLease]:
        """Reserve D import pages while sharing one allocator growth pool."""

        prompt_tokens = int(prompt_tokens)
        decode_growth_tokens = int(decode_growth_tokens)
        if prompt_tokens < 0 or decode_growth_tokens < 0:
            raise ValueError("decode token counts must be non-negative")
        prompt_allocated = self._round_tokens(prompt_tokens)
        counts = self._state_counts(state_slot_counts)
        with self._lock:
            existing_id = self._lease_by_key.get(key)
            if existing_id is not None:
                existing = self._leases[existing_id].lease
                if (
                    existing.kind is LeaseKind.DECODE_RESERVATION
                    and existing.owner == owner
                    and existing.prompt_tokens == prompt_tokens
                ):
                    return existing
                raise RuntimeError(f"attempt already owns another lease: {key}")
            if self.available_tokens() < prompt_allocated:
                return None
            device_indices = self._token_allocator.alloc(prompt_allocated)
            if device_indices is None:
                return None
            state_indices = self._allocate_state(counts)
            if state_indices is None:
                self._token_allocator.free(device_indices)
                return None
            lease = PhysicalMemoryLease(
                lease_id=next(self._next_lease_id),
                key=key,
                owner=str(owner),
                kind=LeaseKind.DECODE_RESERVATION,
                page_size=self._page_size,
                parent_tokens=prompt_tokens,
                parent_allocated_tokens=prompt_allocated,
                prompt_tokens=prompt_tokens,
                prompt_allocated_tokens=prompt_allocated,
                growth_reserved_tokens=0,
                device_indices=device_indices,
                state_indices=state_indices,
            )
            self._leases[lease.lease_id] = _LeaseRecord(lease=lease)
            self._lease_by_key[key] = lease.lease_id
            return lease

    def begin_io(self, lease_id: int, io_attempt: str) -> bool:
        """Fence one lease against reuse while a physical transfer is active."""

        with self._lock:
            record = self._leases.get(int(lease_id))
            if record is None or record.release_requested:
                return False
            if record.phase is LeasePhase.IO_INFLIGHT:
                return record.io_attempt == str(io_attempt)
            if record.phase not in {
                LeasePhase.RESERVED,
                LeasePhase.COMPUTE_COMPLETE,
            }:
                return False
            record.io_resume_phase = record.phase
            record.phase = LeasePhase.IO_INFLIGHT
            record.io_attempt = str(io_attempt)
            return True

    def complete_io(self, lease_id: int, io_attempt: str, *, success: bool) -> bool:
        """Record a real DMA terminal; stale completions have no effect."""

        with self._lock:
            record = self._leases.get(int(lease_id))
            if (
                record is None
                or record.phase is not LeasePhase.IO_INFLIGHT
                or record.io_attempt != str(io_attempt)
            ):
                return False
            resume = record.io_resume_phase or LeasePhase.RESERVED
            record.io_attempt = None
            record.io_resume_phase = None
            record.phase = LeasePhase.IO_COMPLETE if success else resume
            if record.release_requested:
                record.phase = LeasePhase.RELEASE_PENDING
            return True

    def publish_ready(self, lease_id: int, queue_name: str) -> Optional[MemoryReadyEvent]:
        """Publish a completed local lease without scanning allocator state."""

        with self._ready_condition:
            record = self._leases.get(int(lease_id))
            if record is None or record.release_requested:
                return None
            if record.phase is LeasePhase.READY:
                return record.ready_event
            if record.phase not in {
                LeasePhase.RESERVED,
                LeasePhase.IO_COMPLETE,
            }:
                return None
            event = MemoryReadyEvent(
                queue_name=str(queue_name),
                ready_sequence=next(self._next_ready_sequence),
                lease=record.lease,
            )
            record.phase = LeasePhase.READY
            record.ready_event = event
            self._ready[event.queue_name].append(event)
            self._ready_condition.notify_all()
            return event

    def run_memory_transaction(self, mutate: Callable[[], Any]) -> Any:
        """Run one allocator/Radix metadata transaction under the authority.

        Callers must keep the callback short and must not wait for network or
        DMA completion inside it.  This is the bridge used for parent Radix
        bind and native request cleanup until those operations have dedicated
        typed methods.
        """

        with self.native_guard("controller-memory-transaction"):
            return mutate()

    @contextmanager
    def native_guard(self, owner: str):
        """Serialize one short native allocator/Radix CPU transaction.

        The guard is reentrant because typed authority operations may be
        called by a guarded scheduler mutation.  It must never cover network
        waits, DMA waits, collectives, or GPU Forward.
        """

        owner = str(owner)
        with self._lock:
            depth = int(getattr(self._guard_local, "depth", 0))
            self._guard_local.depth = depth + 1
            self._guard_local.owner = owner
            try:
                yield
            finally:
                self._guard_local.depth = depth
                if depth == 0:
                    self._guard_local.owner = None

    def native_guard_held(self) -> bool:
        return int(getattr(self._guard_local, "depth", 0)) > 0

    def assert_native_guard(self) -> None:
        if not self.native_guard_held():
            raise RuntimeError(
                "native allocator/Radix mutation is outside memory authority"
            )

    def take_ready(
        self,
        queue_name: str,
        *,
        max_items: int = 1,
        timeout: Optional[float] = None,
    ) -> Tuple[MemoryReadyEvent, ...]:
        """Block on an edge-triggered ready queue; no state scanning is used."""

        max_items = int(max_items)
        if max_items <= 0:
            return ()
        with self._ready_condition:
            queue = self._ready[str(queue_name)]
            if not queue and timeout is not None and timeout > 0:
                self._ready_condition.wait_for(lambda: bool(queue), timeout=timeout)
            out = []
            while queue and len(out) < max_items:
                event = queue.popleft()
                record = self._leases.get(event.lease.lease_id)
                if record is not None and record.phase is LeasePhase.READY:
                    out.append(event)
            return tuple(out)

    def take_ready_attempts(
        self,
        queue_name: str,
        attempts: Sequence[RequestGenerationAttempt],
    ) -> Tuple[MemoryReadyEvent, ...]:
        """Atomically take the exact TP0-authorized ready sequence.

        Physical completion can become visible to the rank-local authority at
        slightly different times on different TP ranks.  The scheduler must
        therefore never consume the local FIFO directly for TP>1.  Instead,
        TP0 broadcasts one ordered tuple of attempt identities after the group
        controller has observed every rank's staged completion, and each rank
        resolves that tuple here.

        The operation is all-or-nothing.  A missing attempt is a protocol bug;
        no earlier item is removed, so ranks cannot silently diverge.
        """

        wanted = tuple(attempts)
        if len(set(wanted)) != len(wanted):
            raise ValueError("ready activation contains a duplicate attempt")
        if not wanted:
            return ()
        with self._ready_condition:
            ready_queue = self._ready[str(queue_name)]
            by_key = {
                event.lease.key: event
                for event in ready_queue
                if (
                    (record := self._leases.get(event.lease.lease_id))
                    is not None
                    and record.phase is LeasePhase.READY
                )
            }
            missing = tuple(key for key in wanted if key not in by_key)
            if missing:
                raise RuntimeError(
                    "TP0 activated work that is not staged on this rank: "
                    + ", ".join(map(str, missing))
                )
            selected = tuple(by_key[key] for key in wanted)
            selected_ids = {event.lease.lease_id for event in selected}
            self._ready[str(queue_name)] = deque(
                event
                for event in ready_queue
                if event.lease.lease_id not in selected_ids
            )
            return selected

    def adopt_ready(self, lease_id: int) -> bool:
        """Transfer a ready lease to computation without another allocation."""

        with self._lock:
            record = self._leases.get(int(lease_id))
            if record is None or record.phase is not LeasePhase.READY:
                return False
            record.phase = LeasePhase.COMPUTE
            return True

    def finish_compute(self, lease_id: int) -> bool:
        with self._lock:
            record = self._leases.get(int(lease_id))
            if record is None or record.phase is not LeasePhase.COMPUTE:
                return False
            record.phase = LeasePhase.COMPUTE_COMPLETE
            return True

    def consume_prefill_suffix(
        self,
        lease_id: int,
        extend_tokens: int,
        *,
        final_prompt_chunk: bool,
    ):
        """Give the compute consumer its already-reserved suffix slots."""

        extend_tokens = int(extend_tokens)
        if extend_tokens < 0:
            raise ValueError("extend_tokens must be non-negative")
        with self._lock:
            record = self._leases.get(int(lease_id))
            if (
                record is None
                or record.lease.kind is not LeaseKind.PREFILL_WORKSET
                or record.phase is not LeasePhase.COMPUTE
            ):
                raise RuntimeError("suffix consumption requires a P compute lease")
            logical_suffix = record.lease.prompt_tokens - record.lease.parent_tokens
            physical_suffix = (
                record.lease.prompt_allocated_tokens
                - record.lease.parent_allocated_tokens
            )
            start = record.suffix_cursor
            end = start + extend_tokens
            if end > physical_suffix:
                raise RuntimeError("prefill consumed beyond its reserved suffix")
            if final_prompt_chunk and end < logical_suffix:
                raise RuntimeError("final prefill chunk did not consume the prompt")
            if (
                not final_prompt_chunk
                and end < physical_suffix
                and end % self._page_size
            ):
                raise RuntimeError("non-final prefill chunk must end on a page")
            result = record.lease.suffix_indices[start:end]
            # The final logical token donates ownership of its complete KV
            # page to the native request.  Padding is not another scheduler
            # chunk and must not remain visible as unconsumed work.
            record.suffix_cursor = physical_suffix if final_prompt_chunk else end
            return result

    def remaining_prefill_suffix(self, lease_id: int):
        with self._lock:
            record = self._leases.get(int(lease_id))
            if record is None or record.lease.kind is not LeaseKind.PREFILL_WORKSET:
                raise RuntimeError("unknown P workset lease")
            return record.lease.suffix_indices[record.suffix_cursor :]

    def install_release_handler(
        self,
        lease_id: int,
        handler: Callable[[PhysicalMemoryLease], None],
    ) -> bool:
        """Install the sole cleanup path after pages are bound to a live Req.

        Once Radix/request ownership includes the lease pages, freeing the
        original allocation tensor would double-free shared or deduplicated
        pages.  The handler must release the live request through the native
        cache API and clean any donated recurrent state.  It executes under
        the authority lock and cannot be replaced by a stale attempt.
        """

        if not callable(handler):
            raise TypeError("release handler must be callable")
        with self._lock:
            record = self._leases.get(int(lease_id))
            if record is None:
                return False
            if record.release_handler is not None:
                return record.release_handler is handler
            if record.phase not in {
                # Initial Prefill admission has no physical I/O.  Its
                # all-rank NO_IO_REQUIRED fence binds directly from RESERVED.
                LeasePhase.RESERVED,
                LeasePhase.IO_COMPLETE,
                LeasePhase.READY,
                LeasePhase.COMPUTE,
                LeasePhase.COMPUTE_COMPLETE,
            }:
                return False
            record.release_handler = handler
            return True

    def run_decode_growth(
        self,
        lease_id: int,
        allocate: Callable[[Any], Any],
    ) -> Any:
        """Run the native paged-decode allocation under the same authority.

        ``allocate`` receives the existing SGLang allocator, allowing the
        adapter to reuse ``alloc_decode`` without copying its page-placement
        logic.  Newly materialized pages may consume the allocator-wide growth
        reserve that new ingress is not allowed to use.
        """

        with self._lock:
            record = self._leases.get(int(lease_id))
            if (
                record is None
                or record.lease.kind is not LeaseKind.DECODE_RESERVATION
                or record.phase is not LeasePhase.COMPUTE
            ):
                raise RuntimeError("decode growth requires a compute-owned lease")
            if record.release_handler is None:
                raise RuntimeError(
                    "decode growth requires native request cleanup ownership"
                )
            result = allocate(self._token_allocator)
            return result

    def run_decode_growth_batch(
        self,
        growth_tokens_by_lease: Dict[int, int],
        allocate: Callable[[Any], Any],
    ) -> Any:
        """Materialize one native decode step for several owned requests.

        SGLang allocates decode pages once per batch.  The caller computes the
        exact page-boundary charge for each request before invoking the native
        allocator.  Holding this authority lock across that short operation
        prevents a controller reservation from racing the scheduler's decode
        growth without introducing another free list.
        """

        charges = {
            int(lease_id): int(tokens)
            for lease_id, tokens in growth_tokens_by_lease.items()
        }
        if any(tokens < 0 for tokens in charges.values()):
            raise ValueError("decode growth charge must be non-negative")
        with self._lock:
            records = self._decode_growth_records_locked(charges)
            if not self._ensure_decode_growth_credit_locked(records):
                # The scheduler's normal check_decode_mem path turns this
                # condition into request retraction/retry.  Returning None is
                # also safe for direct controller callers and, critically,
                # does not mutate the native allocator.
                return None

            result = allocate(self._token_allocator)
            if result is None:
                return None
            return result

    def _decode_growth_records_locked(
        self, charges: Dict[int, int]
    ) -> list[tuple[_LeaseRecord, int]]:
        records = []
        for lease_id, tokens in charges.items():
            record = self._leases.get(int(lease_id))
            if (
                record is None
                or record.lease.kind is not LeaseKind.DECODE_RESERVATION
                or record.phase is not LeasePhase.COMPUTE
            ):
                raise RuntimeError(
                    f"decode growth requires a compute lease: {lease_id}"
                )
            if record.release_handler is None:
                raise RuntimeError(
                    "decode growth requires native request cleanup ownership"
                )
            records.append((record, int(tokens)))
        return records

    def _ensure_decode_growth_credit_locked(
        self, records: Sequence[tuple[_LeaseRecord, int]]
    ) -> bool:
        """Check one decode step against real pages in the shared reserve.

        New ingress observes ``available_tokens()`` and therefore leaves the
        reserve untouched.  Decode growth is the sole consumer allowed to use
        that floor, so it checks physical availability rather than subtracting
        the floor a second time.
        """

        required = sum(tokens for _record, tokens in records)
        return required <= int(self._token_allocator.available_size())

    def ensure_decode_growth_credit(
        self, growth_tokens_by_lease: Dict[int, int]
    ) -> bool:
        """Atomically check the next page(s) or report native backpressure.

        A failed check leaves every lease unchanged so SGLang can retract or
        retry the batch through its existing OOM path.
        """

        charges = {
            int(lease_id): int(tokens)
            for lease_id, tokens in growth_tokens_by_lease.items()
        }
        if any(tokens < 0 for tokens in charges.values()):
            raise ValueError("decode growth charge must be non-negative")
        with self._lock:
            records = self._decode_growth_records_locked(charges)
            return self._ensure_decode_growth_credit_locked(records)

    def run_native_transaction(
        self,
        required_tokens: int,
        mutate: Callable[[Any], Any],
    ) -> Any:
        """Temporary bridge for native scheduler allocation through one lock.

        The call refuses to consume capacity promised to decode growth.  It is
        the intended migration point for ``alloc_for_extend/decode``; direct
        mutation of the wrapped allocator can be removed once all callers use
        this bridge or a typed lease API.
        """

        required_tokens = int(required_tokens)
        if required_tokens < 0:
            raise ValueError("required_tokens must be non-negative")
        with self._lock:
            if self.available_tokens() < required_tokens:
                return None
            return mutate(self._token_allocator)

    def request_release(self, lease_id: int) -> bool:
        """Request idempotent release; an in-flight DMA remains fenced."""

        lease_id = int(lease_id)
        with self._lock:
            if lease_id in self._terminal:
                return True
            record = self._leases.get(lease_id)
            if record is None:
                return False
            if record.phase is LeasePhase.COMPUTE:
                return False
            record.release_requested = True
            if record.phase is LeasePhase.IO_INFLIGHT:
                return True
            record.phase = LeasePhase.RELEASE_PENDING
            return True

    def commit_release(self, lease_id: int, *, reason: str = "released") -> bool:
        """Return exact pages after the coordinator has closed all fences."""

        lease_id = int(lease_id)
        capacity_sink = None
        available_tokens = 0
        with self._lock:
            if lease_id in self._terminal:
                return True
            record = self._leases.get(lease_id)
            if record is None:
                return False
            if record.phase is LeasePhase.IO_INFLIGHT:
                return False
            if record.phase is LeasePhase.COMPUTE:
                return False
            if record.release_handler is None:
                self._token_allocator.free(record.lease.device_indices)
                self._free_state(record.lease.state_indices)
            else:
                # Keep the record live if native Radix/request cleanup raises;
                # a retry must retain the exact lease identity and credit.
                record.release_handler(record.lease)
            self._leases.pop(lease_id)
            self._lease_by_key.pop(record.lease.key, None)
            self._remember_terminal(lease_id, reason)
            capacity_sink = self._capacity_available_sink
            available_tokens = max(
                0,
                int(self._token_allocator.available_size())
                - self._decode_growth_reserve,
            )
        if capacity_sink is not None:
            capacity_sink(available_tokens)
        return True

    def phase(self, lease_id: int) -> Optional[LeasePhase]:
        with self._lock:
            record = self._leases.get(int(lease_id))
            return None if record is None else record.phase

    def get(self, lease_id: int) -> Optional[PhysicalMemoryLease]:
        with self._lock:
            record = self._leases.get(int(lease_id))
            return None if record is None else record.lease

    def active_lease_count(self) -> int:
        with self._lock:
            return len(self._leases)
