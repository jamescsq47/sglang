"""Pure CPU, rank-zero authority for complete workset address ownership.

This module does not allocate GPU memory, execute TP commands, observe fences,
or integrate with the runtime allocator. A plan reserves addresses in an
already-created pool; it is NOT proof that GPU descriptors, KV, or Mamba state
are ready. All native/Direct/Slow users must share this authority before a
runtime adapter can safely use it independently of the scheduler.

Lifecycle: grant -> live -> cancel (permanent no-new-use cut) -> all-rank quiet
AND unreferenced reports -> free. ``cancel`` also closes successful worksets
when their final owner retires; it never frees addresses. Quiet must include
outstanding control operations, descriptor initialization, DMA and compute;
unreferenced must include native Request/Radix/shared-prefix references. These
are explicit caller-supplied facts, not inferred from timeout or missing ranks.
Disconnect/shutdown has no implicit reclamation path.

One controller thread writes the ledger. Immutable plans/views may be read by
other threads. Decision sequence numbers cover grant, cancel, subset-close,
subset-return and free, not receipt collection; duplicate commands return the
original decision without advancing the sequence. Run-long retired-attempt
tombstones prevent replay resurrection. There is no restart/recovery support:
an adapter must never recreate this empty ledger over live addresses, and must
use a new incarnation only after the old physical owners are safely gone.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from bisect import bisect_left, bisect_right
from collections import deque
import threading


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _name(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class PageRun:
    """Half-open address interval represented without per-token objects."""

    start: int
    count: int

    def __post_init__(self):
        _integer(self.start, "run start")
        _integer(self.count, "run count", minimum=1)

    @property
    def end(self):
        return self.start + self.count


@dataclass(frozen=True)
class LeaseKey:
    incarnation: str
    snapshot_id: str
    attempt_id: str
    version: int

    def __post_init__(self):
        for name in ("incarnation", "snapshot_id", "attempt_id"):
            _name(getattr(self, name), name)
        _integer(self.version, "lease version", minimum=1)


@dataclass(frozen=True)
class WorksetPlan:
    key: LeaseKey
    sequence: int
    owner: str
    page_size: int
    parent_tokens: int
    prompt_tokens: int
    parent_pages: tuple[PageRun, ...]
    suffix_pages: tuple[PageRun, ...]
    checkpoint_slots: tuple[PageRun, ...]
    runtime_slots: tuple[PageRun, ...]

    @property
    def allocated_tokens(self):
        return self.page_size * sum(r.count for r in self.parent_pages + self.suffix_pages)

    def to_wire(self):
        """Fresh JSON-compatible exact plan; adapters enforce frame limits."""
        return {
            "incarnation": self.key.incarnation,
            "snapshot_id": self.key.snapshot_id,
            "attempt_id": self.key.attempt_id,
            "version": self.key.version,
            "sequence": self.sequence,
            "owner": self.owner,
            "page_size": self.page_size,
            "parent_tokens": self.parent_tokens,
            "prompt_tokens": self.prompt_tokens,
            **{name: [[r.start, r.count] for r in getattr(self, name)]
               for name in ("parent_pages", "suffix_pages", "checkpoint_slots", "runtime_slots")},
        }


@dataclass(frozen=True)
class LedgerDecision:
    sequence: int
    operation: str
    key: LeaseKey
    return_id: str | None = None


@dataclass(frozen=True)
class ReturnPlan:
    """Exact subset cut; donation remains owned until its last reference ends."""

    key: LeaseKey
    return_id: str
    sequence: int
    pages: tuple[PageRun, ...]
    slots: tuple[PageRun, ...]

    def to_wire(self):
        return {"incarnation": self.key.incarnation, "snapshot_id": self.key.snapshot_id,
                "attempt_id": self.key.attempt_id, "version": self.key.version,
                "return_id": self.return_id, "sequence": self.sequence,
                "pages": [[r.start, r.count] for r in self.pages],
                "slots": [[r.start, r.count] for r in self.slots]}


@dataclass(frozen=True)
class LeaseView:
    plan: WorksetPlan
    closing: bool
    quiet_ranks: frozenset[int]
    unreferenced_ranks: frozenset[int]
    remaining_pages: tuple[PageRun, ...]
    remaining_slots: tuple[PageRun, ...]


@dataclass(frozen=True)
class LedgerCounts:
    sequence: int
    free_pages: int
    free_mamba_slots: int
    live_leases: int


@dataclass
class _Lease:
    plan: WorksetPlan
    signature: tuple
    pages: tuple[PageRun, ...]
    slots: tuple[PageRun, ...]
    cancel: LedgerDecision | None = None
    quiet: frozenset[int] = frozenset()
    unreferenced: frozenset[int] = frozenset()
    returns: dict[str, _Return] = field(default_factory=dict)


@dataclass
class _Return:
    plan: ReturnPlan
    quiet: frozenset[int] = frozenset()
    unreferenced: frozenset[int] = frozenset()
    decision: LedgerDecision | None = None


def _take(runs, count):
    """Return immutable (taken, remaining), without mutating a pool."""
    taken, remaining = [], []
    for run in runs:
        amount = min(count, run.count)
        if amount:
            taken.append(PageRun(run.start, amount))
            count -= amount
        if amount < run.count:
            remaining.append(PageRun(run.start + amount, run.count - amount))
    if count:
        return None
    return tuple(taken), tuple(remaining)


def _return(runs, returned):
    merged = []
    for run in sorted(runs + returned, key=lambda r: r.start):
        if merged and run.start < merged[-1].end:
            raise RuntimeError("ownership ledger attempted an overlapping free")
        if merged and run.start == merged[-1].end:
            previous = merged.pop()
            run = PageRun(previous.start, previous.count + run.count)
        merged.append(run)
    return tuple(merged)


def _normalize(runs):
    runs = tuple(runs)
    if any(not isinstance(run, PageRun) for run in runs):
        raise ValueError("explicit PageRun values required")
    try:
        return _return((), runs)
    except RuntimeError as error:
        raise ValueError("overlapping return ranges") from error


def _subtract(owned, subset):
    """Subtract exact owned runs without expanding pages or tokens."""
    remaining = []
    cuts = iter(subset)
    cut = next(cuts, None)
    for run in owned:
        start = run.start
        while cut is not None and cut.start < run.end:
            if cut.start < start or cut.end > run.end:
                raise ValueError("return range is not wholly owned by this lease")
            if cut.start > start:
                remaining.append(PageRun(start, cut.start - start))
            start = cut.end
            cut = next(cuts, None)
        if start < run.end:
            remaining.append(PageRun(start, run.end - start))
    if cut is not None:
        raise ValueError("return range is not wholly owned by this lease")
    return tuple(remaining)


class WorksetLedger:
    """Single-writer CPU address authority; no priority, eviction or timeout.

    Pool counts describe usable addresses (dummy page/slot zero excluded by
    default). Parent and suffix are rounded independently to complete pages;
    neither region can consume the other's reservation. Mamba checkpoint and
    runtime slots come atomically from the same optional slot pool.
    Partial returns preserve lease provenance for donated/shared ranges;
    reference counting and the actual donor/recipient handoff remain external.
    """

    def __init__(self, *, incarnation: str, page_count: int, page_size: int,
                 mamba_slots: int = 0, tp_size: int = 1, rank: int = 0,
                 first_page: int = 1, first_mamba_slot: int = 1,
                 max_native_returns: int = 100000):
        if type(rank) is not int or rank != 0:
            raise ValueError("only rank zero may own a WorksetLedger")
        self.incarnation = _name(incarnation, "incarnation")
        self.page_size = _integer(page_size, "page_size", minimum=1)
        self.tp_size = _integer(tp_size, "tp_size", minimum=1)
        _integer(page_count, "page_count")
        _integer(mamba_slots, "mamba_slots")
        _integer(first_page, "first_page")
        _integer(first_mamba_slot, "first_mamba_slot")
        self._pages = (PageRun(first_page, page_count),) if page_count else ()
        self._slots = (PageRun(first_mamba_slot, mamba_slots),) if mamba_slots else ()
        self._leases: dict[str, _Lease] = {}
        self._retired: dict[tuple[str, str], LedgerDecision] = {}
        self._sequence = 0
        self._version = 0
        self._writer = None
        self._lock = threading.RLock()
        self._max_native_returns = _integer(max_native_returns, "max_native_returns", minimum=1)
        self._native_returns = {}
        self._native_pending: dict[str, set[ReturnPlan]] = {}
        self._native_by_plan: dict[ReturnPlan, set[str]] = {}
        self._native_completed = deque()
        # NativeLastRefFreeAdapter issues increasing tickets per rank. Once
        # every plan of a ticket is committed, its full receipt can be
        # compacted into an interval that rejects late replay without keeping
        # every immutable page tuple for the lifetime of the server.
        self._retired_native_tickets: dict[str, list[tuple[int, int]]] = {}

    @staticmethod
    def _native_ticket_parts(ticket: str):
        source, marker, number = ticket.rpartition(":native-free")
        if not source or not marker or not number.isdecimal():
            return None
        ordinal = int(number)
        return (source, ordinal) if ordinal > 0 and str(ordinal) == number else None

    def _native_ticket_retired(self, ticket: str) -> bool:
        parts = self._native_ticket_parts(ticket)
        if parts is None:
            return False
        source, ordinal = parts
        intervals = self._retired_native_tickets.get(source, ())
        index = bisect_right(intervals, (ordinal, float("inf"))) - 1
        return index >= 0 and intervals[index][0] <= ordinal <= intervals[index][1]

    def _retire_native_ticket(self, ticket: str) -> None:
        source, ordinal = self._native_ticket_parts(ticket)
        intervals = self._retired_native_tickets.setdefault(source, [])
        index = bisect_left(intervals, (ordinal, ordinal))
        start = end = ordinal
        if index and intervals[index - 1][1] + 1 >= ordinal:
            index -= 1
            start, end = intervals.pop(index)
            end = max(end, ordinal)
        while index < len(intervals) and intervals[index][0] <= end + 1:
            next_start, next_end = intervals.pop(index)
            start, end = min(start, next_start), max(end, next_end)
        intervals.insert(index, (start, end))

    def _compact_native_receipts(self) -> None:
        while len(self._native_returns) >= self._max_native_returns and self._native_completed:
            ticket = self._native_completed.popleft()
            if ticket not in self._native_returns:
                continue
            self._retire_native_ticket(ticket)
            del self._native_returns[ticket]
            self._native_pending.pop(ticket, None)

    def _native_plan_committed(self, plan: ReturnPlan) -> None:
        for ticket in self._native_by_plan.pop(plan, ()):
            pending = self._native_pending[ticket]
            pending.discard(plan)
            if not pending and self._native_ticket_parts(ticket) is not None:
                self._native_completed.append(ticket)

    @contextmanager
    def _write(self):
        with self._lock:
            writer = threading.current_thread()
            if self._writer is None:
                self._writer = writer
            elif self._writer is not writer:
                raise RuntimeError("ownership ledger has a different controller writer")
            yield

    def _current(self, key):
        if not isinstance(key, LeaseKey) or key.incarnation != self.incarnation:
            return None
        lease = self._leases.get(key.snapshot_id)
        return lease if lease is not None and lease.plan.key == key else None

    @property
    def counts(self):
        with self._lock:
            return LedgerCounts(self._sequence, sum(r.count for r in self._pages),
                                sum(r.count for r in self._slots), len(self._leases))

    def view(self, key):
        with self._lock:
            lease = self._current(key)
            return self._view(lease)

    def current_view(self, snapshot_id: str):
        """Immutable current owner, for cancel-before-grant arbitration only.

        Mutations still require the full exact LeaseKey. A snapshot name must
        never authorize cancellation or address reuse on its own.
        """
        _name(snapshot_id, "snapshot_id")
        with self._lock:
            return self._view(self._leases.get(snapshot_id))

    @staticmethod
    def _view(lease):
        return None if lease is None else LeaseView(
            lease.plan, lease.cancel is not None, lease.quiet, lease.unreferenced,
            lease.pages, lease.slots)

    def may_start(self, key):
        """Logical start gate only; physical descriptor readiness is external."""
        with self._lock:
            lease = self._current(key)
            return lease is not None and lease.cancel is None and not lease.returns

    def may_use(self, key, *, pages=(), slots=()):
        """Gate new users of a subset; not an existing user's physical fence."""
        pages, slots = _normalize(pages), _normalize(slots)
        with self._lock:
            lease = self._current(key)
            if lease is None or lease.cancel is not None or not (pages or slots):
                return False
            available_pages, available_slots = lease.pages, lease.slots
            for returning in lease.returns.values():
                if returning.decision is None:
                    available_pages = _subtract(available_pages, returning.plan.pages)
                    available_slots = _subtract(available_slots, returning.plan.slots)
            try:
                _subtract(available_pages, pages)
                _subtract(available_slots, slots)
            except ValueError:
                return False
            return True

    def grant(self, snapshot_id: str, attempt_id: str, *, owner: str,
              parent_tokens: int, prompt_tokens: int,
              checkpoint_slots: int = 0, runtime_slots: int = 0):
        """Reserve the entire workset, or return None without any mutation.

        An identical live retry returns its immutable original plan. A changed
        retry, competing live owner, or retired attempt is an error, not a
        fresh grant. Capacity failure does not burn an attempt/version/sequence.
        """
        for value, name in ((snapshot_id, "snapshot_id"), (attempt_id, "attempt_id"), (owner, "owner")):
            _name(value, name)
        for value, name in ((parent_tokens, "parent_tokens"), (prompt_tokens, "prompt_tokens"),
                            (checkpoint_slots, "checkpoint_slots"), (runtime_slots, "runtime_slots")):
            _integer(value, name)
        if prompt_tokens < parent_tokens or prompt_tokens == 0:
            raise ValueError("a nonempty complete prompt must contain its parent")
        signature = (owner, parent_tokens, prompt_tokens, checkpoint_slots, runtime_slots)
        with self._write():
            old = self._leases.get(snapshot_id)
            if old is not None:
                if (old.plan.key.attempt_id == attempt_id and old.signature == signature
                        and old.cancel is None and not old.returns):
                    return old.plan
                raise ValueError("snapshot already owns a different or closing workset")
            if (snapshot_id, attempt_id) in self._retired:
                raise ValueError("retired attempt cannot acquire another workset")
            parent_count = (parent_tokens + self.page_size - 1) // self.page_size
            suffix_count = (prompt_tokens - parent_tokens + self.page_size - 1) // self.page_size
            pages = _take(self._pages, parent_count + suffix_count)
            slots = _take(self._slots, checkpoint_slots + runtime_slots)
            if pages is None or slots is None:
                return None
            parent, suffix = _take(pages[0], parent_count)
            checkpoint, runtime = _take(slots[0], checkpoint_slots)
            key = LeaseKey(self.incarnation, snapshot_id, attempt_id, self._version + 1)
            plan = WorksetPlan(key, self._sequence + 1, owner, self.page_size,
                               parent_tokens, prompt_tokens, parent, suffix, checkpoint, runtime)
            self._leases[snapshot_id] = _Lease(plan, signature, _normalize(parent + suffix),
                                               _normalize(checkpoint + runtime))
            self._pages, self._slots = pages[1], slots[1]
            self._version, self._sequence = key.version, plan.sequence
            return plan

    def cancel(self, key):
        """Close new starts; no pages/slots or KV ownership are relinquished."""
        with self._write():
            lease = self._current(key)
            if lease is None:
                return None
            if lease.cancel is None:
                self._sequence += 1
                lease.cancel = LedgerDecision(self._sequence, "cancel", key)
            return lease.cancel

    def report_fence(self, key, rank: int, *, quiet: bool, unreferenced: bool):
        """Accept post-close facts only; unknown/False is never completion.

        The owner adapter must first stop admitting new uses, then observe its
        actual physical and reference fences. Reports are monotonic because a
        closed attempt can never start again. This collects ACKs; it does not
        make an allocation decision or advance the decision sequence.
        """
        _integer(rank, "rank")
        if rank >= self.tp_size or type(quiet) is not bool or type(unreferenced) is not bool:
            raise ValueError("invalid rank or fence facts")
        with self._write():
            lease = self._current(key)
            if lease is None or lease.cancel is None:
                return False
            if quiet:
                lease.quiet |= {rank}
            if unreferenced:
                lease.unreferenced |= {rank}
            return len(lease.quiet) == len(lease.unreferenced) == self.tp_size

    def begin_return(self, key, return_id: str, *, pages=(), slots=()):
        """Close an exact subset to new use, retaining all addresses until ACK.

        Req/Radix donation is not a return: the caller keeps the original key
        and ranges as provenance and starts this operation only when retiring
        those references. Other disjoint subsets may remain in use.
        """
        _name(return_id, "return_id")
        pages, slots = _normalize(pages), _normalize(slots)
        if not (pages or slots):
            raise ValueError("a partial return must name some owned addresses")
        with self._write():
            lease = self._current(key)
            if lease is None:
                return None
            previous = lease.returns.get(return_id)
            if previous is not None:
                if previous.plan.pages != pages or previous.plan.slots != slots:
                    raise ValueError("return identity reused with a different subset")
                return previous.plan
            available_pages, available_slots = lease.pages, lease.slots
            for returning in lease.returns.values():
                if returning.decision is None:
                    available_pages = _subtract(available_pages, returning.plan.pages)
                    available_slots = _subtract(available_slots, returning.plan.slots)
            _subtract(available_pages, pages)
            _subtract(available_slots, slots)
            plan = ReturnPlan(key, return_id, self._sequence + 1, pages, slots)
            lease.returns[return_id] = _Return(plan)
            self._sequence = plan.sequence
            return plan

    def native_return(self, return_id: str, resource: str, indices: tuple[int, ...],
                      allocation_sequence: int):
        """Translate a valid native last-reference free into exact subset cuts.

        This bridge is ONLY for the native single-writer last-ref contract.
        ``indices`` were cloned at that boundary; the captured allocation epoch
        rejects reuse after enqueue. It cannot identify an already-invalid old
        raw tensor first submitted AFTER address reuse. Transport/cancellation
        callbacks must use LeaseKey provenance instead of this bridge.

        Validate every address before closing any subset. This creates no rank
        ACK and frees nothing: each rank must independently prove its matching
        native references and physical fences for the resulting ReturnPlans.
        Run-retained bounded receipts make a duplicate ticket safe after reuse.
        """
        _name(return_id, "native return_id")
        _integer(allocation_sequence, "allocation_sequence")
        if resource not in {"attention", "mamba"} or type(indices) is not tuple or not indices:
            raise ValueError("native return requires a resource and immutable nonempty indices")
        divisor = self.page_size if resource == "attention" else 1
        addresses = sorted({_integer(index, "native index") // divisor for index in indices})
        runs = []
        for address in addresses:
            if runs and runs[-1].end == address:
                last = runs.pop()
                runs.append(PageRun(last.start, last.count + 1))
            else:
                runs.append(PageRun(address, 1))
        runs = tuple(runs)
        signature = resource, runs, allocation_sequence
        with self._write():
            previous = self._native_returns.get(return_id)
            if previous is not None:
                if previous[0] != signature:
                    raise ValueError("native return ticket changed its addresses/epoch")
                return previous[1]
            if self._native_ticket_retired(return_id):
                # Its physical return was already committed. Never reinterpret
                # a compacted ticket against addresses that may be reused.
                raise ValueError("retired native return ticket cannot be replayed")
            self._compact_native_receipts()
            if len(self._native_returns) >= self._max_native_returns:
                unresolved = sum(bool(pending) for pending in self._native_pending.values())
                raise RuntimeError(
                    "native return receipt capacity exhausted; retaining ownership "
                    f"pending={unresolved} total={len(self._native_returns)}"
                )
            unresolved, groups = runs, []
            for lease in self._leases.values():
                owned = lease.pages if resource == "attention" else lease.slots
                overlap = []
                for requested in unresolved:
                    for current in owned:
                        start, end = max(requested.start, current.start), min(requested.end, current.end)
                        if start < end:
                            overlap.append(PageRun(start, end - start))
                overlap = _normalize(overlap)
                if not overlap:
                    continue
                if lease.plan.sequence > allocation_sequence:
                    raise ValueError("native return points at an owner newer than its capture epoch")
                available = owned
                for pending in lease.returns.values():
                    if pending.decision is None:
                        available = _subtract(available, pending.plan.pages if resource == "attention" else pending.plan.slots)
                _subtract(available, overlap)  # Reject already-pending returns.
                groups.append((lease.plan.key, overlap))
                unresolved = _subtract(unresolved, overlap)
            if unresolved:
                raise ValueError("native return contains unknown or already-returned addresses")
            plans = tuple(self.begin_return(
                key, "native:" + return_id,
                pages=group if resource == "attention" else (),
                slots=group if resource == "mamba" else (),
            ) for key, group in groups)
            self._native_returns[return_id] = (signature, plans)
            self._native_pending[return_id] = set(plans)
            for plan in plans:
                self._native_by_plan.setdefault(plan, set()).add(return_id)
            return plans

    def _returning(self, plan):
        if not isinstance(plan, ReturnPlan):
            return None, None
        lease = self._current(plan.key)
        returning = None if lease is None else lease.returns.get(plan.return_id)
        if returning is None or returning.plan != plan:
            return None, None
        return lease, returning

    def report_return_fence(self, plan: ReturnPlan, rank: int, *, quiet: bool, unreferenced: bool):
        """A rank ACK names the complete immutable subset, not just its id."""
        _integer(rank, "rank")
        if rank >= self.tp_size or type(quiet) is not bool or type(unreferenced) is not bool:
            raise ValueError("invalid rank or fence facts")
        with self._write():
            _, returning = self._returning(plan)
            if returning is None:
                return False
            if quiet:
                returning.quiet |= {rank}
            if unreferenced:
                returning.unreferenced |= {rank}
            return len(returning.quiet) == len(returning.unreferenced) == self.tp_size

    def commit_return(self, plan: ReturnPlan):
        """Return only this all-rank-fenced subset; the lease keeps provenance."""
        with self._write():
            lease, returning = self._returning(plan)
            if returning is None:
                return None
            if returning.decision is not None:
                return returning.decision
            if len(returning.quiet) != self.tp_size or len(returning.unreferenced) != self.tp_size:
                return None
            remaining_pages = _subtract(lease.pages, plan.pages)
            remaining_slots = _subtract(lease.slots, plan.slots)
            pages = _return(self._pages, plan.pages)
            slots = _return(self._slots, plan.slots)
            decision = LedgerDecision(self._sequence + 1, "return", plan.key, plan.return_id)
            lease.pages, lease.slots = remaining_pages, remaining_slots
            self._pages, self._slots, self._sequence = pages, slots, decision.sequence
            returning.decision = decision
            self._native_plan_committed(plan)
            return decision

    def free(self, key):
        """Return addresses only after every rank supplied both exact fences.

        Retried frees return the old decision without touching a successor.
        Unknown/stale keys or incomplete fences return None and retain memory.
        """
        with self._write():
            lease = self._current(key)
            if lease is None:
                if isinstance(key, LeaseKey):
                    old = self._retired.get((key.snapshot_id, key.attempt_id))
                    if old is not None and old.key == key:
                        return old
                return None
            if (lease.cancel is None or len(lease.quiet) != self.tp_size
                    or len(lease.unreferenced) != self.tp_size):
                return None
            pages = _return(self._pages, lease.pages)
            slots = _return(self._slots, lease.slots)
            decision = LedgerDecision(self._sequence + 1, "free", key)
            self._retired[(key.snapshot_id, key.attempt_id)] = decision
            del self._leases[key.snapshot_id]
            self._pages, self._slots, self._sequence = pages, slots, decision.sequence
            return decision
