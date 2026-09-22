"""Ordered TCP workset ownership decisions, independent of model collectives.

GRANT, CANCEL, subset-close, RETURN and FREE share the ledger's decision log.
This component executes exact addresses; it never selects them, binds Radix,
launches payload DMA, or independently admits a model batch.

Each rank installs the exact rank-zero CPU address plan in sequence. Installation
ACK means CPU ownership only. Adapter.prepare returns a Future carrying a
descriptor with is_ready(); both the Future and that physical initialization
event must be ready, NOT merely the Python task that submitted CUDA work.
Neither proves payload DMA or compute completion. Rank zero
can capture the existing all-rank facts as one immutable ReadyCut for a future
compute-admission adapter; a local Future alone is never a group-ready decision.

The injected adapter provides bounded, idempotent install(plan), cancel(plan),
release(plan, pages, slots, *, decision), plus prepare(plan)->Future and close(plan, scope)
->Future[FenceProof]. close synchronously closes the exact scope to new uses;
its Future observes (never creates) real control/CUDA/IO/reference fences.
Whole-close proof covers all remaining ranges, including pending subset closes
and partial preparation failures. release only applies the committed CPU return
and must be idempotent even after an exception. install reserves the supplied
addresses, not choose different ones. prepare is called at most once: an
exception may follow partial CUDA submission and is retained, never retried or
treated as proof of no I/O. The future stays attached after cancellation.
Only the owning progress/controller thread calls this component's methods.
"""

from __future__ import annotations

from concurrent.futures import Future
from collections import deque
from dataclasses import dataclass, field
import hashlib
import json

from sglang.srt.disaggregation.agentic_tp_events import EventKey
from sglang.srt.disaggregation.agentic_workset_ledger import (
    LeaseKey, PageRun, WorksetPlan, LedgerDecision, ReturnPlan, _normalize, _subtract,
)


class WorksetProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReadyCut:
    """A ready set at an installed log cut, not an executable batch grant.

    Earlier grants may still be preparing: readiness is not a FIFO barrier.
    A runtime adapter must distribute this identical cut at its native compute
    boundary and arbitrate cancellation/references before admitting any batch.
    """

    incarnation: str
    through_sequence: int
    ready: tuple[LeaseKey, ...]


@dataclass(frozen=True)
class FenceProof:
    """Adapter-observed facts for one immutable close, never inferred by time.

    quiet includes every queued/unknown control operation and real descriptor,
    DMA and compute completion. unreferenced includes Req and shared Radix.
    A whole-close proof also covers outstanding subset-close observers.
    """

    key: LeaseKey
    close_sequence: int
    quiet: bool
    unreferenced: bool


@dataclass(frozen=True)
class LocalWorksetView:
    """Immutable CPU snapshot; descriptor readiness is not group admission."""

    plan: WorksetPlan
    installed: bool
    descriptor: object
    descriptor_ready: bool
    closing: bool
    remaining_pages: tuple[PageRun, ...]
    remaining_slots: tuple[PageRun, ...]


@dataclass
class _Pending:
    plan: WorksetPlan
    command_id: int
    installed: bool = False
    acked: bool = False
    prepare_started: bool = False
    future: Future | None = None
    result_received: bool = False
    descriptor_ready: bool = False
    descriptor: object = None
    cancelled: bool = False
    cancel_applied: bool = False
    install_error: Exception | None = None
    prepare_error: Exception | None = None
    cancel_error: Exception | None = None
    pages: tuple[PageRun, ...] = ()
    slots: tuple[PageRun, ...] = ()
    closes: set[int] = field(default_factory=set)


@dataclass
class _DecisionPending:
    scope: LedgerDecision | ReturnPlan
    command_id: int = 1
    install_error: Exception | None = None
    applied: bool = False


@dataclass
class _Closing:
    scope: LedgerDecision | ReturnPlan
    future: Future | None = None
    proof: FenceProof | None = None
    error: Exception | None = None
    reported: bool = False


def event_key(key: LeaseKey):
    return EventKey(key.snapshot_id, json.dumps(
        [key.incarnation, key.attempt_id, key.version], separators=(",", ":")))


def decision_key(scope):
    key = scope.key
    return EventKey(key.snapshot_id, json.dumps(
        [key.incarnation, key.attempt_id, key.version, "decision", scope.sequence],
        separators=(",", ":")))


def _fingerprint(wire):
    return hashlib.sha256(json.dumps(wire, sort_keys=True, separators=(",", ":")).encode()).digest()


def decision_to_wire(scope):
    if isinstance(scope, ReturnPlan):
        return dict(scope.to_wire(), operation="close_return")
    if not isinstance(scope, LedgerDecision):
        raise WorksetProtocolError("exact ledger decision required")
    key = scope.key
    return dict(incarnation=key.incarnation, snapshot_id=key.snapshot_id,
                attempt_id=key.attempt_id, version=key.version,
                sequence=scope.sequence, operation=scope.operation, return_id=scope.return_id)


def decision_from_wire(wire):
    common = {"incarnation", "snapshot_id", "attempt_id", "version", "sequence", "operation", "return_id"}
    if not isinstance(wire, dict):
        raise WorksetProtocolError("decision object required")
    operation = wire.get("operation")
    expected = common | {"pages", "slots"} if operation == "close_return" else common
    if set(wire) != expected or operation not in {"cancel", "close_return", "return", "free"}:
        raise WorksetProtocolError("invalid decision shape/operation")
    try:
        key = LeaseKey(wire["incarnation"], wire["snapshot_id"], wire["attempt_id"], wire["version"])
    except ValueError as error:
        raise WorksetProtocolError(str(error)) from error
    sequence = _positive(wire["sequence"], "decision sequence")
    return_id = wire["return_id"]
    if operation in {"close_return", "return"}:
        if not isinstance(return_id, str) or not return_id:
            raise WorksetProtocolError("exact subset identity required")
    elif return_id is not None:
        raise WorksetProtocolError("whole-workset decision cannot name a subset")
    if operation == "close_return":
        pages, slots = _normalize(_runs(wire["pages"])), _normalize(_runs(wire["slots"]))
        if not (pages or slots):
            raise WorksetProtocolError("empty subset close")
        return ReturnPlan(key, return_id, sequence, pages, slots)
    return LedgerDecision(sequence, operation, key, return_id)


def _positive(value, name, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise WorksetProtocolError(f"invalid {name}")
    return value


def _runs(values):
    if not isinstance(values, (list, tuple)):
        raise WorksetProtocolError("run list required")
    result = []
    for value in values:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise WorksetProtocolError("run must be [start, count]")
        run = PageRun(_positive(value[0], "run start", zero=True),
                      _positive(value[1], "run count"))
        if result and run.start < result[-1].end:
            raise WorksetProtocolError("unordered or overlapping plan runs")
        result.append(run)
    return tuple(result)


def _nonoverlapping(runs):
    ordered = tuple(sorted(runs, key=lambda run: run.start))
    if any(right.start < left.end for left, right in zip(ordered, ordered[1:])):
        raise WorksetProtocolError("address already owned by another grant")
    return ordered


def plan_from_wire(wire):
    """Validate the immutable ledger wire shape, without torch or per-token data."""
    required = {"incarnation", "snapshot_id", "attempt_id", "version", "sequence", "owner",
                "page_size", "parent_tokens", "prompt_tokens", "parent_pages", "suffix_pages",
                "checkpoint_slots", "runtime_slots"}
    if not isinstance(wire, dict) or set(wire) != required:
        raise WorksetProtocolError("unexpected workset plan fields")
    try:
        key = LeaseKey(wire["incarnation"], wire["snapshot_id"], wire["attempt_id"], wire["version"])
    except ValueError as error:
        raise WorksetProtocolError(str(error)) from error
    if not isinstance(wire["owner"], str) or not wire["owner"]:
        raise WorksetProtocolError("nonempty owner required")
    sequence = _positive(wire["sequence"], "sequence")
    page_size = _positive(wire["page_size"], "page size")
    parent = _positive(wire["parent_tokens"], "parent tokens", zero=True)
    prompt = _positive(wire["prompt_tokens"], "prompt tokens")
    if parent > prompt:
        raise WorksetProtocolError("prompt does not contain parent")
    ranges = {name: _runs(wire[name]) for name in (
        "parent_pages", "suffix_pages", "checkpoint_slots", "runtime_slots")}
    if (sum(r.count for r in ranges["parent_pages"]) != (parent + page_size - 1) // page_size
            or sum(r.count for r in ranges["suffix_pages"]) != (prompt - parent + page_size - 1) // page_size):
        raise WorksetProtocolError("plan does not reserve the complete parent and suffix")
    _nonoverlapping(ranges["parent_pages"] + ranges["suffix_pages"])
    _nonoverlapping(ranges["checkpoint_slots"] + ranges["runtime_slots"])
    return WorksetPlan(key, sequence, wire["owner"], page_size, parent, prompt, **ranges)


class TPWorksetExecutor:
    """Persistent exact CPU grant installation plus independent preparation.

    The controller is the ledger writer: drain_fenced returns all-rank facts,
    then it commits the ledger return/free and publishes that exact decision.
    Followers install decisions in sequence, so return precedes address reuse.
    Exceptions/disconnection retain ownership, never synthesize a free.
    """

    def __init__(self, client, adapter, *, incarnation: str, page_count: int,
                 page_size: int, mamba_slots: int = 0, first_page: int = 1,
                 first_mamba_slot: int = 1, namespace="workset-grant", progress_budget=32):
        if not isinstance(incarnation, str) or not incarnation or not namespace:
            raise ValueError("incarnation and namespace are required")
        self.client, self.adapter = client, adapter
        self.incarnation, self.namespace = incarnation, namespace
        self.page_size = _positive(page_size, "page size")
        self.page_count = _positive(page_count, "page count", zero=True)
        self.mamba_slots = _positive(mamba_slots, "Mamba slots", zero=True)
        self.first_page = _positive(first_page, "first page", zero=True)
        self.first_mamba_slot = _positive(first_mamba_slot, "first Mamba slot", zero=True)
        self.installed_sequence = 0
        self._published: dict[int, WorksetPlan] = {}
        self._publication_log: dict[int, bytes] = {}
        self._seen: dict[int, bytes] = {}
        self._seen_keys: set[LeaseKey] = set()
        self._pending: dict[int, _Pending | _DecisionPending] = {}
        self._keys: dict[LeaseKey, _Pending] = {}
        self._pages: tuple[PageRun, ...] = ()
        self._slots: tuple[PageRun, ...] = ()
        self._cancelled: set[LeaseKey] = set()
        self._withdrawn: set[LeaseKey] = set()
        self._published_keys: set[LeaseKey] = set()
        self._wire_keys: dict[EventKey, LeaseKey] = {}
        self._active: dict[LeaseKey, _Pending] = {}
        self._active_queue = deque()
        self._ready: set[LeaseKey] = set()
        self._dirty: set[LeaseKey] = set()
        self._closes: dict[int, _Closing] = {}
        self._close_wire: dict[EventKey, int] = {}
        self._whole_close: dict[LeaseKey, int] = {}
        self._subset_close: dict[tuple[LeaseKey, str], int] = {}
        self._closing_active: dict[int, _Closing] = {}
        self._closing_queue = deque()
        self._fenced = deque()
        self._group_fenced: set[int] = set()
        self._fence_dirty: set[int] = set()
        self._retired: set[LeaseKey] = set()
        self._free_published: set[LeaseKey] = set()
        # Terminal FREE is itself a command: local release is not proof that
        # followers have installed it. Retire its event family only on the
        # all-rank execution ACK, driven by the existing update notifications.
        self._retire_pending: dict[EventKey, LedgerDecision] = {}
        self._retire_dirty: set[EventKey] = set()
        self.progress_budget = _positive(progress_budget, "progress budget")
        self.client.subscribe_updates(self.namespace)
        self.client.subscribe_updates(self.namespace + ":prepared")
        self.client.subscribe_updates(self.namespace + ":decisions")
        self.client.subscribe_updates(self.namespace + ":fenced")
        self._error = None

    def _check(self):
        if self._error is not None:
            raise WorksetProtocolError(self._error)

    def _validate(self, plan):
        if plan.key.incarnation != self.incarnation or plan.page_size != self.page_size:
            raise WorksetProtocolError("workset pool incarnation/shape changed")
        for runs, first, count in ((plan.parent_pages + plan.suffix_pages, self.first_page, self.page_count),
                                   (plan.checkpoint_slots + plan.runtime_slots, self.first_mamba_slot, self.mamba_slots)):
            if any(r.start < first or r.end > first + count for r in runs):
                raise WorksetProtocolError("workset address outside the configured physical pool")

    def publish_grant(self, plan: WorksetPlan):
        self._check()
        if self.client.rank != 0:
            raise ValueError("only rank zero publishes worksets")
        plan = plan_from_wire(plan.to_wire())
        self._validate(plan)
        self._publish_order(plan.sequence, plan.to_wire())
        if plan.key in self._retired:
            return  # Exact replay cannot recreate live metadata for an old owner.
        # Preserve an unknown publication before submitting; never manufacture
        # a different plan on retry after a queue/socket error.
        self._published[plan.sequence] = plan
        self._published_keys.add(plan.key)
        self.client.publish_command(self.namespace, event_key(plan.key),
                                    plan.to_wire(), command_id=1)

    def _publish_order(self, sequence, wire):
        fingerprint = _fingerprint(wire)
        previous = self._publication_log.get(sequence)
        if previous is not None and previous != fingerprint:
            raise WorksetProtocolError("decision sequence reused with different plan")
        if previous is None and sequence != len(self._publication_log) + 1:
            raise WorksetProtocolError("cannot skip ledger decisions")
        self._publication_log[sequence] = fingerprint

    def publish_decision(self, scope):
        """Publish the exact committed ledger decision, never select addresses.

        RETURN/FREE require the preceding all-rank close proof. The controller
        must feed those same facts into the ledger before committing/reusing.
        Retried old decisions remain idempotent even after a successor exists.
        """
        self._check()
        if self.client.rank != 0:
            raise ValueError("only rank zero publishes workset decisions")
        scope = decision_from_wire(decision_to_wire(scope))
        if scope.key.incarnation != self.incarnation:
            raise WorksetProtocolError("decision incarnation changed")
        wire = decision_to_wire(scope)
        previous = self._publication_log.get(scope.sequence)
        if previous is None:
            if scope.key not in self._published_keys or scope.key in self._retired:
                raise WorksetProtocolError("decision has no live exact grant")
            if isinstance(scope, LedgerDecision) and scope.operation in {"free", "return"}:
                close = self._close_for_commit(scope)
                if (not self.client.command_complete(self.namespace + ":decisions", decision_key(close.scope))
                        or self.client.group_status(self.namespace + ":fenced", decision_key(close.scope)) != 1):
                    raise WorksetProtocolError("return/free requires all-rank physical and reference fences")
        self._publish_order(scope.sequence, wire)
        if scope.key in self._retired:
            return  # Exact old replay cannot recreate a retired event family.
        if isinstance(scope, LedgerDecision) and scope.operation == "free":
            self._free_published.add(scope.key)
        # A published close immediately withdraws ready authority on the leader.
        # Followers close only when its ordered command is installed.
        if isinstance(scope, ReturnPlan) or scope.operation == "cancel":
            self._ready.discard(scope.key)
            self._withdrawn.add(scope.key)
        self.client.publish_command(self.namespace + ":decisions", decision_key(scope), wire, command_id=1)

    def _close_for_commit(self, decision):
        sequence = (self._whole_close.get(decision.key) if decision.operation == "free"
                    else self._subset_close.get((decision.key, decision.return_id)))
        close = self._closes.get(sequence)
        if close is None:
            raise WorksetProtocolError("commit has no matching exact close")
        return close

    def cancel(self, key: LeaseKey):
        """Leader closes future use; this reports no quiet/reference/free fact."""
        self._check()
        if self.client.rank != 0:
            raise ValueError("only rank zero cancels a group workset")
        if key not in self._published_keys:
            raise WorksetProtocolError("cannot cancel an unpublished exact workset")
        self._cancelled.add(key)
        self._ready.discard(key)
        self.client.publish_receipt(self.namespace, event_key(key), -1)

    def _receive(self):
        for key, command_id, wire in self.client.drain_commands(self.namespace):
            plan = plan_from_wire(wire)
            self._validate(plan)
            if key != event_key(plan.key) or command_id != 1:
                raise WorksetProtocolError("command does not match its immutable grant identity")
            if not self._receive_sequence(plan.sequence, wire):
                continue
            if plan.key in self._seen_keys:
                raise WorksetProtocolError("lease identity reused under another sequence")
            pending = _Pending(plan, command_id)
            self._pending[plan.sequence] = pending
            self._keys[plan.key] = pending
            self._seen_keys.add(plan.key)
            self._wire_keys[key] = plan.key
        for key, command_id, wire in self.client.drain_commands(self.namespace + ":decisions"):
            scope = decision_from_wire(wire)
            if (scope.key.incarnation != self.incarnation or key != decision_key(scope)
                    or command_id != 1):
                raise WorksetProtocolError("decision does not match exact pool/command identity")
            if self._receive_sequence(scope.sequence, wire):
                self._pending[scope.sequence] = _DecisionPending(scope, command_id)

    def _receive_sequence(self, sequence, wire):
        fingerprint = _fingerprint(wire)
        previous = self._seen.get(sequence)
        if previous is not None:
            if previous != fingerprint:
                raise WorksetProtocolError("received conflicting decision sequence")
            return False
        self._seen[sequence] = fingerprint
        return True

    def _activate(self, pending):
        key = pending.plan.key
        if key not in self._active:
            self._active[key] = pending
            self._active_queue.append(key)

    def _install_decision(self, command):
        scope = command.scope
        pending = self._keys.get(scope.key)
        if pending is None or not pending.installed:
            raise WorksetProtocolError("decision targets no currently installed exact lease")
        closing = isinstance(scope, ReturnPlan) or scope.operation == "cancel"
        if closing:
            if scope.sequence not in self._closes:
                if isinstance(scope, ReturnPlan):
                    if (scope.key, scope.return_id) in self._subset_close:
                        raise WorksetProtocolError("subset close identity reused")
                    available_pages, available_slots = pending.pages, pending.slots
                    for sequence in pending.closes:
                        other = self._closes[sequence].scope
                        if isinstance(other, ReturnPlan):
                            available_pages = _subtract(available_pages, other.pages)
                            available_slots = _subtract(available_slots, other.slots)
                    try:
                        _subtract(available_pages, scope.pages)
                        _subtract(available_slots, scope.slots)
                    except ValueError as error:
                        raise WorksetProtocolError("subset is not exactly owned and open") from error
                    self._subset_close[(scope.key, scope.return_id)] = scope.sequence
                else:
                    if scope.key in self._whole_close:
                        raise WorksetProtocolError("whole close identity reused")
                    self._whole_close[scope.key] = scope.sequence
                    pending.cancelled = True
                    self._cancelled.add(scope.key)
                self._withdrawn.add(scope.key)
                self._ready.discard(scope.key)
                self._dirty.add(scope.key)
                close = _Closing(scope)
                self._closes[scope.sequence] = close
                self._close_wire[decision_key(scope)] = scope.sequence
                pending.closes.add(scope.sequence)
                # At-most-once invocation: errors may follow partial submission.
                try:
                    future = self.adapter.close(pending.plan, scope)
                    if not isinstance(future, Future):
                        raise TypeError("close must return a retained physical/reference fence Future")
                    close.future = future
                    if not isinstance(scope, ReturnPlan):
                        pending.cancel_applied = True
                except Exception as error:
                    close.error = error
                if close.error is None:
                    self._closing_active[scope.sequence] = close
                    self._closing_queue.append(scope.sequence)
                self._fence_dirty.add(scope.sequence)
        else:
            close = self._close_for_commit(scope)
            if not close.reported:
                raise WorksetProtocolError("commit received without this rank's real fence")
            if scope.operation == "return":
                pages, slots = close.scope.pages, close.scope.slots
            else:
                pages, slots = pending.pages, pending.slots
            try:
                remaining_pages = _subtract(pending.pages, pages)
                remaining_slots = _subtract(pending.slots, slots)
                all_pages = _subtract(self._pages, pages)
                all_slots = _subtract(self._slots, slots)
            except ValueError as error:
                raise WorksetProtocolError("committed return crosses exact current ownership") from error
            # A failed/unknown installation blocks later address reuse. Adapter
            # retries are idempotent for this exact lease and subset.
            self.adapter.release(pending.plan, pages, slots, decision=scope)
            pending.pages, pending.slots = remaining_pages, remaining_slots
            self._pages, self._slots = all_pages, all_slots
            if scope.operation == "return":
                self._finish_close(close)
            else:
                for sequence in tuple(pending.closes):
                    self._finish_close(self._closes[sequence])
                self._active.pop(scope.key, None)
                self._keys.pop(scope.key)
                self._wire_keys.pop(event_key(scope.key), None)
                self._published.pop(pending.plan.sequence, None)
                self._published_keys.discard(scope.key)
                self._ready.discard(scope.key)
                self._dirty.discard(scope.key)
                self._withdrawn.discard(scope.key)
                self._cancelled.discard(scope.key)
                self._retired.add(scope.key)
                if self.client.rank == 0:
                    wire_key = decision_key(scope)
                    self._retire_pending[wire_key] = scope
                    self._retire_dirty.add(wire_key)

    def _finish_close(self, close):
        scope = close.scope
        self._closes.pop(scope.sequence)
        self._closing_active.pop(scope.sequence, None)
        self._close_wire.pop(decision_key(scope), None)
        self._group_fenced.discard(scope.sequence)
        self._fence_dirty.discard(scope.sequence)
        self._keys[scope.key].closes.discard(scope.sequence)
        if isinstance(scope, ReturnPlan):
            self._subset_close.pop((scope.key, scope.return_id), None)
        else:
            self._whole_close.pop(scope.key, None)

    def _progress_closes(self):
        for _ in range(min(self.progress_budget, len(self._closing_queue))):
            sequence = self._closing_queue.popleft()
            close = self._closing_active.get(sequence)
            if close is None:
                continue
            self._closing_queue.append(sequence)
            pending = self._keys[close.scope.key]
            if close.future.done() and close.proof is None and close.error is None:
                try:
                    proof = close.future.result()
                    if (not isinstance(proof, FenceProof) or proof.key != close.scope.key
                            or type(proof.close_sequence) is not int or proof.close_sequence != sequence
                            or type(proof.quiet) is not bool or type(proof.unreferenced) is not bool
                            or not proof.quiet or not proof.unreferenced):
                        raise WorksetProtocolError("close Future returned no exact complete physical/reference proof")
                    close.proof = proof
                except Exception as error:
                    close.error = error
            # An unresolved preparation task can still submit work, even when
            # an external close observer prematurely reports completion.
            preparation_drained = (pending.future is None or pending.future.done()) and (
                not pending.prepare_started or pending.prepare_error is not None
                or (pending.result_received and pending.descriptor_ready))
            if close.proof is not None and preparation_drained:
                self.client.report(self.namespace + ":fenced", decision_key(close.scope), 1)
                close.reported = True
                self._fence_dirty.add(sequence)
            if close.reported or close.error is not None:
                self._closing_active.pop(sequence)
                self._closing_queue.pop()

    def _update_fenced(self):
        if self.client.rank != 0:
            self._fence_dirty.clear()
            return
        for sequence in tuple(self._fence_dirty):
            close = self._closes.get(sequence)
            if close is not None and sequence not in self._group_fenced:
                key = decision_key(close.scope)
                if (self.client.command_complete(self.namespace + ":decisions", key)
                        and self.client.group_status(self.namespace + ":fenced", key) == 1):
                    self._group_fenced.add(sequence)
                    self._fenced.append(close.scope)
            self._fence_dirty.discard(sequence)

    def drain_fenced(self, limit=128):
        """Leader consumes each all-rank exact close once, for ledger commit.

        Feed quiet=True/unreferenced=True for every group rank to the matching
        ledger report method, then commit and publish its return/free decision.
        No ledger mutation or automatic memory reuse occurs in this method.
        """
        self._check()
        if self.client.rank != 0:
            raise ValueError("only rank zero consumes group fences")
        _positive(limit, "fence drain limit")
        self._updates()
        self._update_fenced()
        result = []
        for _ in range(min(limit, len(self._fenced))):
            scope = self._fenced.popleft()
            if scope.sequence in self._closes and scope.key not in self._free_published:
                result.append(scope)
        # A whole-close proof subsumes still-pending subset observers. Avoid
        # handing the writer a stale subset after it commits the same batch's
        # whole free; an already committed subset remains excluded by ledger
        # remaining-range accounting.
        whole = {scope.key for scope in result if isinstance(scope, LedgerDecision)}
        return [scope for scope in result if isinstance(scope, LedgerDecision) or scope.key not in whole]

    def _updates(self):
        for namespace in (self.namespace, self.namespace + ":prepared"):
            for wire_key in self.client.drain_update_keys(namespace):
                key = self._wire_keys.get(wire_key)
                if key is None:
                    continue  # Command is still buffered; installation reads its cache.
                self._dirty.add(key)
                if namespace == self.namespace:
                    pending = self._keys[key]
                    receipt = self.client.receipt(namespace, wire_key)
                    if receipt is not None and receipt < 0:
                        pending.cancelled = True
                        self._ready.discard(key)
                        if pending.installed:
                            self._activate(pending)
        for namespace in (self.namespace + ":decisions", self.namespace + ":fenced"):
            for wire_key in self.client.drain_update_keys(namespace):
                if namespace == self.namespace + ":decisions" and wire_key in self._retire_pending:
                    self._retire_dirty.add(wire_key)
                sequence = self._close_wire.get(wire_key)
                if sequence is not None:
                    self._fence_dirty.add(sequence)

    def _retire_events(self):
        """Drop terminal control history, never infer physical completion.

        FREE was only published after the exact close fence; the final command
        ACK additionally proves every rank installed the CPU release. The
        server repeats that check and retains compact anti-replay authority.
        No network wait or historical scan is added to scheduler/Forward.
        """
        for _ in range(min(self.progress_budget, len(self._retire_dirty))):
            wire_key = self._retire_dirty.pop()
            scope = self._retire_pending[wire_key]
            if self.client.command_complete(self.namespace + ":decisions", wire_key):
                self.client.retire_workset(self.namespace, event_key(scope.key), wire_key)
                del self._retire_pending[wire_key]
        if self._retire_dirty:
            self.client.changed.set()

    def _update_ready(self):
        if self.client.rank != 0:
            self._dirty.clear()
            return
        for key in tuple(self._dirty):
            pending = self._keys[key]
            wire_key = event_key(key)
            receipt = self.client.receipt(self.namespace, wire_key)
            ready = (pending.installed and key not in self._cancelled and key not in self._withdrawn and not pending.cancelled
                     and not (receipt is not None and receipt < 0)
                     and self.client.command_complete(self.namespace, wire_key)
                     and self.client.group_status(self.namespace + ":prepared", wire_key) == 1)
            if ready:
                self._ready.add(key)
            else:
                self._ready.discard(key)
            self._dirty.remove(key)

    def progress(self):
        """Incremental commands plus budgeted active preparation; never wait."""
        self._check()
        try:
            self._receive()
            for _ in range(self.progress_budget):
                pending = self._pending.get(self.installed_sequence + 1)
                if pending is None:
                    break
                if isinstance(pending, _DecisionPending):
                    if not pending.applied:
                        try:
                            self._install_decision(pending)
                        except WorksetProtocolError:
                            raise
                        except Exception as error:
                            pending.install_error = error
                            break
                        pending.applied, pending.install_error = True, None
                    self.client.ack_command(self.namespace + ":decisions", decision_key(pending.scope), pending.command_id)
                    self.installed_sequence = pending.scope.sequence
                    self._pending.pop(self.installed_sequence)
                    continue
                plan = pending.plan
                pages = _normalize(_nonoverlapping(self._pages + plan.parent_pages + plan.suffix_pages))
                slots = _normalize(_nonoverlapping(self._slots + plan.checkpoint_slots + plan.runtime_slots))
                try:
                    # A retry must be idempotent in the injected CPU adapter.
                    self.adapter.install(plan)
                except Exception as error:
                    pending.install_error = error
                    break  # CPU ordering is mandatory; only Futures may finish out of order.
                pending.installed, pending.install_error = True, None
                pending.pages = _normalize(plan.parent_pages + plan.suffix_pages)
                pending.slots = _normalize(plan.checkpoint_slots + plan.runtime_slots)
                self._pages, self._slots = pages, slots
                self.installed_sequence = plan.sequence
                self._pending.pop(plan.sequence)
                self._activate(pending)
                self._dirty.add(plan.key)
            self._updates()
            for _ in range(min(self.progress_budget, len(self._active_queue))):
                lease_key = self._active_queue.popleft()
                pending = self._active.get(lease_key)
                if pending is None:
                    continue
                # Keep this attempt scheduled even if a client submission fails.
                self._active_queue.append(lease_key)
                key = event_key(pending.plan.key)
                if not pending.acked:
                    # CPU installation, not preparation or physical completion.
                    self.client.ack_command(self.namespace, key, pending.command_id)
                    pending.acked = True
                receipt = self.client.receipt(self.namespace, key)
                pending.cancelled |= pending.plan.key in self._cancelled or (receipt is not None and receipt < 0)
                if pending.cancelled and not pending.cancel_applied:
                    try:
                        self.adapter.cancel(pending.plan)
                        pending.cancel_applied, pending.cancel_error = True, None
                    except Exception as error:
                        pending.cancel_error = error
                if not pending.prepare_started and not pending.cancelled:
                    pending.prepare_started = True
                    try:
                        future = self.adapter.prepare(pending.plan)
                        if not isinstance(future, Future):
                            raise TypeError("prepare must return a retained Future")
                        pending.future = future
                    except Exception as error:
                        pending.prepare_error = error
                if pending.future is not None and pending.future.done() and not pending.result_received and pending.prepare_error is None:
                    try:
                        pending.descriptor = pending.future.result()
                        pending.result_received = True
                        if getattr(pending.descriptor, "plan", None) != pending.plan:
                            raise TypeError("prepared descriptor does not match its exact workset plan")
                        if not callable(getattr(pending.descriptor, "is_ready", None)):
                            raise TypeError("prepared descriptor requires an explicit is_ready fence")
                    except Exception as error:
                        pending.prepare_error = error
                if pending.result_received and not pending.descriptor_ready and pending.prepare_error is None:
                    try:
                        ready = pending.descriptor.is_ready()
                        if type(ready) is not bool:
                            raise TypeError("descriptor fence must report explicit boolean readiness")
                        pending.descriptor_ready = ready
                    except Exception as error:
                        pending.prepare_error = error
                if pending.cancelled or pending.prepare_error is not None:
                    self.client.report(self.namespace + ":prepared", key, -1)
                elif pending.descriptor_ready:
                    self.client.report(self.namespace + ":prepared", key, 1)
                terminal = pending.descriptor_ready or pending.prepare_error is not None or (
                    pending.cancelled and not pending.prepare_started)
                if terminal and (not pending.cancelled or pending.cancel_applied):
                    del self._active[lease_key]
                    self._active_queue.pop()
            self._progress_closes()
            self._update_ready()
            self._update_fenced()
            self._retire_events()
        except WorksetProtocolError as error:
            self._error = str(error)
            raise

    def ready_snapshot(self):
        """Rank-zero all-rank prepared facts, NOT local-Future compute admission."""
        self._check()
        if self.client.rank != 0:
            raise ValueError("only rank zero freezes group readiness")
        self._updates()
        self._update_ready()
        return ReadyCut(self.incarnation, self.installed_sequence,
                        tuple(sorted(self._ready, key=lambda key: self._keys[key].plan.sequence)))

    def status(self, key: LeaseKey):
        """Diagnostic only; a local ready result is not a batch/start grant."""
        self._check()
        pending = self._keys.get(key)
        if pending is None:
            return None
        return {"installed": pending.installed, "descriptor_ready": pending.descriptor_ready,
                "cancelled": pending.cancelled, "future_pending": pending.future is not None and not pending.future.done(),
                "install_error": pending.install_error, "prepare_error": pending.prepare_error,
                "cancel_error": pending.cancel_error,
                "remaining_pages": pending.pages, "remaining_slots": pending.slots}

    def close_status(self, scope):
        """Exact local diagnostic, never substitute it for all-rank fencing."""
        self._check()
        close = self._closes.get(scope.sequence)
        if close is None or close.scope != scope:
            return None
        return {"reported": close.reported, "error": close.error,
                "future_pending": close.future is not None and not close.future.done()}

    def local_view(self, key):
        """Progress-thread accessor; adapters synchronize cross-thread use."""
        self._check()
        pending = self._keys.get(key)
        if pending is None:
            return None
        return LocalWorksetView(pending.plan, pending.installed, pending.descriptor,
                                pending.descriptor_ready, pending.cancelled,
                                pending.pages, pending.slots)

    def may_use(self, key, *, pages=(), slots=()):
        """CPU range authorization only; caller still requires group readiness.

        Adapter close and native/transport acquisition must share their own
        synchronization boundary; this progress-thread query is not a lock.
        """
        self._check()
        pending = self._keys.get(key)
        if pending is None or not pending.installed or pending.cancelled or key in self._cancelled:
            return False
        try:
            available_pages, available_slots = pending.pages, pending.slots
            for sequence in pending.closes:
                scope = self._closes[sequence].scope
                if isinstance(scope, ReturnPlan):
                    available_pages = _subtract(available_pages, scope.pages)
                    available_slots = _subtract(available_slots, scope.slots)
            _subtract(available_pages, _normalize(pages))
            _subtract(available_slots, _normalize(slots))
        except ValueError:
            return False
        return True
