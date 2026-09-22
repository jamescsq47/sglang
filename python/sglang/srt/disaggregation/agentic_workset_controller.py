"""Event-driven, CPU-only single writer for WorksetLedger.

This component is not a runtime adapter: it performs no CUDA work, transport,
Radix mutation, or TP broadcast. A returned address plan is not an I/O-ready
or runnable workset. Callers must provide the ledger's exact physical and
reference fences; shutdown deliberately retains outstanding ownership.

Pending grants are retried only after actual resource returns, in insertion
order without a Direct/Slow/fresh priority. There is no timer or capacity poll.
Future callbacks run on the writer: only enqueue nonblocking completion work,
never prepare CUDA, call network RPC, or wait for an ACK inside callbacks.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import Future
from dataclasses import dataclass
import threading

from sglang.srt.disaggregation.agentic_workset_ledger import (
    LeaseKey, PageRun, ReturnPlan, WorksetLedger,
)


class ControllerStopped(RuntimeError):
    pass


class ControllerQueueFull(RuntimeError):
    pass


class PendingWorksetCancelled(RuntimeError):
    """This intent was cancelled before it acquired any addresses."""


@dataclass(frozen=True)
class WorksetIntent:
    snapshot_id: str
    attempt_id: str
    owner: str
    parent_tokens: int
    prompt_tokens: int
    checkpoint_slots: int = 0
    runtime_slots: int = 0

    def __post_init__(self):
        for name in ("snapshot_id", "attempt_id", "owner"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("parent_tokens", "prompt_tokens", "checkpoint_slots", "runtime_slots"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.prompt_tokens == 0 or self.prompt_tokens < self.parent_tokens:
            raise ValueError("a nonempty prompt must contain its parent")

    @property
    def identity(self):
        return self.snapshot_id, self.attempt_id


@dataclass(frozen=True)
class _Command:
    method: str
    args: tuple
    kwargs: tuple
    future: Future


class WorksetController:
    """An unmodified ledger becomes worker-owned on its first mutation.

    ``max_queue`` bounds queued commands, ``max_pending`` bounds waiting grant
    Futures (including duplicates), and ``max_cancelled`` bounds run-retained
    cancelled-intent tombstones. Overflow is explicit, never silent eviction.
    No other thread may mutate this ledger after controller construction.
    """

    def __init__(self, ledger: WorksetLedger, *, max_queue=4096,
                 max_pending=4096, max_cancelled=100000):
        for value in (max_queue, max_pending, max_cancelled):
            if type(value) is not int or value < 1:
                raise ValueError("controller bounds must be positive integers")
        self.ledger = ledger
        self._max_queue, self._max_pending = max_queue, max_pending
        self._max_cancelled = max_cancelled
        self._commands = deque()
        self._condition = threading.Condition()
        self._closing = False
        self._failure = None
        self._pending = OrderedDict()
        self._pending_count = 0
        self._cancelled = set()
        self._thread = threading.Thread(target=self._run, name="p-workset-controller", daemon=True)
        self._thread.start()

    def _submit(self, method, *args, **kwargs):
        future = Future()
        # Future.cancel() is not an ownership transition. In particular it
        # must never discard a grant that raced a consumer's cancellation.
        future.set_running_or_notify_cancel()
        command = _Command(method, args, tuple(kwargs.items()), future)
        with self._condition:
            if self._closing:
                raise ControllerStopped("workset controller stopped; ownership retained") from self._failure
            if len(self._commands) >= self._max_queue:
                raise ControllerQueueFull("workset command queue is full; command not accepted")
            self._commands.append(command)
            self._condition.notify()
        return future

    def request(self, intent: WorksetIntent):
        if not isinstance(intent, WorksetIntent):
            raise TypeError("an immutable WorksetIntent is required")
        return self._submit("request", intent)

    def cancel_pending(self, intent: WorksetIntent):
        """True cancels an ungranted intent; False requires exact lease cancel.

        If grant won, its original Future still yields the plan. No capacity
        is freed by this operation, nor by cancelling a granted lease.
        """
        if not isinstance(intent, WorksetIntent):
            raise TypeError("an immutable WorksetIntent is required")
        return self._submit("cancel_pending", intent)

    def cancel(self, key: LeaseKey):
        return self._submit("cancel", self._key(key))

    def report_fence(self, key: LeaseKey, rank: int, *, quiet: bool, unreferenced: bool):
        self._facts(rank, quiet, unreferenced)
        return self._submit("report_fence", self._key(key), rank,
                            quiet=quiet, unreferenced=unreferenced)

    def free(self, key: LeaseKey):
        return self._submit("free", self._key(key))

    def begin_return(self, key: LeaseKey, return_id: str, *, pages=(), slots=()):
        if not isinstance(return_id, str) or not return_id:
            raise ValueError("return_id must be a nonempty string")
        return self._submit("begin_return", self._key(key), return_id,
                            pages=self._runs(pages), slots=self._runs(slots))

    def report_return_fence(self, plan: ReturnPlan, rank: int, *, quiet: bool, unreferenced: bool):
        self._facts(rank, quiet, unreferenced)
        return self._submit("report_return_fence", self._return_plan(plan), rank,
                            quiet=quiet, unreferenced=unreferenced)

    def commit_return(self, plan: ReturnPlan):
        return self._submit("commit_return", self._return_plan(plan))

    def native_return(self, return_id: str, resource: str, indices, allocation_sequence: int):
        """Native-only subset close; no implicit rank fence or address free."""
        frozen = tuple(indices)
        if any(type(index) is not int or index < 0 for index in frozen):
            raise ValueError("native indices must be nonnegative CPU integers")
        if (not isinstance(return_id, str) or not return_id
                or resource not in {"attention", "mamba"}
                or type(allocation_sequence) is not int or allocation_sequence < 0):
            raise ValueError("invalid native return identity")
        return self._submit("native_return", return_id, resource, frozen, allocation_sequence)

    @staticmethod
    def _key(key):
        if not isinstance(key, LeaseKey):
            raise TypeError("an exact immutable LeaseKey is required")
        return key

    @staticmethod
    def _runs(runs):
        result = tuple(runs)
        if any(not isinstance(run, PageRun) for run in result):
            raise TypeError("immutable PageRun ranges are required")
        return result

    @classmethod
    def _return_plan(cls, plan):
        if not isinstance(plan, ReturnPlan):
            raise TypeError("an exact immutable ReturnPlan is required")
        if (not isinstance(plan.return_id, str) or not plan.return_id
                or type(plan.sequence) is not int or plan.sequence < 1):
            raise ValueError("invalid immutable return identity")
        # Freeze even manually constructed dataclasses with list-valued fields.
        return ReturnPlan(cls._key(plan.key), plan.return_id, plan.sequence,
                          cls._runs(plan.pages), cls._runs(plan.slots))

    @staticmethod
    def _facts(rank, quiet, unreferenced):
        if type(rank) is not int or rank < 0 or type(quiet) is not bool or type(unreferenced) is not bool:
            raise ValueError("invalid rank or explicit fence facts")

    def _grant(self, intent):
        if intent.identity in self._cancelled:
            raise PendingWorksetCancelled("cancelled intent cannot acquire a workset")
        return self.ledger.grant(intent.snapshot_id, intent.attempt_id,
            owner=intent.owner, parent_tokens=intent.parent_tokens,
            prompt_tokens=intent.prompt_tokens, checkpoint_slots=intent.checkpoint_slots,
            runtime_slots=intent.runtime_slots)

    def _request(self, intent, future):
        pending = self._pending.get(intent.identity)
        if pending is not None:
            if pending[0] != intent:
                raise ValueError("pending attempt changed its workset shape/owner")
        else:
            plan = self._grant(intent)
            if plan is not None:
                future.set_result(plan)
                return
        if self._pending_count >= self._max_pending:
            raise ControllerQueueFull("waiting workset capacity is full; intent not accepted")
        if pending is None:
            pending = (intent, [])
            self._pending[intent.identity] = pending
        pending[1].append(future)
        self._pending_count += 1

    def _cancel_pending(self, intent):
        pending = self._pending.get(intent.identity)
        if intent.identity in self._cancelled:
            return True
        current = self.ledger.current_view(intent.snapshot_id)
        if current is not None and current.plan.key.attempt_id == intent.attempt_id:
            return False  # Grant won: only the exact lease cancellation may close it.
        if pending is not None and pending[0] != intent:
            raise ValueError("pending cancellation changed its workset identity")
        if len(self._cancelled) >= self._max_cancelled:
            raise ControllerQueueFull("cancelled-intent receipt capacity exhausted")
        self._cancelled.add(intent.identity)
        if pending is None:
            return True  # Close a not-yet-observed intent before it can arrive.
        del self._pending[intent.identity]
        self._pending_count -= len(pending[1])
        for future in pending[1]:
            future.set_exception(PendingWorksetCancelled("intent cancelled before grant"))
        return True

    def _retry_pending(self):
        for identity, (intent, futures) in tuple(self._pending.items()):
            with self._condition:
                if self._closing:
                    return
            failure = None
            try:
                plan = self._grant(intent)
            except (ValueError, PendingWorksetCancelled) as error:
                plan = None
                failure = error
            else:
                if plan is None:
                    continue
            del self._pending[identity]
            self._pending_count -= len(futures)
            for future in futures:
                if failure is None:
                    future.set_result(plan)
                else:
                    future.set_exception(failure)

    def _execute(self, command):
        if command.method == "request":
            self._request(command.args[0], command.future)
            return
        if command.method == "cancel_pending":
            command.future.set_result(self._cancel_pending(command.args[0]))
            return
        before = self.ledger.counts if command.method in {"free", "commit_return"} else None
        result = getattr(self.ledger, command.method)(*command.args, **dict(command.kwargs))
        # Receipt publication occurs before follow-up grants, but never under
        # the ledger ownership lock. Only an actual return wakes waiters.
        command.future.set_result(result)
        if before is not None:
            after = self.ledger.counts
            if (after.free_pages > before.free_pages
                    or after.free_mamba_slots > before.free_mamba_slots):
                self._retry_pending()

    def _run(self):
        try:
            while True:
                with self._condition:
                    while not self._commands and not self._closing:
                        self._condition.wait()
                    if self._closing:
                        break
                    command = self._commands.popleft()
                try:
                    self._execute(command)
                except (ValueError, PendingWorksetCancelled, ControllerQueueFull) as error:
                    command.future.set_exception(error)
                except BaseException as error:
                    if not command.future.done():
                        command.future.set_exception(error)
                    with self._condition:
                        self._failure, self._closing = error, True
                    break
        finally:
            with self._condition:
                self._closing = True
                queued = tuple(self._commands)
                self._commands.clear()
            error = ControllerStopped("workset controller stopped; existing ownership retained")
            for command in queued:
                command.future.set_exception(error)
            for _, futures in self._pending.values():
                for future in futures:
                    if not future.done():
                        future.set_exception(error)
            self._pending.clear()
            self._pending_count = 0

    def shutdown(self, *, wait=True):
        """Stop and fail queued/ungranted operations; never release live leases."""
        with self._condition:
            self._closing = True
            self._condition.notify()
        if wait and threading.current_thread() is not self._thread:
            self._thread.join()

    def check_health(self):
        """Adapters must reject new use after stop, even with an old grant.

        This is not a device-start fence: close/start arbitration still needs
        the exact ledger key and runtime adapter's physical ownership protocol.
        """
        with self._condition:
            if self._closing:
                raise ControllerStopped("workset controller stopped; ownership retained") from self._failure
