"""P allocator actor wiring, deliberately opt-in and not model admission.

This facade joins the real CPU authority, ordered TP executor, descriptor worker,
and existing I/O lease broker. Native compute still needs its original broadcast
cut. ``reference_fence`` is mandatory: no timeout, Python task completion, or
shutdown is substituted for the actual last-reference/DMA/compute proof.
"""
from concurrent.futures import Future, ThreadPoolExecutor
from collections import deque
import threading
import time

import torch

from sglang.srt.disaggregation.agentic_workset_controller import (
    ControllerQueueFull, PendingWorksetCancelled, WorksetIntent,
)
from sglang.srt.disaggregation.agentic_workset_device import WorksetPreparationFailure, prepare_workset
from sglang.srt.disaggregation.agentic_workset_ledger import LedgerCounts, LedgerDecision, ReturnPlan, WorksetPlan
from sglang.srt.disaggregation.agentic_workset_tp import FenceProof, ReadyCut, TPWorksetExecutor


class WorksetRuntimeStopped(RuntimeError):
    pass


def _future():
    future = Future()
    future.set_running_or_notify_cancel()
    return future


class PWorksetRuntime:
    """One control actor per P rank; rank zero alone owns the CPU controller.

    Construct before admitting work, against an empty broker and ledger. Public
    request/close methods only enqueue CPU records. Descriptor preparation and
    its CUDA wait run on a separate single worker, never on Forward or on the
    ledger writer. Native raw-free receipts are already producer-event fenced;
    they are paired across ranks by exact address scope, NOT local ticket order.
    """

    def __init__(self, client, broker, *, controller=None, device="cpu",
                 page_capacity, mamba_pool=None, incarnation, page_size,
                 mamba_slots=0, reference_fence, release_callback=None,
                 max_pending=4096, namespace="p-workset-runtime", dedicated_client):
        if dedicated_client is not True or getattr(client, "_workset_runtime_owner", None) is not None:
            raise ValueError("runtime needs an exclusive TPEventClient; changed event cannot have another consumer")
        if (client.rank == 0) != (controller is not None):
            raise ValueError("only rank zero must supply the authority controller")
        if not callable(reference_fence) or type(max_pending) is not int or max_pending < 1:
            raise ValueError("a real reference fence and bounded queue are required")
        if broker._leases or broker._tp_plan_epoch >= 0:
            raise ValueError("runtime requires an empty broker without native allocation plans")
        if controller is not None and controller.ledger.counts.sequence:
            raise ValueError("runtime requires a fresh authority log")
        self.client, self.broker, self.controller = client, broker, controller
        client._workset_runtime_owner = self
        self.device, self.mamba_pool = torch.device(device), mamba_pool
        if self.device.type == "cuda" and self.device.index is None:
            # Native model initialization selected the owning GPU on THIS
            # thread. A new preparation thread has an independent default.
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.page_capacity, self.page_size = page_capacity, page_size
        self._reference_fence, self._release_callback = reference_fence, release_callback
        self._max_pending = max_pending
        self._lock = threading.Condition(threading.RLock())
        self._queue, self._jobs, self._public = deque(), set(), set()
        self._closing, self._error = False, None
        self._plans, self._descriptors, self._prepare_jobs = {}, {}, {}
        self._prepare_errors, self._fence_joins = {}, {}
        self._installed_keys = set()
        self._views, self._dirty_views = {}, set()
        self._auto_retire_empty = False
        self._publishing, self._next_publish = {}, 1
        self._cut = ReadyCut(incarnation, 0, ())
        self._counts = LedgerCounts(0, page_capacity, mamba_slots, 0)
        self._free_pages, self._free_slots, self._live = page_capacity, mamba_slots, 0
        self._native_receipts, self._native_closes = {}, {}
        self._prep = ThreadPoolExecutor(1, thread_name_prefix="p-workset-prepare")
        self._stream = None
        self.executor = TPWorksetExecutor(client, self, incarnation=incarnation,
            page_count=page_capacity, page_size=page_size, mamba_slots=mamba_slots,
            namespace=namespace)
        # Once attached, no accidental legacy service() may choose addresses.
        broker._controller_owned = True
        self._thread = threading.Thread(target=self._run, name="p-workset-runtime", daemon=True)
        bind_actor = getattr(broker, "_bind_runtime_actor", None)
        if bind_actor is not None:
            bind_actor(self)
        self._thread.start()

    def check_health(self):
        with self._lock:
            if self._closing or self._error is not None:
                raise WorksetRuntimeStopped("P workset runtime stopped; ownership retained") from self._error

    def _enqueue(self, fn, *args):
        future = _future()
        with self._lock:
            self.check_health()
            if len(self._queue) + len(self._jobs) >= self._max_pending:
                raise WorksetRuntimeStopped("runtime queue full; operation not accepted")
            self._queue.append((fn, args, future))
            self._public.add(future)
        self.client.changed.set()
        return future

    def _leader(self):
        if self.controller is None:
            raise ValueError("only rank zero selects allocator decisions")

    def request(self, intent: WorksetIntent):
        self._leader()
        return self._enqueue(self._authority, "request", (intent,))

    def cancel(self, key):
        self._leader()
        return self._enqueue(self._authority, "cancel", (key,))

    def cancel_pending(self, intent):
        self._leader()
        return self._enqueue(self._authority, "cancel_pending", (intent,))

    def enable_empty_retirement(self):
        """Opt-in before ingress: retire a provenance record only when empty.

        Every address has already passed its own all-rank return fences. The
        ordered empty close/free still goes through the normal TP log.
        """
        with self._lock:
            if self._counts.sequence:
                raise RuntimeError("enable empty retirement before admitting work")
            self._auto_retire_empty = True

    def begin_return(self, key, return_id, *, pages=(), slots=()):
        self._leader()
        return self._enqueue(self._authority, "begin_return", (key, return_id),
                             dict(pages=tuple(pages), slots=tuple(slots)))

    def native_free(self, receipt):
        """Accept an already fenced native last-ref receipt, not a free ACK."""
        return self._enqueue(self._native_free, receipt)

    def ready_cut(self):
        self.check_health()
        if self.client.rank != 0:
            raise ValueError("only rank zero selects the native compute ready cut")
        with self._lock:
            return self._cut

    def counts(self):
        """Local installed authority mirror, safe for native admission reads."""
        self.check_health()
        with self._lock:
            return self._counts

    def local_view(self, key):
        """Immutable local installed view; NOT a TP compute authorization."""
        self.check_health()
        with self._lock:
            return self._views.get(key)

    def conservation_snapshot(self):
        """One actor publication cut; idle diagnostics may inspect live owners."""
        self.check_health()
        with self._lock:
            return self._counts, tuple(self._views.values())

    def wait_ready(self, key, timeout=None):
        """Driver/worker convenience only; never call from model scheduling."""
        end = None if timeout is None else time.monotonic() + timeout
        with self._lock:
            while True:
                self.check_health()
                if key in self._installed_keys and key in self._views and (
                    self.client.rank != 0 or key in self._cut.ready
                ):
                    break
                remaining = None if end is None else end - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("workset did not become ready")
                self._lock.wait(remaining)
        # Never hold the runtime lock while acquiring the broker lock: native
        # prefix adoption holds the broker lock while enqueueing an exact return.
        lease = self.broker.get(key.snapshot_id)
        if lease is None or lease.controller_plan.key != key:
            raise RuntimeError("prepared lease was already handed off or retired")
        return lease  # Follower result is local, NOT group admission.

    def _watch(self, source, callback, on_error=None):
        self._jobs.add(source)
        def done(future):
            with self._lock:
                # Results remain retained by _jobs after stop; never lose a
                # late physical/ownership operation by cancelling its Future.
                if self._closing:
                    return
                self._queue.append((self._finished, (future, callback, on_error), None))
            self.client.changed.set()
        source.add_done_callback(done)

    def _finished(self, source, callback, on_error):
        try:
            result = source.result()
        except Exception as error:
            if on_error is None:
                raise
            on_error(error)
        else:
            callback(result)
        self._jobs.discard(source)

    def _authority(self, method, args, kwargs=None):
        result = _future()
        source = getattr(self.controller, method)(*args, **(kwargs or {}))
        def completed(value):
            values = value if isinstance(value, tuple) else (value,)
            for decision in values:
                if isinstance(decision, (WorksetPlan, LedgerDecision, ReturnPlan)):
                    old = self._publishing.get(decision.sequence)
                    if old is not None and old != decision:
                        raise RuntimeError("conflicting allocator sequence")
                    if decision.sequence >= self._next_publish:
                        self._publishing[decision.sequence] = decision
                    if (self._auto_retire_empty and isinstance(decision, LedgerDecision)
                            and decision.operation == "return"):
                        view = self.controller.ledger.view(decision.key)
                        if view is not None and not view.remaining_pages and not view.remaining_slots:
                            self._authority("cancel", (decision.key,))
            result.set_result(value)
        def failed(error):
            if isinstance(error, (ValueError, PendingWorksetCancelled, ControllerQueueFull)):
                result.set_exception(error)
            else:
                raise error
        self._watch(source, completed, failed)
        return result

    def _publish(self):
        while self._next_publish in self._publishing:
            decision = self._publishing[self._next_publish]
            if isinstance(decision, WorksetPlan):
                self.executor.publish_grant(decision)
            else:
                self.executor.publish_decision(decision)
            del self._publishing[self._next_publish]
            self._next_publish += 1

    def _group_fenced(self, scope):
        # drain_fenced is the existing executor's exact all-rank proof, never
        # one local rank or a successful task submission.
        futures = []
        for rank in range(self.client.size):
            if isinstance(scope, ReturnPlan):
                future = self.controller.report_return_fence(scope, rank, quiet=True, unreferenced=True)
            else:
                future = self.controller.report_fence(scope.key, rank, quiet=True, unreferenced=True)
            futures.append(future)
        # Controller processes these FIFO, so the commit cannot overtake its
        # facts. Retain every report error rather than interpreting it as ACK.
        for future in futures:
            self._watch(future, lambda _: None)
        method, value = ("commit_return", scope) if isinstance(scope, ReturnPlan) else ("free", scope.key)
        self._authority(method, (value,))

    # TPWorksetExecutor adapter API, called only by this actor.
    def install(self, plan):
        self._dirty_views.add(plan.key)
        previous = self._plans.get(plan.key)
        if previous is not None and previous != plan:
            raise RuntimeError("workset identity changed")
        self._plans[plan.key] = plan
        if previous is None:
            self._free_pages -= sum(r.count for r in plan.parent_pages + plan.suffix_pages)
            self._free_slots -= sum(r.count for r in plan.checkpoint_slots + plan.runtime_slots)
            self._live += 1

    def _prepare(self, plan):
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            if self._stream is None:
                self._stream = torch.cuda.Stream(device=self.device)
        try:
            descriptor = prepare_workset(plan, device=self.device,
                page_capacity=self.page_capacity, mamba_pool=self.mamba_pool, stream=self._stream)
        except WorksetPreparationFailure as error:
            if error.ready_event is not None:
                try:
                    error.ready_event.synchronize()
                except Exception:
                    pass  # Unknown device fence remains quarantined.
            raise
        if descriptor.ready_event is not None:
            descriptor.ready_event.synchronize()  # This dedicated worker only.
        return descriptor

    def prepare(self, plan):
        future = self._prep.submit(self._prepare, plan)
        self._prepare_jobs[plan.key] = future
        installed = _future()
        def completed(descriptor):
            self._prepared(descriptor)
            installed.set_result(descriptor)
        def failed(error):
            self._preparation_failed(plan, error)
            installed.set_exception(error)
        self._watch(future, completed, failed)
        # TP preparation ACK includes local broker publication, not just the
        # raw device worker finishing before its actor callback was consumed.
        return installed

    def _prepared(self, descriptor):
        self._dirty_views.add(descriptor.plan.key)
        self._descriptors[descriptor.plan.key] = descriptor
        # A close may have arrived before the device task finished. Never
        # resurrect a lease after that authoritative local close.
        view = self.executor.local_view(descriptor.plan.key)
        if view is not None and not view.closing:
            self.broker.install_prepared(descriptor)
        with self._lock:
            if view is not None and not view.closing:
                self._installed_keys.add(descriptor.plan.key)
            self._lock.notify_all()
        self._join_fences(descriptor.plan.key)

    def _preparation_failed(self, plan, error):
        self._prepare_errors[plan.key] = error
        self._join_fences(plan.key)

    def _join_fences(self, key):
        for scope, source, combined in tuple(self._fence_joins.get(key, ())):
            if combined.done() or not source.done():
                continue
            preparation = self._prepare_jobs.get(key)
            if preparation is not None:
                if not preparation.done():
                    continue
                try:
                    prepared = preparation.result()
                    if not prepared.is_ready():
                        continue
                except Exception as error:
                    if not isinstance(error, WorksetPreparationFailure) or not error.is_quiescent():
                        continue  # Never infer quiet from an arbitrary exception.
            try:
                combined.set_result(source.result())
            except Exception as error:
                combined.set_exception(error)

    def close(self, plan, scope):
        self._dirty_views.add(plan.key)
        if not isinstance(scope, ReturnPlan):
            with self.broker._lock:
                self.broker._tp_retire_requested.add(plan.key.snapshot_id)
        if isinstance(scope, ReturnPlan) and scope.return_id.startswith("native:"):
            future = _future()
            self._native_closes[scope.sequence] = (plan, scope, future)
            self._native_progress()
        else:
            future = self._reference_fence(plan, scope)
            if not isinstance(future, Future):
                raise TypeError("reference_fence must return a retained Future")
        combined = _future()
        self._fence_joins.setdefault(plan.key, []).append((scope, future, combined))
        # Local preparation and the external native/DMA/reference observer
        # must BOTH settle; one callback cannot vouch for the other worker.
        self._watch(future, lambda _: self._join_fences(plan.key),
                    lambda _: self._join_fences(plan.key))
        combined.add_done_callback(lambda _: self.client.changed.set())
        self._join_fences(plan.key)
        return combined

    def release(self, plan, pages, slots, *, decision):
        self._dirty_views.add(plan.key)
        if self._release_callback is not None:
            self._release_callback(plan, pages, slots)
        # No GPU allocator/free operation is performed. The authority already
        # committed the exact all-rank return; local executor applies its log.
        released_pages = {i for run in pages for i in range(run.start, run.end)}
        released_slots = {i for run in slots for i in range(run.start, run.end)}
        self._free_pages += len(released_pages)
        self._free_slots += len(released_slots)
        for ticket, (receipt, addresses) in tuple(self._native_receipts.items()):
            addresses.difference_update(released_pages if receipt.resource == "attention" else released_slots)
            if not addresses:
                del self._native_receipts[ticket]
        # A complete retirement can remove only this exact broker generation.
        if decision.operation == "free":
            with self._lock:
                self._installed_keys.discard(plan.key)
            self._live -= 1
            self._plans.pop(plan.key, None)
            self._descriptors.pop(plan.key, None)
            self._prepare_jobs.pop(plan.key, None)
            self._prepare_errors.pop(plan.key, None)
            self._fence_joins.pop(plan.key, None)
            with self.broker._lock:
                self.broker._controller_retired.add(plan.key)
                lease = self.broker._leases.get(plan.key.snapshot_id)
                if lease is None or lease.controller_plan == plan:
                    self.broker._leases.pop(plan.key.snapshot_id, None)
                    self.broker._release_requested.pop(plan.key.snapshot_id, None)
                    self.broker._tp_retire_requested.discard(plan.key.snapshot_id)
            callback = getattr(self.broker, "controller_retired", None)
            if callback is not None:
                callback(plan)
        else:
            self._fence_joins[plan.key] = [item for item in self._fence_joins.get(plan.key, ())
                if not (isinstance(item[0], ReturnPlan) and item[0].return_id == decision.return_id)]
        for sequence, (_, scope, future) in tuple(self._native_closes.items()):
            if (scope.key == plan.key and future.done()
                    and {i for r in scope.pages for i in range(r.start, r.end)} <= released_pages
                    and {i for r in scope.slots for i in range(r.start, r.end)} <= released_slots):
                del self._native_closes[sequence]

    def _native_free(self, receipt):
        if receipt.rank != self.client.rank or receipt.resource not in {"attention", "mamba"}:
            raise ValueError("foreign native receipt")
        previous = self._native_receipts.get(receipt.ticket_id)
        if previous is not None:
            if previous[0] != receipt:
                raise ValueError("native receipt identity changed")
            return True
        divisor = self.page_size if receipt.resource == "attention" else 1
        self._native_receipts[receipt.ticket_id] = (receipt, {i // divisor for i in receipt.indices})
        self._native_progress()
        if self.controller is not None:
            return self._authority("native_return", (receipt.ticket_id, receipt.resource,
                receipt.indices, receipt.allocation_sequence))
        return True

    def _native_progress(self):
        for sequence, (plan, scope, future) in tuple(self._native_closes.items()):
            if future.done():
                continue
            pages, slots = set(), set()
            for receipt, addresses in self._native_receipts.values():
                if receipt.allocation_sequence >= plan.sequence:
                    (pages if receipt.resource == "attention" else slots).update(addresses)
            wanted_pages = {i for run in scope.pages for i in range(run.start, run.end)}
            wanted_slots = {i for run in scope.slots for i in range(run.start, run.end)}
            if wanted_pages <= pages and wanted_slots <= slots:
                future.set_result(FenceProof(plan.key, sequence, True, True))

    def _run(self):
        try:
            while True:
                self.client.changed.wait()
                self.client.changed.clear()
                with self._lock:
                    if self._closing:
                        break
                    batch = [self._queue.popleft() for _ in range(min(128, len(self._queue)))]
                for fn, args, public in batch:
                    result = fn(*args)
                    if public is not None:
                        if isinstance(result, Future):
                            self._watch(result, public.set_result, public.set_exception)
                        else:
                            public.set_result(result)
                self._publish()
                self.executor.progress()
                if self.controller is not None:
                    for scope in self.executor.drain_fenced():
                        self._group_fenced(scope)
                    cut = self.executor.ready_snapshot()
                with self._lock:
                    for key in self._dirty_views:
                        view = self.executor.local_view(key)
                        if key in self._plans:
                            self._views[key] = view
                        else:
                            self._views.pop(key, None)
                    self._dirty_views.clear()
                    self._counts = LedgerCounts(self.executor.installed_sequence,
                        self._free_pages, self._free_slots, self._live)
                    if self.controller is not None:
                        self._cut = cut
                    self._lock.notify_all()
                    self._public = {future for future in self._public if not future.done()}
                    if self._queue:
                        self.client.changed.set()
        except BaseException as error:
            with self._lock:
                self._error = error
        finally:
            with self._lock:
                self._closing = True
                failure = WorksetRuntimeStopped(f"runtime stopped; ownership retained: {self._error}")
                failure.__cause__ = self._error
                for future in self._public:
                    if not future.done():
                        future.set_exception(failure)
                for _, _, future in self._queue:
                    if future is not None and not future.done():
                        future.set_exception(WorksetRuntimeStopped("runtime stopped; ownership retained"))
                self._lock.notify_all()

    def shutdown(self, *, timeout=5):
        with self._lock:
            self._closing = True
            self._lock.notify_all()
        self.client.changed.set()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout)
        # An unresolved CUDA task must remain alive with its buffers/ownership.
        self._prep.shutdown(wait=False, cancel_futures=False)
        if self.controller is not None:
            self.controller.shutdown(wait=False)
        return not self._thread.is_alive()
