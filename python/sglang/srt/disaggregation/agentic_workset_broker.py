"""Opt-in existing P lease FSM backed by the sole asynchronous CPU authority.

No allocator is selected here. The scheduler still owns Req/Radix mutations;
this facade returns only private ranges whose exact local ownership and producer
event are known. Donated ranges return through native last-reference receipts.
"""
from concurrent.futures import Future
import threading

import torch

from sglang.srt.disaggregation.agentic_workset import AgenticPWorksetLeaseBroker
from sglang.srt.disaggregation.agentic_workset_controller import PendingWorksetCancelled, WorksetIntent
from sglang.srt.disaggregation.agentic_workset_ledger import PageRun, ReturnPlan, _normalize
from sglang.srt.disaggregation.agentic_workset_tp import FenceProof


def completed(value=True):
    future = Future()
    future.set_result(value)
    return future


def slice_runs(runs, start, count):
    selected, offset = [], 0
    for run in runs:
        lo, hi = max(start, offset), min(start + count, offset + run.count)
        if hi > lo:
            selected.append(PageRun(run.start + lo - offset, hi - lo))
        offset += run.count
    if sum(r.count for r in selected) != count:
        raise ValueError("slice exceeds exact plan")
    return tuple(selected)


class ControllerWorksetBroker(AgenticPWorksetLeaseBroker):
    controller_mode = True

    def __init__(self, page_size, *, rank, device="cpu", **kwargs):
        super().__init__(page_size, **kwargs)
        self.rank, self.device = rank, torch.device(device)
        self._native_owner = threading.current_thread()
        self._native_cleanup_owner = None
        self._native_cleanup_lock = None
        self.runtime = self.native_bridge = None
        self._requests, self._known_plans = {}, {}
        self._request_epoch = 0
        self._closed_keys, self._external_keys = set(), set()
        self._pending_closed_owners = set()
        self._restore_epochs = {}
        self._permits, self._observers = {}, {}
        self._facade_error = None

    def _bind_runtime_actor(self, runtime):
        """Install internal observer target before peer commands can arrive."""
        with self._lock:
            if self.runtime is not None or runtime.broker is not self or runtime.client.rank != self.rank:
                raise ValueError("one exact P runtime per broker")
            self.runtime = runtime
            runtime.enable_empty_retirement()

    def attach_runtime(self, runtime, *, native_bridge):
        with self._lock:
            if (self.runtime is not runtime or self.native_bridge is not None
                    or runtime.broker is not self or runtime.client.rank != self.rank):
                raise ValueError("one exact P runtime/native bridge per broker")
            self.runtime, self.native_bridge = runtime, native_bridge
            self._controller_owned = True

    def check_health(self):
        if self._facade_error is not None:
            raise RuntimeError("workset facade failed; physical ownership retained") from self._facade_error
        if self.runtime is None:
            raise RuntimeError("attach P runtime before ingress")
        if self.native_bridge is None:
            raise RuntimeError("attach native return bridge before ingress")
        self.runtime.check_health()

    def request(self, snapshot_id, parent_tokens, prompt_tokens, *, owner="legacy"):
        self.check_health()
        with self._lock:
            if (snapshot_id, owner) in self._pending_closed_owners:
                return False
            if not super().request(snapshot_id, parent_tokens, prompt_tokens, owner=owner):
                return False
            identity = snapshot_id, owner
            if self.rank or identity in self._requests or snapshot_id in self._leases:
                return True
            self._request_epoch += 1
            checkpoint = int(bool(parent_tokens)) if self._state_allocators else 0
            runtime = self._runtime_state_slots if self._state_allocators else 0
            if self._state_allocators:
                runtime += 2 - int(self._reserve_mamba_checkpoint)
            intent = WorksetIntent(snapshot_id, f"{owner}:allocator:{self._request_epoch}",
                owner, int(parent_tokens), int(prompt_tokens), checkpoint, runtime)
            record = dict(intent=intent, future=None, plan=None, cancelled=False)
            self._requests[identity] = record
            record["future"] = self.runtime.request(intent)
            record["future"].add_done_callback(lambda result: self._requested(record, result))
            return True

    def bind_restore_epoch(self, snapshot_id, owner, epoch):
        """Accept an authoritative Host recovery claim's epoch, not a timer.

        Caller must have successfully claimed the existing Host ledger and
        validated its owner/claim/read_epoch. A strictly newer epoch is possible
        only after that ledger's all-rank retry barrier; it may rearm a cancelled
        intent that never acquired addresses. It cannot revoke a live grant,
        unresolved authority operation, permanent Direct closure or DMA fence.
        """
        if (type(epoch) is not int or epoch < 1
                or not owner.startswith(f"slow:{snapshot_id}:")):
            raise ValueError("Host retry requires exact Slow owner and positive read epoch")
        self.check_health()
        identity = snapshot_id, owner
        with self._lock:
            previous = self._restore_epochs.get(identity)
            if previous is not None and epoch < previous:
                return False
            if previous == epoch:
                return identity not in self._pending_closed_owners
            if previous is not None:
                if (snapshot_id in self._leases or snapshot_id in self._intents
                        or identity in self._requests
                        or any(plan.key.snapshot_id == snapshot_id and plan.owner == owner
                               for plan in self._known_plans.values())
                        or snapshot_id in self._tp_cancel_pending
                        or snapshot_id in self._tp_release_pending):
                    return False
                self._pending_closed_owners.discard(identity)
                self._tp_retire_requested.discard(snapshot_id)
            self._restore_epochs[identity] = epoch
            return True

    def _requested(self, record, future):
        with self._lock:
            try:
                plan = future.result()
                record["plan"] = plan
                self._known_plans[plan.key] = plan
                if record["cancelled"]:
                    self._close_private(plan)
            except PendingWorksetCancelled:
                if not record["cancelled"]:
                    self._facade_error = RuntimeError("unexpected intent cancellation")
                else:
                    intent = record["intent"]
                    self._requests.pop((intent.snapshot_id, intent.owner), None)
                    self._tp_retire_requested.discard(intent.snapshot_id)
            except Exception as error:
                self._facade_error = error

    def install_prepared(self, prepared):
        with self._lock:
            self._known_plans[prepared.plan.key] = prepared.plan
            owner = (prepared.plan.key.snapshot_id, prepared.plan.owner)
            superseded = owner in self._superseded_owners
            # The authority may already have granted before native cancellation
            # was observed locally. Install only the quarantined ownership
            # descriptor, never revive its right to start I/O or bind.
            if superseded:
                self._superseded_owners.remove(owner)
            try:
                lease = super().install_prepared(prepared)
            finally:
                if superseded:
                    self._superseded_owners.add(owner)
            if (lease.snapshot_id, lease.owner) in self._pending_closed_owners:
                self._close_private(prepared.plan)
            if superseded:
                self._close_private(prepared.plan)
            return lease

    def get(self, snapshot_id, *, owner=None):
        self.check_health()
        with self._lock:
            lease = super().get(snapshot_id, owner=owner)
            if (lease is not None and self.rank == 0 and lease.state == "active"
                    and lease.controller_plan.key not in self.runtime.ready_cut().ready
                    and lease.controller_plan.key not in self._closed_keys):
                return None
            return lease

    def service(self, allocator=None, **kwargs):
        """Compatibility safe-point: health only, never choose/free addresses."""
        self.check_health()

    def install_tp_plan(self, *args, **kwargs):
        raise RuntimeError("controller broker cannot install a native allocation plan")

    def prepare_tp_control(self, *args, **kwargs):
        raise RuntimeError("broadcast runtime ready cuts, not native allocator plans")

    def _close_private(self, plan):
        if plan.key in self._closed_keys:
            return
        self._closed_keys.add(plan.key)
        self._tp_retire_requested.add(plan.key.snapshot_id)
        # A native retirement notification is not authority to cancel pages
        # already donated to Req/Radix. Their exact subset/last-ref returns
        # eventually empty the provenance record; only then runtime closes it.
        if self.rank == 0 and plan.key not in self._external_keys:
            future = self.runtime.cancel(plan.key)
            future.add_done_callback(self._record_error)
        self._progress_observers()

    def _record_error(self, future):
        try:
            future.result()
        except Exception as error:
            self._facade_error = error

    def cancel_unstarted(self, snapshot_id, *, owner=None):
        self.check_health()
        with self._lock:
            lease = self._leases.get(snapshot_id)
            if lease is not None:
                if (owner is not None and lease.owner != owner) or lease.state != "active":
                    return False
                self._close_private(lease.controller_plan)
                return True
            pending = self._intents.get(snapshot_id)
            if pending is None or (owner is not None and pending[0] != owner):
                return False
            self._intents.pop(snapshot_id, None)
            self._intent_requested_at.pop(snapshot_id, None)
            self._tp_retire_requested.add(snapshot_id)
            record = self._requests.get((snapshot_id, pending[0]))
            if record is not None:
                record["cancelled"] = True
                if record["plan"] is not None:
                    self._close_private(record["plan"])
                else:
                    self.runtime.cancel_pending(record["intent"]).add_done_callback(self._record_error)
            elif self.rank:
                # A follower does not own an ungranted authority intent. Close
                # this exact logical owner, not every future owner of the SID.
                self._pending_closed_owners.add((snapshot_id, pending[0]))
                self._tp_retire_requested.discard(snapshot_id)
            return True

    def _prepare_tp_retire_locked(self, snapshot_id):
        if snapshot_id not in self._leases and snapshot_id not in self._intents:
            # An old native abort can follow the already-applied FREE log.
            # It must not recreate a snapshot tombstone for a successor.
            return True
        ready = super()._prepare_tp_retire_locked(snapshot_id)
        lease = self._leases.get(snapshot_id)
        if lease is not None and lease.state not in {"binding", "handed", "consumed"}:
            self._close_private(lease.controller_plan)
        elif lease is None:
            self.cancel_unstarted(snapshot_id)
        return ready

    def commit_tp_retire(self, snapshot_id):
        # Native retirement remains a logical reference boundary only. The
        # ordered allocator log alone can reclaim or reuse physical addresses.
        with self._lock:
            lease = self._leases.get(snapshot_id)
            if lease is None:
                return snapshot_id not in self._intents
            if lease.io_attempt is not None or lease.state in {"binding", "handed", "consumed"}:
                return False
            self._close_private(lease.controller_plan)
            return True

    def request_release(self, snapshot_id, lease=None, *, owner=None, io_attempt=None):
        self.check_health()
        with self._lock:
            current = self._leases.get(snapshot_id)
            if (lease is None or current is not lease or
                    (owner is not None and current.owner != owner)):
                return False
            if current.state in {"binding", "handed", "consumed"}:
                return False
            if current.state in {"io_reserved", "io_inflight", "release_pending"}:
                if current.io_attempt != io_attempt:
                    return False
                # Preserve reserved vs posted: cancellation of a claim Future
                # still has to drain before cancel_io_attempt is legal.
                if current.state == "io_inflight":
                    current.state = "release_pending"
                self._close_private(current.controller_plan)
                return False
            self._close_private(current.controller_plan)
            return True

    def cancel_io_attempt(self, *args, **kwargs):
        with self._lock:
            result = super().cancel_io_attempt(*args, **kwargs)
            self._progress_observers()
            return result

    def mark_io_quiesced(self, *args, **kwargs):
        with self._lock:
            result = super().mark_io_quiesced(*args, **kwargs)
            self._progress_observers()
            return result

    def _producer_done(self, event=None):
        current = threading.current_thread()
        if current is self._native_cleanup_owner:
            if not self._native_cleanup_lock._is_owned():
                raise RuntimeError("native cleanup requires the scheduler state lock")
        elif current is not self._native_owner:
            raise RuntimeError("native ownership return must run on scheduler thread")
        if event is None and self.device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
        return self.native_bridge.observe_event(event)

    def allow_native_cleanup_owner(self, owner, state_lock):
        if threading.current_thread() is not owner:
            raise RuntimeError("native cleanup owner must register itself")
        if self._native_cleanup_owner not in (None, owner):
            raise RuntimeError("native cleanup owner changed")
        self._native_cleanup_owner = owner
        self._native_cleanup_lock = state_lock

    def _return_scope(self, plan, return_id, pages=(), slots=(), event=None):
        pages, slots = _normalize(pages), _normalize(slots)
        if not pages and not slots:
            return completed(None)
        identity = plan.key, return_id
        with self._lock:
            if self._known_plans.get(plan.key) != plan:
                raise RuntimeError("private return does not name a live exact workset")
            old = self._permits.get(identity)
            if old is not None:
                if old[:2] != (pages, slots):
                    raise RuntimeError("private return identity changed")
                return old[3]
            physical = self._producer_done(event)
            accepted = (self.runtime.begin_return(plan.key, return_id, pages=pages, slots=slots)
                        if self.rank == 0 else completed(None))
            self._permits[identity] = (pages, slots, physical, accepted)
            physical.add_done_callback(lambda _: self._progress_observers())
            accepted.add_done_callback(self._record_error)
            self._progress_observers()
            return accepted

    def return_fresh_prefix(self, plan, prefix_tokens):
        if plan.parent_tokens:
            raise ValueError("only a fresh private prompt can adopt a cached prefix")
        if type(prefix_tokens) is not int or prefix_tokens < 0 or prefix_tokens % self.page_size:
            raise ValueError("private cached prefix must be page aligned")
        pages = slice_runs(plan.suffix_pages, 0, prefix_tokens // self.page_size)
        return self._return_scope(plan, "fresh-cached-prefix", pages=pages)

    def return_state_slots(self, plan, slot_ids, event=None):
        slot_ids = tuple(slot_ids)
        runtime = tuple(i for run in plan.runtime_slots for i in range(run.start, run.end))
        if (not runtime or any(type(index) is not int or index not in runtime[-2:]
                               for index in slot_ids)
                or len(set(slot_ids)) != len(slot_ids)):
            raise ValueError("checkpoint return must name only this lease's output slots")
        slots = _normalize(tuple(PageRun(index, 1) for index in slot_ids))
        return self._return_scope(plan, "checkpoint:" + ",".join(str(r.start) for r in slots),
                                  slots=slots, event=event)

    def prepare_req_checkpoints(self, req, lease):
        from sglang.srt.disaggregation.agentic_workset_native import install_checkpoint_rotation
        rotation = getattr(req, "_agentic_checkpoint_rotation", None)
        if rotation is not None:
            if rotation.key != lease.controller_plan.key:
                raise RuntimeError("Req checkpoint rotation belongs to another workset")
            return
        if not self._state_allocators:
            return
        plan = lease.controller_plan
        runtime = tuple(i for run in plan.runtime_slots for i in range(run.start, run.end))
        install_checkpoint_rotation(req, plan.key, runtime[-2:],
            req._agentic_mamba_prefill_checkpoint,
            lambda receipt: self.return_state_slots(plan, (receipt.slot,), receipt.event))

    def handoff_to_req(self, snapshot_id, req, lease):
        super().handoff_to_req(snapshot_id, req, lease)
        with self._lock:
            self._external_keys.add(lease.controller_plan.key)

    def handoff_fresh_to_req(self, snapshot_id, req, lease):
        super().handoff_fresh_to_req(snapshot_id, req, lease)
        with self._lock:
            self._external_keys.add(lease.controller_plan.key)

    def release_handed(self, snapshot_id, lease, *, req):
        with self._lock:
            if (self._leases.get(snapshot_id) is not lease or lease.state != "handed"
                    or getattr(req, "_agentic_p_workset_lease", None) is not lease):
                return False
            plan = lease.controller_plan
            if lease.suffix_cursor % self.page_size:
                raise RuntimeError("cannot return a private suffix sharing a live partial page")
            start = lease.suffix_cursor // self.page_size
            count = sum(r.count for r in plan.suffix_pages) - start
            pages = slice_runs(plan.suffix_pages, start, count)
            self._return_scope(plan, "unconsumed-suffix", pages=pages)
            lease.state = "releasing"
            self._tp_retire_requested.add(snapshot_id)
            return True

    def abort_bind(self, snapshot_id, lease, *, parent_bound):
        with self._lock:
            if self._leases.get(snapshot_id) is not lease or lease.state != "binding":
                return False
            plan = lease.controller_plan
            # Existing rollback clears any uncommitted Req aliases. No native
            # allocator is called by this state transition.
            if not super().abort_bind(snapshot_id, lease, parent_bound=parent_bound):
                return False
            if parent_bound:
                self._external_keys.add(plan.key)
                self._return_scope(plan, "aborted-bind-private", pages=plan.suffix_pages,
                    slots=plan.runtime_slots + (plan.checkpoint_slots if lease.state_device_indices else ()))
            else:
                physical = self._producer_done()
                self._permits[(plan.key, "whole-private")] = ((), (), physical, completed())
                physical.add_done_callback(lambda _: self._progress_observers())
                self._close_private(plan)
            return True

    def reference_fence(self, plan, scope):
        """Called by the runtime actor; no GPU query or scheduler allocation."""
        future = Future()
        with self._lock:
            # Capture this actor-owned view at close installation; background
            # event callbacks must never read the executor's mutable maps.
            view = self.runtime.executor.local_view(plan.key)
            empty = view is not None and not view.remaining_pages and not view.remaining_slots
            self._observers[(plan.key, scope.sequence)] = (plan, scope, future, empty)
            self._progress_observers()
        return future

    def _progress_observers(self):
        with self._lock:
            for identity, (plan, scope, future, empty) in tuple(self._observers.items()):
                if future.done():
                    continue
                if isinstance(scope, ReturnPlan):
                    permit = self._permits.get((plan.key, scope.return_id))
                    if permit is None or permit[:2] != (scope.pages, scope.slots):
                        continue
                    physical = permit[2]
                else:
                    lease = self._leases.get(plan.key.snapshot_id)
                    if not empty:
                        if plan.key in self._external_keys:
                            continue
                        if lease is not None and (lease.controller_plan != plan or lease.io_attempt is not None
                                or lease.parent_bound or lease.runtime_state_req is not None
                                or lease.state in {"binding", "handed", "consumed"}):
                            continue
                    permit = self._permits.get((plan.key, "whole-private"))
                    physical = permit[2] if permit is not None else completed()
                if not physical.done():
                    continue
                try:
                    if physical.result() is not True:
                        raise RuntimeError("producer event supplied no physical completion")
                    future.set_result(FenceProof(plan.key, scope.sequence, True, True))
                    self._observers.pop(identity, None)
                except Exception as error:
                    future.set_exception(error)

    def controller_retired(self, plan):
        with self._lock:
            sid = plan.key.snapshot_id
            current = self._leases.get(sid)
            if current is None or current.controller_plan == plan:
                self._tp_cancel_pending.pop(sid, None)
                self._tp_release_pending.pop(sid, None)
                self._tp_retire_requested.discard(sid)
                pending = self._intents.get(sid)
                if pending is not None and pending[0] == plan.owner:
                    self._intents.pop(sid, None)
                    self._intent_requested_at.pop(sid, None)
            self._known_plans.pop(plan.key, None)
            self._pending_closed_owners.discard((plan.key.snapshot_id, plan.owner))
            self._closed_keys.discard(plan.key)
            self._external_keys.discard(plan.key)
            record = self._requests.get((plan.key.snapshot_id, plan.owner))
            if record is not None and record["plan"] == plan:
                del self._requests[(plan.key.snapshot_id, plan.owner)]
            for identity in tuple(self._permits):
                if identity[0] == plan.key:
                    del self._permits[identity]
            for identity in tuple(self._observers):
                if identity[0] == plan.key:
                    del self._observers[identity]
