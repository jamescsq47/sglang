"""Background admission for complete Host-backed Prefill worksets.

The existing Host worker calls ``progress``; no extra thread, allocator, CUDA
stream or file protocol is introduced. Ingress supplies immutable CPU views.
Only rank zero selects restores. PREPARE travels over the existing TP event
connection; the existing Host worker starts DMA after its all-shard preparation
barrier. The native scheduler sees only bind/commit/admit/terminal commands.

``native_cleared`` must be called by every rank after executing native CLEAR.
That execution acknowledgment, not a timeout, retires the background attempt.
"""
from collections import OrderedDict
from dataclasses import dataclass
import logging
import os
import threading
from types import SimpleNamespace

from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration
from sglang.srt.disaggregation.agentic_tp_events import EventKey

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RestoreIngress:
    parent: RequestGeneration
    rid: str
    tokens: tuple

    def view(self):
        return SimpleNamespace(rid=self.rid, origin_input_ids=self.tokens)


class AgenticHostRestoreController:
    """One ingress mirror per rank and a single leader admission queue.

    ``observe(req, parent)`` is scheduler-side O(prompt-copy), exactly once per
    HTTP ingress, not per Forward. ``progress`` is single Host-worker-owned.
    ``native_commands`` copies a bounded already-selected control cut; it never
    probes capacity or performs a network call. ``cancel`` only posts identity.
    """

    def __init__(self, manager, mailbox, *, depth=None):
        self.manager, self.mailbox = manager, mailbox
        self.client = mailbox.client
        self.rank, self.size = int(mailbox.tp_rank), int(mailbox.tp_size)
        if not getattr(manager.workset_broker, "controller_mode", False):
            raise ValueError("background Host admission requires the workset authority")
        if not getattr(manager.ledger, "is_event_control", False):
            raise ValueError("background Host admission requires socket lifecycle events")
        lanes = max(1, int(manager.max_h2d_inflight))
        if depth is None:
            depth = int(os.getenv("SGLANG_AGENTIC_KV_TP_HOST_PIPELINE_DEPTH", str(lanes)))
        self.depth = min(lanes, max(1, int(depth)))
        self.namespace = mailbox.namespace + ":background-prepare"
        self._lock = threading.RLock()
        self._ingress = OrderedDict()
        self._jobs = OrderedDict()
        self._cancelled = {}
        self._cleared = set()
        self._native = ()
        self._serial = 0
        self._error = None

    def observe(self, req, parent):
        """Called only after tool output/full next prompt has reached this rank."""
        sid, rid = parent.snapshot_id, str(req.rid)
        with self._lock:
            old = self._ingress.get(sid)
            if old is not None:
                # A retry cannot overwrite the old owner's immutable input.
                return old.rid == rid
            self._ingress[sid] = RestoreIngress(parent, rid, tuple(req.origin_input_ids))
        self.manager._control_wakeup.set()
        return True

    def cancel(self, snapshot_id, rid):
        with self._lock:
            item = self._ingress.get(snapshot_id)
            if item is None or item.rid != str(rid):
                return
            self._cancelled[snapshot_id] = str(rid)
            if snapshot_id not in self._jobs:
                self._ingress.pop(snapshot_id, None)
        self.manager._control_wakeup.set()

    def native_cleared(self, snapshot_id, attempt):
        with self._lock:
            job = self._jobs.get(snapshot_id)
            if job is not None and job["key"].attempt_id == attempt:
                self._cleared.add(job["key"])
        self.manager._control_wakeup.set()

    def native_commands(self):
        with self._lock:
            if self._error is not None:
                raise RuntimeError("background Host admission failed closed") from self._error
            return [dict(command, **({"terminal": dict(command["terminal"])}
                    if "terminal" in command else {})) for command in self._native]

    def owns(self, snapshot_id):
        with self._lock:
            return snapshot_id in self._jobs

    def has_pending(self):
        with self._lock:
            return bool(self._jobs or self._ingress or self._cancelled)

    def forget_waiter(self, snapshot_id, rid):
        """Remove a Direct/terminal waiter that never entered this pipeline."""
        with self._lock:
            item = self._ingress.get(snapshot_id)
            if snapshot_id not in self._jobs and item is not None and item.rid == str(rid):
                self._ingress.pop(snapshot_id)
                self._cancelled.pop(snapshot_id, None)

    def take_native_terminal(self, snapshot_id, rid):
        """Atomically yield unselected ingress to the existing terminal owner.

        Rank zero calls before choosing a legacy terminal command; followers
        call when installing it. A selected background attempt cannot be
        relabeled: its own terminal command must close the exact old identity.
        """
        with self._lock:
            if snapshot_id in self._jobs:
                return False
            item = self._ingress.get(snapshot_id)
            if item is not None and item.rid != str(rid):
                return False
            self._ingress.pop(snapshot_id, None)
            self._cancelled.pop(snapshot_id, None)
            return True

    def _new_job(self, key, rid, parent):
        return {"key": key, "rid": rid, "parent": parent, "prepared": False,
                "prepare_seen": False, "retire_seen": False, "retire_sent": False}

    def _consume_commands(self):
        for key, command_id, payload in self.client.drain_commands(self.namespace):
            sid = key.snapshot_id
            with self._lock:
                job = self._jobs.get(sid)
                if payload["op"] == "PREPARE":
                    if job is None:
                        parent = RequestGeneration(payload["request_id"], int(payload["generation"]))
                        job = self._new_job(key, str(payload["rid"]), parent)
                        self._jobs[sid] = job
                    if job["key"] != key or job["rid"] != str(payload["rid"]):
                        raise RuntimeError("Host attempt changed before native CLEAR")
                    if payload.get("cancelled"):
                        self._cancelled[sid] = job["rid"]
                    job["prepare_seen"] = True
                elif payload["op"] == "RETIRE" and command_id == 2:
                    if job is None or job["key"] != key:
                        raise RuntimeError("Host RETIRE has no matching live attempt")
                    job["retire_seen"] = True
                else:
                    raise RuntimeError("invalid Host admission command")

    def _forget(self, sid, job):
        with self._lock:
            if self._jobs.get(sid) is job:
                self._jobs.pop(sid)
                item = self._ingress.get(sid)
                if item is not None and item.rid == job["rid"]:
                    self._ingress.pop(sid)
                self._cancelled.pop(sid, None)
                self._cleared.discard(job["key"])

    def _prepare_local(self):
        with self._lock:
            jobs = tuple(self._jobs.items())
        for sid, job in jobs:
            key = job["key"]
            with self._lock:
                item = self._ingress.get(sid)
                cancelled = self._cancelled.get(sid) == job["rid"]
                cleared = key in self._cleared
            if job["retire_seen"] and cleared:
                if not job.get("retire_acked"):
                    self.client.ack_command(self.namespace, key, 2)
                    job["retire_acked"] = True
                if self.rank != 0:
                    self._forget(sid, job)
                continue
            if cleared or job["retire_seen"] or job["retire_sent"]:
                continue
            if not job["prepare_seen"]:
                continue
            if not cancelled and (item is None or item.rid != job["rid"]):
                continue  # Native ingress may arrive later on a follower.
            self.manager.register_tp_host_progress(sid, key, self.mailbox.publish_local)
            if not job["prepared"]:
                # ACK only installing this exact metadata intent, NOT pages
                # or DMA. Actual preparation is reported by the Host worker.
                self.client.ack_command(self.namespace, key, 1)
                job["prepared"] = True
            if cancelled:
                if not job.get("cancel_submitted"):
                    self.manager.abort_request(job["rid"], job["parent"])
                    job["cancel_submitted"] = True
                if (not job.get("cancel_acked") and self.manager.tp_host_control_quiescent(sid, job["rid"])):
                    self.client.report_state(self.namespace + ":cancel", key, 1)
                    job["cancel_acked"] = True
            elif self.manager.terminal_restore_reason(item.parent) is None:
                status = self.mailbox.local_status(key)
                if status is not None and (status < 0 or status >= 2):
                    continue
                # This reserves a lane and freezes a worker-private view. The
                # existing preparation worker alone claims Host and workset.
                # Capacity/claim rejection may remove a previous prepare, so
                # retry here rather than require another scheduler selection.
                self.manager._queue_host_prepare(item.view(), item.parent)

    def _cancel_unselected(self):
        """Leader closes cancelled metadata even if PREPARE never happened.

        Followers can cancel before a queued PREPARE reaches them. Retain only
        the rid until this explicit cancellation command is acknowledged; no
        prompt copies or forever-unmatched cancellation tombstones remain.
        """
        with self._lock:
            pending = tuple(self._cancelled.items())
        for sid, rid in pending:
            with self._lock:
                if sid in self._jobs or self._cancelled.get(sid) != rid:
                    continue
                self._serial += 1
                key = EventKey(sid, "host-background:" + str(self._serial))
                request_id, generation = sid.rsplit(":", 1)
                parent = RequestGeneration(request_id, int(generation))
                self._jobs[sid] = self._new_job(key, rid, parent)
            self.client.publish_command(self.namespace, key, {
                "op": "PREPARE", "rid": rid, "request_id": parent.request_id,
                "generation": parent.generation, "cancelled": True,
            }, command_id=1)

    def _select(self):
        with self._lock:
            jobs, ingress = tuple(self._jobs.items()), tuple(self._ingress.items())
        # _jobs owns the complete control lifetime, not a transport credit.
        # All-rank phase 2 proves the copies have quiesced. BIND/ADMIT/CLEAR
        # remain in the independent native completion queue with their exact
        # worksets, but must not prevent the next Host->P copy from starting.
        # CLEAR hides the phase mailbox: do not turn that missing old report
        # back into an outstanding copy while RETIRE ACKs are arriving.
        copying = sum(
            (self.client.group_status(self.namespace + ":cancel", job["key"]) != 1
             if sid in self._cancelled
             else (self.mailbox.group_status(job["key"]) or 0) < 2)
            for sid, job in jobs
            if not job["retire_sent"] and not job["retire_seen"]
        )
        for sid, item in ingress:
            if copying >= self.depth:
                break
            with self._lock:
                if sid in self._jobs or sid in self._cancelled:
                    continue
            if self.manager.terminal_restore_reason(item.parent) is not None:
                continue  # Existing native terminal protocol owns unselected eviction.
            if not self.manager.snapshot_ready(item.parent):
                continue
            with self._lock:
                # Cancellation can race the cached readiness check above.
                if sid in self._cancelled or self._ingress.get(sid) is not item:
                    continue
                self._serial += 1
                key = EventKey(sid, "host-background:" + str(self._serial))
                job = self._new_job(key, item.rid, item.parent)
                self._jobs[sid] = job
            self.client.publish_command(self.namespace, key, {
                "op": "PREPARE", "rid": item.rid,
                "request_id": item.parent.request_id, "generation": item.parent.generation,
            }, command_id=1)
            copying += 1
            logger.info("AgenticKV tp_host_selected snapshot=%s copying=%d/%d background=true",
                        sid, copying, self.depth)

    def _build_native(self):
        commands = []
        with self._lock:
            jobs = tuple(self._jobs.items())
        for sid, job in jobs:
            key, parent = job["key"], job["parent"]
            if job.get("retire_acked") and self.client.command_complete(self.namespace, key):
                self.client.clear(self.namespace, key)
                if job.get("cancel_acked"):
                    self.client.clear(self.namespace + ":cancel", key)
                self._forget(sid, job)
                continue
            status = self.mailbox.group_status(key)
            status = 0 if status is None else int(status)
            terminal = self.manager.terminal_restore_reason(parent)
            with self._lock:
                cancelled = self._cancelled.get(sid) == job["rid"]
            if job["retire_sent"]:
                # Native CLEAR may already have hidden the phase mailbox on
                # one rank; retain CLEAR until all execution ACKs arrive.
                action = "clear"
            elif cancelled:
                action = ("clear" if self.client.group_status(self.namespace + ":cancel", key) == 1
                          else "abort")
            elif terminal is not None:
                action = "clear" if status >= 7 else "terminal_admit" if status == 6 else "terminal_prepare"
            elif status < 0:
                action = "abort"
            elif status < 2:
                continue  # PREPARE and START are never scheduler commands.
            elif status == 2:
                action = "bind"
            elif status == 3:
                if not self.manager._complete_shared_host_manifest(parent):
                    continue
                action = "commit"
            elif status == 4:
                action = "admit"
            else:
                action = "clear"
            if action == "clear" and not job["retire_sent"]:
                if not self.client.command_complete(self.namespace, key):
                    continue
                self.client.publish_command(self.namespace, key, {"op": "RETIRE"}, command_id=2)
                job["retire_sent"] = True
            commands.append({"snapshot": sid, "request_id": parent.request_id,
                             "generation": parent.generation, "action": action,
                             "control_attempt": key.attempt_id,
                             **({"terminal": {"rid": job["rid"], "reason": terminal}}
                                if terminal is not None and not cancelled else {})})
        with self._lock:
            self._native = tuple(commands)

    def progress(self):
        """Call from the existing sole Host control worker, never Forward."""
        try:
            if self.rank == 0:
                self._cancel_unselected()
            self._consume_commands()
            self._prepare_local()
            if self.rank == 0:
                self._build_native()
                self._select()
        except Exception as error:
            with self._lock:
                self._error = error
            raise
