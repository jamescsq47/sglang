"""Native two-phase admission for fresh/recompute complete worksets.

The asynchronous allocator's ready cut is observation, not Forward permission.
One existing native header prepares exact Req references on every rank; a later
header commits only after all preparation receipts. No allocator, CUDA wait,
network wait or model execution is performed here. Parent restores retain their
existing protocol. Cancellation is an ordered native abort, never a local veto
of an already-frozen commit header or permission to recycle physical pages.
"""
from dataclasses import dataclass
import json

from sglang.srt.disaggregation.agentic_tp_events import EventKey
from sglang.srt.disaggregation.agentic_workset_ledger import LeaseKey


class DuplicateGenerationRequest(ValueError):
    """Ingress conflict, not permission to replace a live generation owner."""

    def __init__(self, snapshot_id, existing_rid, incoming_rid):
        self.snapshot_id = snapshot_id
        self.existing_rid = existing_rid
        self.incoming_rid = incoming_rid
        super().__init__(
            "one live request per request-generation is required: "
            f"snapshot={snapshot_id} existing_rid={existing_rid} incoming_rid={incoming_rid}"
        )


@dataclass
class _Request:
    req: object
    snapshot_id: str
    workset_id: str
    generation_key: str
    owner: str | None
    prompt_tokens: int
    key: LeaseKey | None = None
    nonce: int | None = None
    prepared: bool = False
    committed: bool = False
    cancelled: bool = False
    failed: bool = False
    aborted: bool = False


class FreshWorksetAdmission:
    """Construct/consume on the native scheduler thread.

    ``mailbox`` is the existing socket mailbox connection in a distinct
    namespace, not the runtime actor's exclusive event client. ``on_abort``
    runs only at the native header boundary; it must perform/advance ordinary
    exact Req cleanup and return True once local logical references are gone.
    Actual physical fences and all-rank release stay with the broker/runtime.
    A False result retains the record and retries on a later native header.
    The caller removes completed/aborted Req records with forget(), preventing
    this admission table from becoming another full-run history store.
    """

    def __init__(self, broker, mailbox, *, rank, tp_size, on_abort, max_pending=4096):
        if type(rank) is not int or type(tp_size) is not int or not 0 <= rank < tp_size:
            raise ValueError("invalid admission TP group")
        if not callable(on_abort) or type(max_pending) is not int or max_pending < 1:
            raise ValueError("bounded admission and native cleanup callback required")
        self.broker, self.mailbox = broker, mailbox
        self.rank, self.tp_size, self.on_abort = rank, tp_size, on_abort
        self._max_pending = max_pending
        self._requests, self._snapshots = {}, {}
        self._committed_order = []
        self._sequence = self._last_applied = self._nonce = 0
        self._last_control = None

    @staticmethod
    def generation_namespace(snapshot_id):
        try:
            request_id, generation = snapshot_id.rsplit(":", 1)
            if not request_id or str(int(generation)) != generation or int(generation) < 0:
                raise ValueError()
        except (ValueError, AttributeError):
            raise ValueError("fresh worksets require request-id:generation snapshot identity") from None
        return f"agentic-v1:{request_id}:g{generation}"

    def observe(self, req, snapshot_id, generation_key):
        """Common native ingress: retain identity, never request addresses.

        A follower's bounded metadata scan need not run before a leader's
        PREPARE or ABORT. Retain this exact Req through that ordered decision,
        including HTTP cancellation which removes it from ordinary queues.
        Parent restores remain unselected and use their existing protocol.
        """
        if generation_key != self.generation_namespace(snapshot_id) or req.extra_key != generation_key:
            raise ValueError("fresh/recompute must use the exact request-generation cache namespace")
        current = self._requests.get(req.rid)
        if current is not None:
            if (current.req is not req or current.snapshot_id != snapshot_id
                    or current.prompt_tokens != len(req.origin_input_ids)):
                raise ValueError("request admission identity changed")
            return
        if snapshot_id in self._snapshots:
            raise DuplicateGenerationRequest(snapshot_id, self._snapshots[snapshot_id], req.rid)
        if len(self._requests) >= self._max_pending:
            raise RuntimeError("fresh admission table full; request not accepted")
        # A fresh generation's private prompt is not its future parent snapshot
        # transport owner. Keep ledger identities disjoint while old Req/Radix
        # references are still retiring when the next turn restores that parent.
        workset_id = f"fresh:{snapshot_id}:{req.rid}"
        record = _Request(req, snapshot_id, workset_id, generation_key, None, len(req.origin_input_ids))
        self._requests[req.rid] = record
        self._snapshots[snapshot_id] = req.rid
        # Observation is local metadata only. Cancellation is sparse: one
        # negative report requests an ordered ABORT; cleanup completion still
        # needs every rank's positive report. Do not broadcast neutral facts
        # for every waiting request (TP8/c128 otherwise emits 2048 updates).

    def register(self, req, snapshot_id, generation_key, *, owner="fresh"):
        """Rank-zero eligibility selection, or exact native header replay."""
        if owner not in {"fresh", "recompute"}:
            raise ValueError("parent restores use their existing admission protocol")
        self.observe(req, snapshot_id, generation_key)
        record = self._requests[req.rid]
        if record.owner is not None:
            if record.owner != owner:
                raise ValueError("request admission identity changed")
            return
        if record.cancelled:
            return  # Sticky local cancellation cannot revive an intent.
        if not self.broker.request(record.workset_id, 0, record.prompt_tokens, owner=owner):
            raise RuntimeError("complete fresh workset intent was rejected")
        record.owner = owner

    @staticmethod
    def _cancel_key(record):
        return EventKey(record.snapshot_id, "fresh-cancel:" + record.req.rid)

    @staticmethod
    def _abort_key(record):
        return EventKey(record.snapshot_id, "fresh-abort:" + record.req.rid)

    @staticmethod
    def _prepare_key(record):
        key = record.key
        identity = [key.incarnation, key.attempt_id, key.version, record.req.rid, record.nonce]
        return EventKey(record.snapshot_id, "fresh-prepare:" + json.dumps(identity, separators=(",", ":")))

    def workset_id(self, req):
        record = self._requests.get(req.rid)
        return record.workset_id if record is not None and record.req is req else None

    def req_by_rid(self):
        return {rid: record.req for rid, record in self._requests.items()}

    def contains(self, req):
        record = self._requests.get(req.rid)
        return record is not None and record.req is req

    def is_committed(self, req):
        return self.contains(req) and self._requests[req.rid].committed

    def pending(self, req):
        return self.selected(req) and not self._requests[req.rid].committed

    def selected(self, req):
        return self.contains(req) and self._requests[req.rid].owner is not None

    def committed_rids(self):
        return tuple(self._committed_order)

    def cancel(self, req):
        record = self._requests.get(req.rid)
        if record is None or record.req is not req:
            return False
        record.cancelled = True
        self.mailbox.publish_local(self._cancel_key(record), -1)
        return True

    def defer_fresh(self, req):
        record = self._requests.get(req.rid)
        if record is None or record.req is not req:
            return True
        # A local cancellation arriving after a frozen COMMIT cannot veto only
        # one TP rank. The next common ABORT is the sole eligibility boundary.
        return not record.committed or record.aborted

    @staticmethod
    def _key_payload(key):
        return None if key is None else [key.incarnation, key.snapshot_id, key.attempt_id, key.version]

    def _command(self, action, record):
        return dict(action=action, rid=record.req.rid, snapshot_id=record.snapshot_id,
                    workset_id=record.workset_id,
                    generation_key=record.generation_key, owner=record.owner,
                    prompt_tokens=record.prompt_tokens, key=self._key_payload(record.key),
                    nonce=record.nonce)

    def build_control(self, waiting_reqs):
        if self.rank:
            raise ValueError("only rank zero selects native fresh admission")
        waiting = {req.rid: req for req in waiting_reqs}
        cut = self.broker.runtime.ready_cut()
        ready = set(cut.ready)
        commands = []
        with self.broker._lock:
            for record in self._requests.values():
                if record.aborted and self.mailbox.group_status(self._abort_key(record)) == 1:
                    commands.append(self._command("forget", record))
                    continue
                failed = record.key is not None and self.mailbox.any_negative_report(self._prepare_key(record))
                cancelled = record.cancelled or self.mailbox.any_negative_report(self._cancel_key(record))
                if cancelled or failed:
                    commands.append(self._command("abort", record))
                    continue
                if record.owner is None:
                    continue  # Metadata observation is not allocation eligibility.
                if record.committed or waiting.get(record.req.rid) is not record.req:
                    continue
                if record.key is not None:
                    if self.mailbox.group_status(self._prepare_key(record)) == 1:
                        commands.append(self._command("commit", record))
                    continue
                lease = self.broker.get(record.workset_id)
                if (lease is None or lease.controller_plan is None or lease.controller_plan.key not in ready
                        or lease.state != "active" or lease.parent_tokens
                        or record.workset_id in self.broker._tp_retire_requested):
                    continue
                self._nonce += 1
                record.key, record.nonce = lease.controller_plan.key, self._nonce
                commands.append(self._command("prepare", record))
        self._sequence += 1
        return dict(sequence=self._sequence, commands=commands)

    def apply_control(self, control, req_by_rid):
        sequence = control["sequence"]
        if sequence == self._last_applied:
            if control != self._last_control:
                raise RuntimeError("native admission header changed on replay")
            return
        if type(sequence) is not int or sequence != self._last_applied + 1:
            raise RuntimeError("native admission header is out of order")
        for command in control["commands"]:
            record = self._requests.get(command["rid"])
            if (record is None or req_by_rid.get(command["rid"]) is not record.req
                    or command["snapshot_id"] != record.snapshot_id
                    or command["workset_id"] != record.workset_id
                    or command["generation_key"] != record.generation_key
                    or command["prompt_tokens"] != record.prompt_tokens
                    or record.req.extra_key != record.generation_key
                    or len(record.req.origin_input_ids) != record.prompt_tokens):
                raise RuntimeError("native admission command does not name the registered Req")
            action = command["action"]
            if record.owner is None and command["owner"] is not None:
                if action not in {"prepare", "abort"} or command["owner"] not in {"fresh", "recompute"}:
                    raise RuntimeError("native admission lacks an allocation selection")
                if action == "prepare" and not record.cancelled:
                    self.register(record.req, record.snapshot_id, record.generation_key,
                                  owner=command["owner"])
                else:
                    # ABORT must not create a new intent, and a cancelled
                    # follower must not revive one just to reject PREPARE.
                    record.owner = command["owner"]
            if command["owner"] != record.owner:
                raise RuntimeError("native admission owner changed")
            key = None if command["key"] is None else LeaseKey(*command["key"])
            if record.key is not None and (record.key != key or record.nonce != command["nonce"]):
                raise RuntimeError("native admission attempt changed")
            record.key, record.nonce = key, command["nonce"]
            if action == "prepare":
                self._prepare(record)
            elif action == "commit":
                if not record.prepared or record.failed or record.aborted:
                    raise RuntimeError("rank-zero COMMIT lacks this rank's prepared Req reference")
                record.committed = True
                if record.req.rid not in self._committed_order:
                    self._committed_order.append(record.req.rid)
            elif action == "abort":
                record.cancelled = True
                # An observed-only parent was never authorized a fresh grant.
                # Retiring this metadata must not touch its unrelated native
                # parent/P->D ownership, which keeps its original fences.
                unselected = record.owner is None and record.key is None
                lease = None if unselected else self.broker.get(record.workset_id)
                if record.aborted or unselected or self.on_abort(record.req, lease) is True:
                    record.aborted = True
                    record.committed = False
                    if record.req.rid in self._committed_order:
                        self._committed_order.remove(record.req.rid)
                    self.mailbox.publish_local(self._abort_key(record), 1)
            elif action == "forget":
                if not record.aborted:
                    raise RuntimeError("native abort retirement lacks local cleanup")
                self._forget(record)
            else:
                raise ValueError("unknown native admission command")
        # Primitive header is copied to prevent caller mutation hiding replay.
        self._last_control = json.loads(json.dumps(control))
        self._last_applied = sequence

    def _prepare(self, record):
        if record.prepared or record.failed:
            self.mailbox.publish_local(self._prepare_key(record), -1 if record.failed else 1)
            return
        try:
            with self.broker._lock:
                lease = self.broker.get(record.workset_id)
                if (record.cancelled or lease is None or lease.controller_plan.key != record.key
                        or lease.parent_tokens or lease.prompt_tokens != record.prompt_tokens
                        or record.req.extra_key != record.generation_key):
                    raise RuntimeError("fresh prepare lost exact ownership before Req handoff")
                # This method arbitrates against cancellation under the same
                # broker lock. A handed Req is a real reference for close().
                self.broker.handoff_fresh_to_req(record.workset_id, record.req, lease)
                record.prepared = True
        except Exception:
            record.failed = True
            self.mailbox.publish_local(self._prepare_key(record), -1)
        else:
            self.mailbox.publish_local(self._prepare_key(record), 1)
            if self.tp_size == 1:
                # No remote rank can disagree; the native handoff itself is
                # the complete single-rank preparation/commit boundary.
                record.committed = True
                if record.req.rid not in self._committed_order:
                    self._committed_order.append(record.req.rid)

    def forget(self, req):
        record = self._requests.get(req.rid)
        if record is None:
            return
        if record.req is not req or not record.committed:
            raise RuntimeError("cannot forget a pending native admission")
        # The caller must invoke only at the request's ordinary terminal native
        # cleanup, never to revoke a live Forward. Physical ledger outlives this.
        self._forget(record)

    def _forget(self, record):
        self._requests.pop(record.req.rid)
        del self._snapshots[record.snapshot_id]
        if record.req.rid in self._committed_order:
            self._committed_order.remove(record.req.rid)

    def terminal(self, req):
        """Ordinary native terminal cleanup; pending refs require ordered abort."""
        if not self.contains(req):
            return True
        if self.is_committed(req):
            self.forget(req)
            return True
        self.cancel(req)
        return False
