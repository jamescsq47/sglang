"""Event-only TP control transport; no CUDA, files, polling or model collectives.

Rank zero remains the sole *logical* decision maker. This service merely stores
rank reports and broadcasts decisions; a report is not a physical fence unless
its caller has observed one. Disconnects invalidate a group, never recycle KV.
Run/attempt identities cannot be reused, including after clear. Recovery from a
dead participant is intentionally a run-level operation, not automatic replay.

Reads and submissions are bounded in-memory operations. Socket work is confined
to background threads. ``wait_ready``/``flush`` are startup/test facilities and
must not be called from a scheduler or transport progress loop.
"""

from __future__ import annotations

import copy
import hmac
import json
import logging
import queue
import socket
import struct
import threading
import time
from bisect import bisect_right
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass


MAX_FRAME = 4 * 1024 * 1024
MAX_COMMAND = 64 * 1024
MAX_TP = 256
logger = logging.getLogger(__name__)
_DIAGNOSTIC_OPERATIONS = frozenset({
    "snapshot", "update", "ack", "observed_receipt", "observer_failure",
    "retired_workset", "clear_observed", "retire_workset", "barrier",
    "report", "rollback", "command_ack", "receipt", "clear", "command",
})


def _diagnostic_operation(value):
    # Keep arbitrary wire fields/payloads out of diagnostics, including invalid
    # messages. Names here are protocol operations, never request identities.
    return value if isinstance(value, str) and value in _DIAGNOSTIC_OPERATIONS else "unknown"


class ControlUnavailable(RuntimeError):
    """Fail closed: retained physical ownership must not be released."""


@dataclass(frozen=True)
class EventKey:
    snapshot_id: str
    attempt_id: str

    def __post_init__(self):
        if any(
            not isinstance(value, str) or not value
            for value in (self.snapshot_id, self.attempt_id)
        ):
            raise ValueError("snapshot and attempt identity are required")


def _workset_identity(identity):
    """Recognize the existing ordered workset wire keys, not arbitrary UUIDs."""
    namespace, snapshot, attempt = identity
    if not attempt.startswith("["):
        return None
    suffix = next((s for s in (":prepared", ":decisions", ":fenced")
                   if namespace.endswith(s)), "")
    base = namespace[:-len(suffix)] if suffix else namespace
    try:
        value = json.loads(attempt)
    except (ValueError, TypeError):
        return None
    if (not isinstance(value, list) or len(value) not in (3, 5)
            or any(not isinstance(v, str) or not v for v in value[:2])
            or type(value[2]) is not int or value[2] < 1):
        return None
    if suffix in (":decisions", ":fenced"):
        if len(value) != 5 or value[3] != "decision" or type(value[4]) is not int or value[4] < 1:
            return None
    elif len(value) != 3:
        return None
    return base, value[0], value[2]


class _RetiredVersions:
    """Exact disjoint closed intervals; holes are outstanding lease versions."""

    def __init__(self):
        self.ranges = []

    def contains(self, version):
        index = bisect_right(self.ranges, (version + 1,)) - 1
        return index >= 0 and self.ranges[index][1] >= version

    def add(self, version):
        start = end = version
        merged = []
        inserted = False
        for left, right in self.ranges:
            if right + 1 < start:
                merged.append((left, right))
            elif end + 1 < left:
                if not inserted:
                    merged.append((start, end))
                    inserted = True
                merged.append((left, right))
            else:
                start, end = min(start, left), max(end, right)
        if not inserted:
            merged.append((start, end))
        self.ranges = merged


def _encode(value):
    data = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    if len(data) > MAX_FRAME:
        raise ValueError("TP event exceeds maximum frame size")
    return struct.pack("!I", len(data)) + data


def _read_exact(sock, length):
    result = bytearray()
    while len(result) < length:
        part = sock.recv(length - len(result))
        if not part:
            raise EOFError("TP event socket closed")
        result.extend(part)
    return bytes(result)


def _recv(sock):
    length = struct.unpack("!I", _read_exact(sock, 4))[0]
    if not 0 < length <= MAX_FRAME:
        raise ValueError("invalid TP event frame size")
    value = json.loads(_read_exact(sock, length))
    if not isinstance(value, dict):
        raise ValueError("TP event must be an object")
    return value


def _shutdown(sock):
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    sock.close()


def _configure_socket(sock):
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    # Kernel liveness, not application polling. A blackholed node must not
    # leave a seemingly healthy mirror indefinitely. These knobs are optional
    # off Linux; deployment must provide equivalent transport failure detection.
    for name, value in (
        ("TCP_KEEPIDLE", 10),
        ("TCP_KEEPINTVL", 3),
        ("TCP_KEEPCNT", 3),
        ("TCP_USER_TIMEOUT", 15000),
    ):
        option = getattr(socket, name, None)
        if option is not None:
            sock.setsockopt(socket.IPPROTO_TCP, option, value)


class _ConnectionProgress:
    """O(1) event-driven diagnostics, never a liveness or ownership decision.

    Each I/O thread writes its own stage. Failure snapshots are approximate
    across threads and deliberately do not acquire the state/cache lock: that
    lock may be the very thing preventing the reader from making progress.
    No payload, authentication token, request identity or historical scan.
    """

    def __init__(self, sock):
        self.local = self.remote = None
        if sock is not None:
            # A peer may close immediately after accept; diagnostics must not
            # make that race kill the listener instead of its own session.
            for attribute, method in (("local", "getsockname"), ("remote", "getpeername")):
                try:
                    setattr(self, attribute, getattr(sock, method)())
                except OSError:
                    pass
        now = time.monotonic()
        self.send_stage = self.recv_stage = ("starting", now)
        self.last_send = self.last_receive = self.last_apply = None
        # The server's coalesced outgoing frames have no single connection
        # sequence; leave it null rather than pretending no ACKs were sent.
        self.last_send_seq = self.last_receive_seq = None
        self.seq = self.acked = 0
        self.send_bytes = 0
        self.send_seq = None
        self.send_operation = None
        self.recv_operation = None
        self.failure = None
        self._failure_lock = threading.Lock()

    def stage(self, direction, value):
        setattr(self, direction + "_stage", (value, time.monotonic()))

    def capture_failure(self, *, origin, group, rank, seq, acked, pending, ack_semantics="received"):
        # Separate tiny lock, released before acquiring any state lock. First
        # failure wins even when shutdown wakes the other I/O thread at once.
        with self._failure_lock:
            if self.failure is not None:
                return None
            now = time.monotonic()
            age = lambda stamp: None if stamp is None else round(now - stamp, 6)
            send_stage, send_since = self.send_stage
            recv_stage, recv_since = self.recv_stage
            self.failure = {
                "origin": origin, "group": group, "rank": rank,
                "local": self.local, "remote": self.remote,
                "submitted_seq": seq, "acked_seq": acked,
                "ack_semantics": ack_semantics,
                "pending_frames": pending,
                "last_send_seq": self.last_send_seq,
                "last_receive_seq": self.last_receive_seq,
                "send_stage": send_stage, "send_stage_age_s": age(send_since),
                "recv_stage": recv_stage, "recv_stage_age_s": age(recv_since),
                "send_bytes": self.send_bytes,
                "send_seq": self.send_seq,
                "send_operation": self.send_operation,
                "recv_operation": self.recv_operation,
                "last_send_age_s": age(self.last_send),
                "last_receive_age_s": age(self.last_receive),
                "last_apply_age_s": age(self.last_apply),
            }
            return self.failure


class _StateOutbox:
    """Bounded pending state mirrors, not a queue of redundant rank deltas.

    An update contains the entire accumulated view for one command incarnation.
    Replacing its unsent copy preserves every rank's report/fence. Different
    command IDs and all non-update events retain distinct FIFO slots. Cumulative
    connection ACKs go last so flush() cannot overtake the states it confirms.
    Producers never wait for a socket; genuinely distinct excess work fails
    closed just as before.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self._condition = threading.Condition()
        self._pending = OrderedDict()
        self._coalesced = {}

    def put_nowait(self, frame, *, key=None, barrier=None, last=False):
        with self._condition:
            slot = self._coalesced.get(key) if key is not None else None
            if slot is None and len(self._pending) >= self.capacity:
                raise queue.Full("TP peer pending-state queue full")
            # A receipt/command/clear or replaceable state is a FIFO boundary:
            # subsequent reports must not overwrite a slot BEFORE that event.
            if barrier is not None:
                self._coalesced.pop(barrier, None)
            if slot is None:
                slot = object()
            self._pending[slot] = (key, frame)
            if key is not None:
                self._coalesced[key] = slot
            if last:
                self._pending.move_to_end(slot)
            self._condition.notify()

    def get(self):
        with self._condition:
            self._condition.wait_for(lambda: bool(self._pending))
            slot, (key, frame) = self._pending.popitem(last=False)
            if key is not None and self._coalesced.get(key) is slot:
                del self._coalesced[key]
            return frame


class _Peer:
    def __init__(self, sock, capacity):
        self.sock = sock
        self.outbox = _StateOutbox(capacity)
        self.group = None
        self.rank = None
        self.progress = _ConnectionProgress(sock)

    def send(self, message, *, cumulative=False):
        key, barrier, last = None, None, False
        if message["type"] == "update":
            entry = message["entry"]
            identity = ("update", *entry["identity"], entry["command_id"])
            if cumulative:
                key = identity
            else:
                barrier = identity
        elif message["type"] == "ack":
            key, last = ("ack",), True
        self.outbox.put_nowait(_encode(message), key=key, barrier=barrier, last=last)


class TPEventServer:
    """Run-scoped control service; bind only to an explicitly trusted network.

    ``token`` authenticates participants, not encryption. Use a private network
    (or an authenticated tunnel); do not expose this endpoint to the Internet.
    Memory is bounded by ``max_entries`` (including tombstones) and per-peer
    queue capacity. Exhaustion fails the affected group rather than dropping an
    event or blocking its scheduler. Generic keys are never automatically
    evicted. Ordered workset families may retire after all-rank FREE execution;
    exact compact version intervals prevent their old messages from reviving.
    """

    def __init__(
        self,
        run_id,
        token,
        address=("127.0.0.1", 0),
        *,
        queue_capacity=1024,
        max_entries=100000,
        max_groups=64,
        max_peers=512,
        receipt_observers=None,
    ):
        if (
            not run_id
            or not token
            or min(queue_capacity, max_entries, max_groups, max_peers) < 1
        ):
            raise ValueError("run/token and positive bounds required")
        self.run_id, self.token = str(run_id), str(token)
        self.capacity, self.max_entries = queue_capacity, max_entries
        self.max_groups, self.max_peers = max_groups, max_peers
        self._entry_count = 0
        # Explicit deployment authority: (producer group, namespace) -> groups
        # allowed to observe its logical receipt, never its shard reports.
        self._receipt_observers = {
            tuple(key): frozenset(groups)
            for key, groups in (receipt_observers or {}).items()
        }
        if any(
            len(key) != 2
            or any(not isinstance(v, str) or not v for v in (*key, *groups))
            for key, groups in self._receipt_observers.items()
        ):
            raise ValueError("invalid cross-group receipt observer mapping")
        self._observed = {}
        self._lock = threading.RLock()
        self._groups, self._peers = {}, set()
        self.errors = deque(maxlen=64)
        self._closed = False
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(address)
        self._socket.listen()
        self.address = self._socket.getsockname()
        self._thread = threading.Thread(target=self._accept, daemon=True)
        self._thread.start()

    def _accept(self):
        while True:
            try:
                sock, _ = self._socket.accept()
            except OSError:
                return
            _configure_socket(sock)
            peer = _Peer(sock, self.capacity)
            with self._lock:
                if self._closed:
                    _shutdown(sock)
                    return
                if len(self._peers) >= self.max_peers:
                    _shutdown(sock)
                    continue
                self._peers.add(peer)
            threading.Thread(target=self._serve, args=(peer,), daemon=True).start()

    def _send_loop(self, peer):
        try:
            while True:
                peer.progress.stage("send", "outbox_wait")
                frame = peer.outbox.get()
                if frame is None:
                    return
                peer.progress.send_bytes = len(frame)
                peer.progress.stage("send", "sendall")
                peer.sock.sendall(frame)
                peer.progress.last_send = time.monotonic()
        except OSError as exc:
            self._fail(peer.group, exc, peer=peer, origin="send")

    def _fail(self, group, error, *, peer=None, origin="application"):
        if peer is not None:
            diagnostic = peer.progress.capture_failure(
                origin=origin, group=group, rank=peer.rank,
                seq=peer.progress.seq, acked=peer.progress.acked,
                pending=len(peer.outbox._pending), ack_semantics="enqueued",
            )
            if diagnostic is not None:
                logger.error("TP event peer failed: error=%s: %s diagnostic=%s",
                             type(error).__name__, error, json.dumps(diagnostic, sort_keys=True))
        with self._lock:
            state = self._groups.get(group)
            if state is None or state["failed"] is not None:
                return
            # queue.Full has an empty str(): never use it as a false-y failure
            # sentinel, or subsequent socket errors repeat group teardown.
            reason = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)
            state["failed"] = reason or "unspecified TP event failure"
            self.errors.append((group, state["failed"]))
            logger.error("TP event group failed: group=%s reason=%s", group, state["failed"])
            for peer in state["peers"].values():
                _shutdown(peer.sock)
                try:
                    peer.outbox.put_nowait(None)
                except queue.Full:
                    pass
            observer_groups = set().union(
                *(
                    groups
                    for (source, _), groups in self._receipt_observers.items()
                    if source == group
                )
            )
            for observer in observer_groups:
                target_state = self._groups.get(observer)
                if target_state and not target_state["failed"]:
                    for peer in target_state["peers"].values():
                        try:
                            peer.send({"type": "observer_failure", "group": group})
                        except queue.Full:
                            self._fail(observer, "receipt observer queue full")

    def _serve(self, peer):
        try:
            peer.sock.settimeout(10)
            peer.progress.stage("recv", "handshake_read")
            hello = _recv(peer.sock)
            peer.progress.last_receive = time.monotonic()
            peer.sock.settimeout(None)
            peer.progress.stage("recv", "handshake_validate")
            if hello.get("run") != self.run_id or not hmac.compare_digest(
                str(hello.get("token", "")), self.token
            ):
                raise ValueError("wrong run/token")
            group, rank, size = hello["group"], hello["rank"], hello["size"]
            if (
                not isinstance(group, str)
                or not group
                or type(rank) is not int
                or type(size) is not int
                or not 0 <= rank < size <= MAX_TP
            ):
                raise ValueError("invalid group/rank")
            with self._lock:
                if group not in self._groups and len(self._groups) >= self.max_groups:
                    raise ValueError("TP group limit reached")
                state = self._groups.setdefault(
                    group,
                    {
                        "size": size,
                        "peers": {},
                        "entries": {},
                        "workset_entries": {},
                        "retired_worksets": {},
                        "failed": None,
                    },
                )
                if state["failed"] or state["size"] != size or rank in state["peers"]:
                    raise ValueError("group failed, size mismatch or duplicate rank")
                peer.group, peer.rank = group, rank
                state["peers"][rank] = peer
                threading.Thread(
                    target=self._send_loop, args=(peer,), daemon=True
                ).start()
                # The snapshot and later updates share a single FIFO, so a late
                # joiner cannot overwrite a new report with older initial state.
                peer.send(
                    {
                        "type": "snapshot",
                        "entries": [
                            self._view(entry, rank)
                            for entry in state["entries"].values()
                        ],
                        "observed": [
                            value
                            for identity, value in self._observed.items()
                            if identity[0] == group
                        ],
                        "observer_failures": [
                            source
                            for (
                                source,
                                _,
                            ), observers in self._receipt_observers.items()
                            if group in observers
                            and self._groups.get(source, {}).get("failed")
                        ],
                        "observer_permissions": [
                            list(key)
                            for key, observers in self._receipt_observers.items()
                            if group in observers
                        ],
                        "retired_worksets": [
                            [base, incarnation, retired.ranges]
                            for (base, incarnation), retired in state["retired_worksets"].items()
                        ],
                    }
                )
            sequence = 0
            while True:
                peer.progress.stage("recv", "socket_read")
                message = _recv(peer.sock)
                peer.progress.last_receive = time.monotonic()
                peer.progress.recv_operation = _diagnostic_operation(message.get("op"))
                peer.progress.stage("recv", "validate")
                if (
                    type(message.get("seq")) is not int
                    or message["seq"] != sequence + 1
                ):
                    raise ValueError("out-of-order connection sequence")
                sequence += 1
                peer.progress.seq = peer.progress.last_receive_seq = sequence
                peer.progress.stage("recv", "state_lock_wait")
                with self._lock:
                    peer.progress.stage("recv", "apply")
                    if state["failed"]:
                        raise ControlUnavailable(state["failed"])
                    if message["op"] == "clear_observed":
                        self._clear_observed(peer, message)
                        peer.send({"type": "ack", "seq": sequence})
                        peer.progress.acked = sequence
                        peer.progress.last_apply = time.monotonic()
                        continue
                    if message["op"] == "retire_workset":
                        event = self._retire_workset(state, rank, message)
                        if event is not None:
                            for target in state["peers"].values():
                                target.send(event)
                        peer.send({"type": "ack", "seq": sequence})
                        peer.progress.acked = sequence
                        peer.progress.last_apply = time.monotonic()
                        continue
                    update = self._apply(state, rank, message)
                    if update is not None:
                        for target in state["peers"].values():
                            # Reports are O(TP), not all-to-all: only rank zero
                            # reduces shards. A follower also gets its own ACK
                            # mirror, but never its peers' reports.
                            if message["op"] in {
                                "report",
                                "rollback",
                                "command_ack",
                            } and target.rank not in (0, rank):
                                continue
                            target.send(
                                {
                                    "type": "update",
                                    "entry": self._view(update, target.rank),
                                },
                                cumulative=(
                                    message["op"] in {"rollback", "command_ack"}
                                    or (message["op"] == "report"
                                        and message.get("mode", "progress") == "progress")
                                ),
                            )
                        if message["op"] in {"receipt", "clear"}:
                            self._publish_observed(group, update)
                    peer.send({"type": "ack", "seq": sequence})
                    peer.progress.acked = sequence
                    peer.progress.last_apply = time.monotonic()
        except Exception as exc:
            # Any unexpected worker failure must invalidate cached decisions;
            # silently losing a background receiver is not a recoverable ACK.
            origin = "recv" if peer.progress.recv_stage[0] in {"socket_read", "handshake_read"} else "apply"
            self._fail(peer.group, exc, peer=peer, origin=origin)
        finally:
            _shutdown(peer.sock)
            with self._lock:
                self._peers.discard(peer)
            try:
                peer.outbox.put_nowait(None)
            except queue.Full:
                pass

    def _publish_observed(self, source, entry):
        namespace, snapshot, attempt = entry["identity"]
        for observer in self._receipt_observers.get((source, namespace), ()):
            key = (observer, source, namespace, snapshot, attempt)
            old = self._observed.get(key)
            if old and old["retired"]:
                continue
            if key not in self._observed and len(self._observed) >= self.max_entries:
                raise ControlUnavailable("receipt observer record limit reached")
            value = {
                "type": "observed_receipt",
                "group": source,
                "identity": entry["identity"],
                "receipt": entry["receipt"],
                "retired": bool(entry["cleared"]),
            }
            self._observed[key] = value
            target_state = self._groups.get(observer)
            if target_state and not target_state["failed"]:
                for peer in target_state["peers"].values():
                    try:
                        peer.send(value)
                    except queue.Full:
                        self._fail(observer, "receipt observer queue full")

    def _clear_observed(self, peer, message):
        source = message["source_group"]
        identity = message["identity"]
        if (
            peer.rank != 0
            or not isinstance(identity, list)
            or len(identity) != 3
            or any(not isinstance(v, str) or not v for v in identity)
            or peer.group not in self._receipt_observers.get((source, identity[0]), ())
        ):
            raise ValueError("unauthorized cross-group receipt cleanup")
        key = (peer.group, source, *identity)
        if key not in self._observed and len(self._observed) >= self.max_entries:
            raise ControlUnavailable("receipt observer record limit reached")
        value = {
            "type": "observed_receipt",
            "group": source,
            "identity": identity,
            "receipt": None,
            "retired": True,
        }
        self._observed[key] = value
        # Retire only this observer's copy. D's authoritative reports/receipt
        # remain until D's own cleanup; P cannot impersonate D rank zero.
        for target in self._groups[peer.group]["peers"].values():
            target.send(value)

    @staticmethod
    def _view(entry, rank):
        if rank == 0:
            return entry
        return {
            **entry,
            "reports": (
                {str(rank): entry["reports"][str(rank)]}
                if str(rank) in entry["reports"]
                else {}
            ),
            "rollback": [rank] if rank in entry["rollback"] else [],
            "command_acks": [rank] if rank in entry["command_acks"] else [],
        }

    def _retire_workset(self, state, rank, msg):
        """Reclaim control history only after the existing all-rank FREE ACK.

        This is not a DMA fence or allocator free. Those already preceded the
        final FREE command. Generic UUID tombstones keep their original rules.
        """
        identity = msg.get("identity")
        final = msg.get("final_identity")
        if (rank != 0 or not isinstance(identity, list) or len(identity) != 3
                or not isinstance(final, list) or len(final) != 3
                or any(not isinstance(v, str) or not v for v in identity + final)):
            raise ValueError("rank zero and exact workset retirement identities required")
        scope = _workset_identity(identity)
        if scope is None or identity[0] != scope[0] or _workset_identity(final) != scope:
            raise ValueError("retirement does not identify one workset family")
        base, incarnation, version = scope
        family = {base, base + ":prepared", base + ":decisions", base + ":fenced"}
        if any(namespace in family for _, namespace in self._receipt_observers):
            raise ValueError("observed receipt namespaces cannot use workset retirement")
        if final[0] != base + ":decisions" or final[1] != identity[1]:
            raise ValueError("retirement requires exact final decision")
        grant_fields, final_fields = json.loads(identity[2]), json.loads(final[2])
        if grant_fields != final_fields[:3]:
            raise ValueError("retirement lease attempt mismatch")
        retired = state["retired_worksets"].get((base, incarnation))
        if retired is not None and retired.contains(version):
            return None  # Exact version cannot regain authority on a retry.
        entry = state["entries"].get(tuple(final))
        command = entry and entry.get("command")
        grant = state["entries"].get(tuple(identity))
        plan = grant and grant.get("command")
        if (not isinstance(plan, dict) or plan.get("incarnation") != incarnation
                or plan.get("snapshot_id") != identity[1]
                or plan.get("attempt_id") != grant_fields[1]
                or plan.get("version") != version):
            raise ValueError("retirement requires exact original workset grant")
        if (not isinstance(command, dict) or command.get("operation") != "free"
                or command.get("incarnation") != incarnation
                or command.get("snapshot_id") != identity[1]
                or command.get("attempt_id") != grant_fields[1]
                or command.get("version") != version
                or command.get("sequence") != final_fields[4]
                or set(entry["command_acks"]) != set(range(state["size"]))):
            raise ValueError("workset retirement requires all-rank exact FREE execution ACK")
        keys = state["workset_entries"].get(scope, set())
        for key in keys:
            current = state["entries"][key]
            if (key[1] != identity[1] or json.loads(key[2])[:3] != grant_fields
                    or (current["command_id"] and set(current["command_acks"]) != set(range(state["size"])))):
                raise ValueError("workset family has conflicting identity or unacknowledged command")
        if any(old_base == base and old_inc != incarnation
               for old_base, old_inc in state["retired_worksets"]):
            raise ValueError("workset incarnation cannot change inside one TP session")
        retired = state["retired_worksets"].setdefault((base, incarnation), _RetiredVersions())
        retired.add(version)
        for key in state["workset_entries"].pop(scope, ()):
            del state["entries"][key]
            self._entry_count -= 1
        return {"type": "retired_workset", "namespace": base,
                "incarnation": incarnation, "version": version}

    def _apply(self, state, rank, msg):
        op = msg["op"]
        if op == "barrier":
            return None
        identity = msg["identity"]
        if (
            not isinstance(identity, list)
            or len(identity) != 3
            or any(not isinstance(v, str) or not v for v in identity)
        ):
            raise ValueError("namespace/snapshot/attempt required")
        key = tuple(identity)
        if op in {"receipt", "command", "clear"} and rank != 0:
            raise ValueError("only rank zero may decide/clear")
        if op not in {
            "report",
            "rollback",
            "receipt",
            "command",
            "command_ack",
            "clear",
        }:
            raise ValueError("unknown TP operation")
        scope = _workset_identity(identity)
        if scope is not None:
            retired = state["retired_worksets"].get(scope[:2])
            if retired is not None and retired.contains(scope[2]):
                return None  # A late old report must not recreate freed history.
        entry = state["entries"].get(key)
        if entry is None:
            if self._entry_count >= self.max_entries:
                # Failure-only inspection: no history scans on the normal
                # control path. Attribute live versus tombstone growth rather
                # than blaming the namespace that happened to hit the bound.
                counts = Counter(
                    (group, item["identity"][0], "cleared" if item["cleared"] else "active")
                    for group, group_state in self._groups.items()
                    for item in group_state["entries"].values()
                )
                detail = ", ".join(f"{group}/{namespace}/{kind}={count}"
                                   for (group, namespace, kind), count in counts.most_common(12))
                active = sum(count for (_, _, kind), count in counts.items() if kind == "active")
                raise ControlUnavailable(
                    f"TP entry/tombstone limit reached: total={self._entry_count} "
                    f"active={active} cleared={sum(counts.values()) - active}; {detail}"
                )
            entry = {
                "identity": identity,
                "reports": {},
                "rollback": [],
                "receipt": None,
                "command": None,
                "command_id": 0,
                "command_acks": [],
                "cleared": False,
            }
            state["entries"][key] = entry
            if scope is not None:
                state["workset_entries"].setdefault(scope, set()).add(key)
            self._entry_count += 1
        if entry["cleared"]:
            # Delayed report from the old operation must not resurrect it.
            return None
        if op == "clear":
            if entry["command_id"] and set(entry["command_acks"]) != set(
                range(state["size"])
            ):
                raise ValueError("cannot clear an unacknowledged command")
            entry.update(
                reports={}, rollback=[], receipt=None, command=None, cleared=True
            )
        elif op == "report":
            value = int(msg["status"])
            previous = entry["reports"].get(str(rank))
            mode = msg.get("mode", "progress")
            if entry.get("mode", mode) != mode:
                raise ValueError("inconsistent report ordering contract")
            entry["mode"] = mode
            if mode == "progress":
                if previous is not None and (
                    previous < 0 or (value >= 0 and value <= previous)
                ):
                    return None
            elif mode == "transfer":
                # Explicit terminal codes supplied consistently by all ranks.
                codes = msg["codes"]
                if entry.get("codes", codes) != codes or len(set(codes)) != 3:
                    raise ValueError("inconsistent transfer status codes")
                entry["codes"] = codes
                failed, success, _ = codes
                if previous in (failed, success):
                    if value != previous:
                        raise ValueError("physical terminal report changed")
                    return None
            elif mode == "state":
                # Scheduler-owned phases are replaceable: failed I/O may
                # later become cleanup-quiescent. Per-peer socket ordering
                # and exact attempt identity still guard wire messages.
                if previous == value:
                    return None
            else:
                raise ValueError("invalid report mode")
            entry["reports"][str(rank)] = value
        elif op == "rollback":
            if rank not in entry["rollback"]:
                entry["rollback"].append(rank)
        elif op == "command":
            command_id = msg["command_id"]
            if type(command_id) is not int or command_id < 1:
                raise ValueError("positive integer command_id required")
            if len(_encode(msg["value"])) > MAX_COMMAND:
                raise ValueError("command payload too large")
            if command_id == entry["command_id"] and msg["value"] == entry["command"]:
                return None
            if command_id != entry["command_id"] + 1:
                raise ValueError("stale/reordered/reused command identity")
            if entry["command_id"] and set(entry["command_acks"]) != set(
                range(state["size"])
            ):
                raise ValueError("previous command not acknowledged by every rank")
            entry.update(command=msg["value"], command_id=command_id, command_acks=[])
        elif op == "command_ack":
            if (
                type(msg["command_id"]) is not int
                or msg["command_id"] < 1
                or msg["command_id"] != entry["command_id"]
            ):
                raise ValueError("wrong command acknowledgment identity")
            if rank not in entry["command_acks"]:
                entry["command_acks"].append(rank)
        else:
            entry[op] = msg["value"]
        return entry

    def close(self):
        with self._lock:
            self._closed = True
            _shutdown(self._socket)
            for group in list(self._groups):
                self._fail(group, "server shutdown")
            for peer in list(self._peers):
                _shutdown(peer.sock)
        self._thread.join(timeout=2)

    def stats(self):
        """Small monitoring snapshot; no per-entry history traversal."""
        with self._lock:
            return {
                "entry_count": self._entry_count,
                "max_entries": self.max_entries,
                "observed_count": len(self._observed),
                "groups": {
                    group: {
                        "entries": len(state["entries"]),
                        "workset_versions": len(state["workset_entries"]),
                        "retired_intervals": sum(len(retired.ranges) for retired in state["retired_worksets"].values()),
                        "retired_versions": sum(end - start + 1 for retired in state["retired_worksets"].values()
                                                for start, end in retired.ranges),
                        "failed": state["failed"],
                    }
                    for group, state in self._groups.items()
                },
            }


class TPEventClient:
    """One persistent session per physical TP rank, shared by all namespaces."""

    def __init__(
        self,
        address,
        *,
        run_id,
        token,
        group,
        rank,
        size,
        queue_capacity=1024,
        connect_timeout=10,
    ):
        if (
            type(rank) is not int
            or type(size) is not int
            or not 0 <= rank < size <= MAX_TP
            or queue_capacity < 1
        ):
            raise ValueError("invalid rank/size/capacity")
        self.rank, self.size = rank, size
        self._condition = threading.Condition(threading.RLock())
        self._entries, self._seq, self._acked = {}, 0, 0
        self._workset_cache_keys, self._retired_worksets = {}, {}
        self._observed, self._observer_failures = {}, set()
        self._observer_permissions = set()
        self._reported = {}
        self._published_receipts = {}
        self._command_ids, self._delivered_commands = {}, {}
        self._command_inboxes = {}
        self._command_inbox_size = 0
        self._command_inbox_limit = queue_capacity
        # Opt-in consumers only: existing transport namespaces do not acquire
        # another queue or incur a historical-state scan.
        self._update_inboxes = {}
        self._update_inbox_size = 0
        self._update_inbox_limit = queue_capacity
        self._error, self._ready = None, False
        self._outbox = queue.Queue(queue_capacity)
        self.changed = threading.Event()
        self._socket = socket.create_connection(address, timeout=connect_timeout)
        self._socket.settimeout(None)
        _configure_socket(self._socket)
        self.group = group
        self._progress = _ConnectionProgress(self._socket)
        self._socket.sendall(
            _encode(
                {
                    "run": run_id,
                    "token": token,
                    "group": group,
                    "rank": rank,
                    "size": size,
                }
            )
        )
        self._sender = threading.Thread(target=self._send_loop, daemon=True)
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._sender.start()
        self._reader.start()

    def _fail(self, exc, *, origin="application"):
        diagnostic = self._progress.capture_failure(
            origin=origin, group=self.group, rank=self.rank,
            seq=self._seq, acked=self._acked, pending=self._outbox.qsize(),
        )
        if diagnostic is not None and origin != "shutdown":
            # Emit before taking the mirror lock, including if its holder is
            # stalled. The original exception remains the fail-closed reason.
            logger.error("TP event client failed: error=%s: %s diagnostic=%s",
                         type(exc).__name__, exc, json.dumps(diagnostic, sort_keys=True))
        with self._condition:
            if self._error is None:
                self._error = str(exc)
            self.changed.set()
            self._condition.notify_all()
        _shutdown(self._socket)
        try:
            self._outbox.put_nowait(None)
        except queue.Full:
            pass

    def _check(self):
        if self._error is not None:
            raise ControlUnavailable(self._error)

    def _send_loop(self):
        try:
            while True:
                self._progress.stage("send", "outbox_wait")
                item = self._outbox.get()
                if item is None:
                    return
                frame, sequence, operation = item
                self._progress.send_bytes = len(frame)
                self._progress.send_seq = sequence
                self._progress.send_operation = operation
                self._progress.stage("send", "sendall")
                self._socket.sendall(frame)
                self._progress.last_send = time.monotonic()
                self._progress.last_send_seq = sequence
        except OSError as exc:
            self._fail(exc, origin="send")

    def _read_loop(self):
        try:
            while True:
                self._progress.stage("recv", "socket_read")
                event = _recv(self._socket)
                self._progress.last_receive = time.monotonic()
                self._progress.recv_operation = _diagnostic_operation(event.get("type"))
                self._progress.stage("recv", "mirror_lock_wait")
                with self._condition:
                    self._progress.stage("recv", "apply")
                    if event["type"] == "snapshot":
                        if self._ready:
                            raise ValueError("duplicate initial snapshot")
                        self._entries = {
                            tuple(v["identity"]): v for v in event["entries"]
                        }
                        for base, incarnation, ranges in event.get("retired_worksets", []):
                            retired = _RetiredVersions()
                            retired.ranges = [tuple(pair) for pair in ranges]
                            self._retired_worksets[(base, incarnation)] = retired
                        for value in self._entries.values():
                            self._index_workset_key(tuple(value["identity"]))
                            self._enqueue_command_event(value, None)
                            self._enqueue_update_event(value)
                        self._observed = {
                            (v["group"], *v["identity"]): v
                            for v in event.get("observed", [])
                        }
                        self._observer_failures = set(
                            event.get("observer_failures", [])
                        )
                        self._observer_permissions = {
                            tuple(v) for v in event.get("observer_permissions", [])
                        }
                        self._ready = True
                    elif event["type"] == "update":
                        value = event["entry"]
                        self._index_workset_key(tuple(value["identity"]))
                        self._enqueue_command_event(
                            value, self._entries.get(tuple(value["identity"]))
                        )
                        self._entries[tuple(value["identity"])] = value
                        self._enqueue_update_event(value)
                    elif event["type"] == "ack":
                        self._acked = event["seq"]
                        self._progress.last_receive_seq = self._acked
                    elif event["type"] == "observed_receipt":
                        self._observed[(event["group"], *event["identity"])] = event
                    elif event["type"] == "observer_failure":
                        self._observer_failures.add(event["group"])
                    elif event["type"] == "retired_workset":
                        self._forget_workset(event["namespace"], event["incarnation"], event["version"])
                    else:
                        raise ValueError("unknown TP event")
                    self.changed.set()
                    self._condition.notify_all()
                    self._progress.last_apply = time.monotonic()
        except Exception as exc:
            self._fail(exc, origin="recv" if self._progress.recv_stage[0] == "socket_read" else "apply")

    def _index_workset_key(self, identity):
        scope = _workset_identity(identity)
        if scope is not None:
            retired = self._retired_worksets.get(scope[:2])
            if retired is None or not retired.contains(scope[2]):
                self._workset_cache_keys.setdefault(scope, set()).add(identity)

    def _forget_workset(self, base, incarnation, version):
        self._retired_worksets.setdefault((base, incarnation), _RetiredVersions()).add(version)
        keys = self._workset_cache_keys.pop((base, incarnation, version), ())
        for identity in keys:
            for cache in (self._entries, self._reported, self._published_receipts,
                          self._command_ids, self._delivered_commands):
                cache.pop(identity, None)
            inbox = self._update_inboxes.get(identity[0])
            if inbox is not None and identity in inbox:
                del inbox[identity]
                self._update_inbox_size -= 1
        # Only bounded live command inboxes are inspected, never history maps.
        for namespace in {identity[0] for identity in keys}:
            inbox = self._command_inboxes.get(namespace)
            if inbox:
                kept = deque(item for item in inbox if item[0] not in keys)
                self._command_inbox_size -= len(inbox) - len(kept)
                self._command_inboxes[namespace] = kept

    def _is_retired_workset(self, identity):
        scope = _workset_identity(identity)
        retired = self._retired_worksets.get(scope[:2]) if scope else None
        return retired is not None and retired.contains(scope[2])

    def _enqueue_command_event(self, value, previous):
        command_id = value.get("command_id", 0)
        if (
            value.get("cleared")
            or not command_id
            or self.rank in value.get("command_acks", ())
            or (previous and previous.get("command_id") == command_id)
        ):
            return
        if self._command_inbox_size >= self._command_inbox_limit:
            raise ControlUnavailable("TP command inbox full")
        identity = tuple(value["identity"])
        self._command_inboxes.setdefault(identity[0], deque()).append(
            (identity, command_id)
        )
        self._command_inbox_size += 1

    def drain_commands(self, namespace, limit=128):
        """Consume only newly pushed commands; no full-history scan or I/O."""
        with self._condition:
            self._check()
            inbox = self._command_inboxes.get(namespace)
            result = []
            while inbox and len(result) < limit:
                identity, command_id = inbox.popleft()
                self._command_inbox_size -= 1
                key = EventKey(identity[1], identity[2])
                command = self.next_command(namespace, key)
                if command is not None:
                    if command[0] != command_id:
                        raise ControlUnavailable("unconsumed command changed")
                    result.append((key, command[0], command[1]))
            return result

    def wait_commands(self, namespace, *, limit=128, timeout=None):
        """Wait for new commands in a background controller, without polling.

        Never call from the model scheduler. Waiting releases the cache lock;
        command arrivals and connection failure wake the consumer. Draining is
        still exactly-once delivery, not an execution ACK: the caller retains
        pending work until its real local completion or explicit failure.
        """
        if not namespace or limit <= 0:
            raise ValueError("namespace and a positive command limit required")
        with self._condition:
            self._check()
            self._condition.wait_for(
                lambda: self._error is not None
                or bool(self._command_inboxes.get(namespace)),
                timeout=timeout,
            )
            self._check()
            return self.drain_commands(namespace, limit=limit)

    def subscribe_updates(self, namespace):
        """Enable one in-memory delta consumer before publishing its work.

        No historical backfill. A late-joining command consumer must inspect
        its initial command's cached state once. Repeated updates for one key
        coalesce; drain_update_keys returns identities whose current state can
        be read from the existing cache, not a second copy of the ledger.
        """
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("nonempty namespace required")
        with self._condition:
            self._check()
            self._update_inboxes.setdefault(namespace, OrderedDict())

    def _enqueue_update_event(self, value):
        identity = tuple(value["identity"])
        inbox = self._update_inboxes.get(identity[0])
        if inbox is None or identity in inbox:
            return
        if self._update_inbox_size >= self._update_inbox_limit:
            raise ControlUnavailable("TP update inbox full")
        inbox[identity] = None
        self._update_inbox_size += 1

    def drain_update_keys(self, namespace, limit=128):
        """Consume only newly changed identities, with bounded O(limit) work."""
        if type(limit) is not int or limit < 1:
            raise ValueError("positive update limit required")
        with self._condition:
            self._check()
            if namespace not in self._update_inboxes:
                raise ValueError("namespace was not subscribed")
            inbox = self._update_inboxes[namespace]
            keys = []
            while inbox and len(keys) < limit:
                identity, _ = inbox.popitem(last=False)
                self._update_inbox_size -= 1
                keys.append(EventKey(identity[1], identity[2]))
            return keys

    @staticmethod
    def _identity(namespace, key):
        if not isinstance(key, EventKey) or not namespace:
            raise ValueError("explicit EventKey and namespace required")
        return [str(namespace), key.snapshot_id, key.attempt_id]

    def _submit(self, op, namespace=None, key=None, **fields):
        with self._condition:
            self._check()
            message = {"op": op, "seq": self._seq + 1, **fields}
            if key is not None:
                message["identity"] = self._identity(namespace, key)
                if self._is_retired_workset(message["identity"]):
                    return self._seq
                self._index_workset_key(tuple(message["identity"]))
            try:
                self._outbox.put_nowait((_encode(message), self._seq + 1, _diagnostic_operation(op)))
            except queue.Full as exc:
                self._fail("TP outbound queue full", origin="enqueue")
                raise ControlUnavailable("TP outbound queue full") from exc
            self._seq += 1
            return self._seq

    def report(self, namespace, key, status):
        return self._report(namespace, key, int(status), mode="progress")

    def report_transfer(self, namespace, key, status, *, failed, success, transferring):
        return self._report(
            namespace,
            key,
            int(status),
            mode="transfer",
            codes=[int(failed), int(success), int(transferring)],
        )

    def report_state(self, namespace, key, status):
        """Replace scheduler state; never use this as a physical DMA fence."""
        return self._report(namespace, key, int(status), mode="state")

    def _report(self, namespace, key, status, **fields):
        with self._condition:
            self._check()
            identity = tuple(self._identity(namespace, key))
            if self._is_retired_workset(identity):
                return self._seq
            previous = self._reported.get(identity)
            value = {"status": status, **fields}
            if previous is not None:
                if previous == value:
                    return self._seq
                if fields["mode"] == previous["mode"] == "progress":
                    old = previous["status"]
                    if old < 0 or (status >= 0 and status <= old):
                        return self._seq
            sequence = self._submit("report", namespace, key, **value)
            self._reported[identity] = value
            return sequence

    def report_rollback(self, namespace, key):
        return self._submit("rollback", namespace, key)

    def _decide(self, op, namespace, key, **fields):
        if self.rank != 0:
            raise ValueError("only rank zero may decide/clear")
        return self._submit(op, namespace, key, **fields)

    def publish_receipt(self, namespace, key, status):
        with self._condition:
            self._check()
            self._require_leader()
            identity = tuple(self._identity(namespace, key))
            if self._is_retired_workset(identity):
                return self._seq
            status = int(status)
            if self._published_receipts.get(identity) == status:
                return self._seq
            sequence = self._decide("receipt", namespace, key, value=status)
            self._published_receipts[identity] = status
            return sequence

    def publish_command(self, namespace, key, command, *, command_id=None):
        """Publish one immutable phase; all ranks must ACK before the next.

        Supplying the same command_id and payload is an idempotent retry. An
        ACK must be emitted only after the caller has performed the command;
        transport delivery alone never ACKs execution or a CUDA fence.
        """
        if len(_encode(command)) > MAX_COMMAND:
            raise ValueError("command payload too large")
        with self._condition:
            identity = tuple(self._identity(namespace, key))
            if self._is_retired_workset(identity):
                return self._seq
            if command_id is None:
                command_id = self._command_ids.get(identity, 0) + 1
            result = self._decide(
                "command", namespace, key, value=command, command_id=command_id
            )
            self._command_ids[identity] = command_id
            return result

    def next_command(self, namespace, key):
        """Take a cached command once, without I/O; returns (id, payload)."""
        with self._condition:
            value = self.entry(namespace, key)
            identity = tuple(self._identity(namespace, key))
            if (
                not value
                or not value["command_id"]
                or self.rank in value["command_acks"]
                or self._delivered_commands.get(identity) == value["command_id"]
            ):
                return None
            self._delivered_commands[identity] = value["command_id"]
            return value["command_id"], value["command"]

    def ack_command(self, namespace, key, command_id):
        return self._submit("command_ack", namespace, key, command_id=command_id)

    def command_complete(self, namespace, key):
        self._require_leader()
        value = self.entry(namespace, key)
        return bool(
            value
            and value["command_id"]
            and set(value["command_acks"]) == set(range(self.size))
        )

    def clear(self, namespace, key):
        return self._decide("clear", namespace, key)

    def retire_workset(self, namespace, grant_key, final_key):
        """Nonblocking rank-zero cleanup after all-rank final FREE execution.

        Returns the submitted connection sequence, like clear(). Never grants
        allocator/DMA authority and never waits on a model scheduler barrier.
        """
        self._require_leader()
        return self._submit("retire_workset", namespace, grant_key,
                            final_identity=self._identity(namespace + ":decisions", final_key))

    def entry(self, namespace, key):
        with self._condition:
            self._check()
            value = self._entries.get(tuple(self._identity(namespace, key)))
            return copy.deepcopy(value) if value and not value["cleared"] else None

    def group_status(self, namespace, key):
        self._require_leader()
        value = self.entry(namespace, key)
        statuses = value["reports"] if value else {}
        return (
            min(statuses.values())
            if set(statuses) == {str(rank) for rank in range(self.size)}
            else None
        )

    def any_negative_report(self, namespace, key):
        """Rank-zero cancellation trigger, NOT a completion/release fence.

        Absent reports grant nothing. One failure can request an ordered abort
        without making every other rank send a neutral initialization message.
        Commit/cleanup still require group_status() from the complete rank set.
        """
        self._require_leader()
        with self._condition:
            self._check()
            value = self._entries.get(tuple(self._identity(namespace, key)))
            return bool(value and not value["cleared"]
                        and any(status < 0 for status in value["reports"].values()))

    def local_status(self, namespace, key, rank=None):
        value = self.entry(namespace, key)
        return (
            value["reports"].get(str(self.rank if rank is None else rank))
            if value
            else None
        )

    def receipt(self, namespace, key):
        value = self.entry(namespace, key)
        return value["receipt"] if value else None

    def observed_receipt(self, source_group, namespace, key):
        with self._condition:
            self._check()
            if (source_group, namespace) not in self._observer_permissions:
                raise ValueError("cross-group receipt mapping not configured")
            if source_group in self._observer_failures:
                raise ControlUnavailable("receipt producer group disconnected")
            value = self._observed.get((source_group, *self._identity(namespace, key)))
            return value["receipt"] if value and not value["retired"] else None

    def clear_observed(self, source_group, namespace, key):
        self._require_leader()
        return self._submit("clear_observed", namespace, key, source_group=source_group)

    def command(self, namespace, key):
        value = self.entry(namespace, key)
        return value["command"] if value else None

    def transfer_group_status(self, namespace, key):
        self._require_leader()
        value = self.entry(namespace, key)
        if not value or "codes" not in value:
            return None, False
        failed, success, transferring = value["codes"]
        statuses = list(value["reports"].values())
        cancel = failed in statuses
        if set(value["reports"]) != {str(rank) for rank in range(self.size)}:
            return None, cancel
        if cancel:
            terminal = all(v in (failed, success) for v in statuses)
            return (failed if terminal else transferring), True
        return (
            success if all(v == success for v in statuses) else min(statuses)
        ), False

    def rollback_group_complete(self, namespace, key):
        self._require_leader()
        value = self.entry(namespace, key)
        return bool(value and set(value["rollback"]) == set(range(self.size)))

    def _require_leader(self):
        if self.rank != 0:
            raise ValueError("only rank zero may reduce group reports")

    def wait_ready(self, timeout=10):
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._ready or self._error, timeout
            )
            self._check()
            if not ready:
                self._fail("TP event handshake timed out")
                raise TimeoutError("TP event handshake timed out")

    def flush(self, timeout=10):
        seq = self._submit("barrier")
        with self._condition:
            done = self._condition.wait_for(
                lambda: self._acked >= seq or self._error, timeout
            )
            self._check()
            if not done:
                self._fail("TP event acknowledgment timed out")
                raise TimeoutError("TP event acknowledgment timed out")

    def close(self):
        self._fail("client shutdown", origin="shutdown")
        for thread in (self._sender, self._reader):
            if thread is not threading.current_thread():
                thread.join(timeout=2)
