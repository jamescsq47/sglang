"""Persistent, run-scoped control RPC and pushed record mirrors.

No files, model collectives, CUDA, automatic reconnect or ownership recovery.
Registered handlers are explicit trusted short CPU transactions; they must not
perform network/DMA/file I/O. They may publish record changes under the same
server lock before returning. The client applies those pushes before making the
result visible. ``submit`` and cache reads are scheduler-safe; ``call`` and
subscription readiness waits belong only in startup or existing I/O workers.

A lost reply may be retried explicitly using the original RPCFuture. The server
returns the exact cached outcome without executing again. Old outcomes outside
the bounded retry window fail explicitly; they are never executed again. Socket
loss fails outstanding futures and cache reads closed, retaining physical KV.
"""

from __future__ import annotations

import copy
import hmac
import json
import os
import queue
import socket
import threading
import time
import uuid
from collections import OrderedDict, deque
from concurrent.futures import Future, InvalidStateError

from sglang.srt.disaggregation.agentic_tp_events import (
    ControlUnavailable,
    _configure_socket,
    _encode,
    _recv,
    _shutdown,
)


class RemoteCallError(RuntimeError):
    pass


_THREAD_POLICY = threading.local()


def set_control_rpc_blocking_allowed(allowed):
    """Disable synchronous RPC on a scheduler thread after startup."""
    previous = getattr(_THREAD_POLICY, "blocking_allowed", True)
    _THREAD_POLICY.blocking_allowed = bool(allowed)
    return previous


def assert_control_rpc_blocking_allowed():
    if not getattr(_THREAD_POLICY, "blocking_allowed", True):
        raise RuntimeError(
            "blocking control operation forbidden on scheduler thread; use submit/cache"
        )


class RPCFuture(Future):
    def __init__(self, owner, operation_id, request):
        super().__init__()
        self.operation_id = operation_id
        self._owner, self._request = owner, request

    def cancel(self):
        # Cancelling a Python waiter must not imply cancelling a committed
        # physical operation. Lifecycle cancellation is a separate typed RPC.
        return False


class Subscription:
    def __init__(self, namespace, capacity, queue_events):
        self.namespace = namespace
        self.ready = Future()
        self.changed = threading.Event()
        self._queue = queue.Queue(capacity) if queue_events else None
        self.initial_snapshot = None
        self.initial_revision = None
        self._close = None

    def get_nowait(self):
        if self._queue is None:
            raise queue.Empty
        return self._queue.get_nowait()

    def close(self):
        if self._close is not None:
            self._close()


class _Snapshot:
    def __init__(self, namespace, revision, records):
        self.namespace, self.revision, self.records = namespace, revision, records

    def frames(self):
        yield _encode(
            {
                "type": "snapshot_begin",
                "namespace": self.namespace,
                "revision": self.revision,
            }
        )
        for frame in self.records:
            yield frame
        yield _encode(
            {
                "type": "snapshot_end",
                "namespace": self.namespace,
                "revision": self.revision,
            }
        )


class _RPCPeer:
    def __init__(self, sock, capacity):
        self.sock = sock
        self.outbox = queue.Queue(capacity)
        self.client_id = None
        self.subscriptions = set()
        self.last_operation = 0
        self.results = OrderedDict()

    def enqueue(self, value):
        self.outbox.put_nowait(
            value if isinstance(value, _Snapshot) else _encode(value)
        )


class ControlRPCServer:
    def __init__(
        self,
        run_id,
        token,
        address=("127.0.0.1", 0),
        *,
        queue_capacity=1024,
        max_clients=256,
        max_records=100000,
        max_namespaces=128,
        retry_window=4096,
        start=True,
    ):
        if (
            not run_id
            or not token
            or min(
                queue_capacity, max_clients, max_records, max_namespaces, retry_window
            )
            < 1
        ):
            raise ValueError("run/token and positive bounds required")
        self.run_id, self.token = str(run_id), str(token)
        self.capacity, self.max_clients = queue_capacity, max_clients
        self.max_records, self.max_namespaces = max_records, max_namespaces
        self.retry_window = retry_window
        self._lock = threading.RLock()
        self._services, self._records, self._revisions = {}, {}, {}
        self._peers, self._client_ids = set(), set()
        self._record_count, self._closed = 0, False
        self.errors = deque(maxlen=64)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(address)
        self._socket.listen()
        self.address = self._socket.getsockname()
        self._accept_thread = threading.Thread(target=self._accept, daemon=True)
        self._started = False
        if start:
            self.start()

    def start(self):
        """Start serving only after deployment has registered every service."""
        with self._lock:
            if self._closed:
                raise ControlUnavailable("control server closed")
            if not self._started:
                self._started = True
                self._accept_thread.start()
        return self

    def register_service(self, name, methods):
        if (
            not isinstance(name, str)
            or not name
            or not methods
            or any(
                not isinstance(method, str) or not method or not callable(handler)
                for method, handler in methods.items()
            )
        ):
            raise ValueError("explicit named callable methods required")
        with self._lock:
            if name in self._services or self._client_ids:
                raise ValueError("register each service once before clients connect")
            self._services[name] = dict(methods)

    def _namespace(self, namespace):
        if not isinstance(namespace, str) or not namespace or len(namespace) > 256:
            raise ValueError("invalid control namespace")
        if namespace not in self._records:
            if len(self._records) >= self.max_namespaces:
                raise ControlUnavailable("control namespace limit reached")
            self._records[namespace], self._revisions[namespace] = {}, 0

    def publish(self, namespace, key, value):
        """Publish a full JSON record or delete with None; returns revision.

        Called inside a registered handler, publication and its transaction
        result are ordered atomically with respect to other calls/snapshots.
        Slow subscribers are disconnected, never waited for under this lock.
        """
        try:
            return self._publish(namespace, key, value)
        except Exception as exc:
            # A handler may already have committed authoritative ownership.
            # Never leave clients using an older mirror after publication fails.
            self.poison(f"control publication failed: {type(exc).__name__}")
            raise

    def _publish(self, namespace, key, value):
        if not isinstance(key, str) or not key or len(key) > 4096:
            raise ValueError("nonempty bounded string record key required")
        with self._lock:
            if self._closed:
                raise ControlUnavailable("control server closed")
            self._namespace(namespace)
            records = self._records[namespace]
            if (
                value is not None
                and key not in records
                and self._record_count >= self.max_records
            ):
                raise ControlUnavailable("control record limit reached")
            revision = self._revisions[namespace] + 1
            message = {
                "type": "delta",
                "namespace": namespace,
                "key": key,
                "revision": revision,
                "value": value,
            }
            frame = _encode(message)  # serialization must succeed before commit
            if value is None:
                if records.pop(key, None) is not None:
                    self._record_count -= 1
            else:
                if key not in records:
                    self._record_count += 1
                # Immutable encoded snapshot items need no deep copy while a
                # late joiner is streaming its initial state in the background.
                records[key] = frame
            self._revisions[namespace] = revision
            for peer in tuple(self._peers):
                if namespace in peer.subscriptions:
                    try:
                        peer.outbox.put_nowait(frame)
                    except queue.Full:
                        self._drop(peer, "control subscriber queue full")
            return revision

    def poison(self, reason):
        """Fatal publication/transaction failure: retain state, reject all use.

        This is not rollback. A caller that mutated state but could not publish
        it must stop the run; it cannot keep serving stale ownership mirrors.
        """
        with self._lock:
            self._closed = True
            _shutdown(self._socket)
            for peer in tuple(self._peers):
                self._drop(peer, reason)

    def _accept(self):
        while True:
            try:
                sock, _ = self._socket.accept()
            except OSError:
                return
            _configure_socket(sock)
            peer = _RPCPeer(sock, self.capacity)
            with self._lock:
                if self._closed or len(self._peers) >= self.max_clients:
                    _shutdown(sock)
                    if self._closed:
                        return
                    continue
                self._peers.add(peer)
            threading.Thread(target=self._serve, args=(peer,), daemon=True).start()

    def _drop(self, peer, reason):
        with self._lock:
            if peer not in self._peers:
                return
            self.errors.append((peer.client_id, str(reason)))
            self._peers.discard(peer)
            _shutdown(peer.sock)
            try:
                peer.outbox.put_nowait(None)
            except queue.Full:
                pass

    def _send(self, peer):
        try:
            while True:
                item = peer.outbox.get()
                if item is None:
                    return
                frames = item.frames() if isinstance(item, _Snapshot) else (item,)
                for frame in frames:
                    peer.sock.sendall(frame)
        except Exception as exc:
            self._drop(peer, exc)

    def _serve(self, peer):
        try:
            peer.sock.settimeout(10)
            hello = _recv(peer.sock)
            peer.sock.settimeout(None)
            if hello.get("run") != self.run_id or not hmac.compare_digest(
                str(hello.get("token", "")), self.token
            ):
                raise ValueError("wrong control run/token")
            identity = hello.get("client_id")
            if not isinstance(identity, str) or not identity or len(identity) > 256:
                raise ValueError("invalid control client identity")
            with self._lock:
                if identity in self._client_ids:
                    raise ValueError("control identity reconnect/reuse forbidden")
                # Bound run-long identities, not merely live sockets.
                if len(self._client_ids) >= self.max_clients:
                    raise ValueError("control client identity limit reached")
                self._client_ids.add(identity)
                peer.client_id = identity
                threading.Thread(target=self._send, args=(peer,), daemon=True).start()
                peer.enqueue({"type": "ready"})
            sequence = 0
            while True:
                request = _recv(peer.sock)
                if (
                    type(request.get("seq")) is not int
                    or request["seq"] != sequence + 1
                ):
                    raise ValueError("control connection sequence mismatch")
                sequence += 1
                with self._lock:
                    if peer not in self._peers or self._closed:
                        raise ControlUnavailable("control connection closed")
                    if request.get("type") == "subscribe":
                        namespace = request["namespace"]
                        self._namespace(namespace)
                        if namespace not in peer.subscriptions:
                            peer.subscriptions.add(namespace)
                            peer.enqueue(
                                _Snapshot(
                                    namespace,
                                    self._revisions[namespace],
                                    tuple(self._records[namespace].values()),
                                )
                            )
                    elif request.get("type") == "call":
                        response = self._dispatch(peer, request)
                        self._reply(
                            peer, {"type": "result", "seq": sequence, **response}
                        )
                    else:
                        raise ValueError("unknown control message")
        except Exception as exc:
            self._drop(peer, exc)
        finally:
            _shutdown(peer.sock)

    def _reply(self, peer, response):
        """Separate hook for committed-but-lost-reply fault tests."""
        peer.enqueue(response)

    def _dispatch(self, peer, request):
        operation_id = request.get("operation_id")
        if type(operation_id) is not int or operation_id < 1:
            raise ValueError("positive integer operation identity required")
        signature = json.dumps(
            {
                name: request.get(name)
                for name in ("service", "method", "args", "kwargs")
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if operation_id <= peer.last_operation:
            cached = peer.results.get(operation_id)
            if cached is None:
                return {
                    "ok": False,
                    "error": "operation result retired; outcome unknown; not re-executed",
                }
            if cached[0] != signature:
                raise ValueError("operation identity reused with different arguments")
            return cached[1]
        if operation_id != peer.last_operation + 1:
            raise ValueError("control operation sequence gap")
        peer.last_operation = operation_id
        try:
            if not isinstance(request["args"], list) or not isinstance(
                request["kwargs"], dict
            ):
                raise ValueError("args list and kwargs object required")
            handler = self._services[request["service"]][request["method"]]
            result = handler(*request["args"], **request["kwargs"])
            # Freeze before caching: later handler mutation cannot change a
            # committed reply returned by an explicit retry.
            response = json.loads(_encode({"ok": True, "value": result})[4:])
        except Exception as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        peer.results[operation_id] = (signature, response)
        while len(peer.results) > self.retry_window:
            peer.results.popitem(last=False)
        return response

    def close(self):
        with self._lock:
            self._closed = True
            _shutdown(self._socket)
            for peer in tuple(self._peers):
                self._drop(peer, "control server shutdown")
        if self._started:
            self._accept_thread.join(timeout=2)


class ControlRPCClient:
    def __init__(
        self,
        address,
        *,
        run_id,
        token,
        client_id=None,
        queue_capacity=1024,
        max_pending=1024,
        connect_timeout=10,
        call_timeout=30,
    ):
        if queue_capacity < 1 or max_pending < 1 or call_timeout <= 0:
            raise ValueError("positive client queue bounds required")
        self._lock = threading.RLock()
        self._ready = threading.Event()
        self._cache_changed = threading.Condition(self._lock)
        self._error = None
        self._seq = self._operation = 0
        self._pending, self._subscriptions = {}, {}
        self._watchers = {}
        self._cache, self._revisions, self._initial = {}, {}, {}
        self.capacity, self.max_pending = queue_capacity, max_pending
        self.call_timeout = call_timeout
        self._outbox = queue.Queue(queue_capacity)
        self._completions = queue.Queue(queue_capacity)
        self.client_id = (
            client_id or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"
        )
        self._socket = socket.create_connection(address, timeout=connect_timeout)
        self._socket.settimeout(None)
        _configure_socket(self._socket)
        self._socket.sendall(
            _encode({"run": run_id, "token": token, "client_id": self.client_id})
        )
        self._threads = [
            threading.Thread(target=target, daemon=True)
            for target in (self._send, self._read, self._complete)
        ]
        for thread in self._threads:
            thread.start()

    def _check(self):
        if self._error is not None:
            raise ControlUnavailable(self._error)

    def _fail(self, reason):
        with self._lock:
            if self._error is not None:
                return
            self._error = str(reason)
            pending = list(self._pending.values())
            subscriptions = list(self._subscriptions.values()) + [
                sub for watchers in self._watchers.values() for sub in watchers
            ]
            pending += [sub.ready for sub in subscriptions if not sub.ready.done()]
            self._pending.clear()
            while True:
                try:
                    completion = self._completions.get_nowait()
                except queue.Empty:
                    break
                if completion is not None:
                    pending.append(completion[0])
            for sub in subscriptions:
                sub.changed.set()
            self._ready.set()
            self._cache_changed.notify_all()
        _shutdown(self._socket)
        for outbox in (self._outbox, self._completions):
            try:
                outbox.put_nowait(None)
            except queue.Full:
                pass

        # Never execute user Future callbacks on the scheduler's queue-full
        # path. This one-shot failure task does no protocol/ownership work.
        def fail_waiters():
            for future in pending:
                try:
                    future.set_exception(ControlUnavailable(self._error))
                except InvalidStateError:
                    # A fully received, committed reply may already have
                    # settled; socket failure cannot retroactively undo it.
                    pass

        threading.Thread(target=fail_waiters, daemon=True).start()

    def _send(self):
        try:
            while True:
                frame = self._outbox.get()
                if frame is None:
                    return
                self._socket.sendall(frame)
        except Exception as exc:
            self._fail(exc)

    def _complete(self):
        try:
            while True:
                item = self._completions.get()
                if item is None:
                    return
                future, response = item
                if future.done():
                    continue
                if response["ok"]:
                    future.set_result(response.get("value"))
                else:
                    future.set_exception(RemoteCallError(response["error"]))
        except Exception as exc:
            self._fail(exc)

    def _queue_completion(self, future, response):
        try:
            self._completions.put_nowait((future, response))
        except queue.Full as exc:
            self._fail("control completion queue full")
            raise ControlUnavailable("control completion queue full") from exc

    def _read(self):
        try:
            while True:
                event = _recv(self._socket)
                with self._lock:
                    kind = event["type"]
                    if kind == "ready":
                        self._ready.set()
                    elif kind == "snapshot_begin":
                        namespace = event["namespace"]
                        if namespace in self._initial or namespace in self._cache:
                            raise ValueError("duplicate control initial snapshot")
                        self._initial[namespace] = (event["revision"], {})
                    elif kind == "snapshot_end":
                        namespace = event["namespace"]
                        revision, records = self._initial.pop(namespace)
                        if event["revision"] != revision:
                            raise ValueError("control snapshot revision mismatch")
                        self._cache[namespace], self._revisions[namespace] = (
                            records,
                            revision,
                        )
                        sub = self._subscriptions[namespace]
                        sub.changed.set()
                        self._queue_completion(sub.ready, {"ok": True})
                        for watcher in self._watchers.get(namespace, ()):
                            self._initialize_watcher(watcher, namespace)
                        self._cache_changed.notify_all()
                    elif kind == "delta":
                        namespace = event["namespace"]
                        if namespace in self._initial:
                            revision, records = self._initial[namespace]
                            if event["revision"] > revision or event["value"] is None:
                                raise ValueError("invalid initial snapshot record")
                        else:
                            if event["revision"] != self._revisions[namespace] + 1:
                                raise ValueError("control delta gap/reorder")
                            records = self._cache[namespace]
                            self._revisions[namespace] = event["revision"]
                        if event["value"] is None:
                            records.pop(event["key"], None)
                        else:
                            records[event["key"]] = event["value"]
                        if namespace not in self._initial:
                            for sub in [
                                self._subscriptions[namespace],
                                *self._watchers.get(namespace, ()),
                            ]:
                                if sub._queue is not None:
                                    sub._queue.put_nowait(copy.deepcopy(event))
                                sub.changed.set()
                            self._cache_changed.notify_all()
                    elif kind == "result":
                        future = self._pending[event["seq"]]
                        self._queue_completion(future, event)
                        self._pending.pop(event["seq"])
                    else:
                        raise ValueError("unknown control response")
        except Exception as exc:
            self._fail(exc)

    def _enqueue(self, message):
        self._check()
        message = {"seq": self._seq + 1, **message}
        frame = _encode(message)
        try:
            self._outbox.put_nowait(frame)
        except queue.Full as exc:
            self._fail("control outbound queue full")
            raise ControlUnavailable("control outbound queue full") from exc
        self._seq += 1
        return self._seq

    def subscribe(self, namespace, *, queue_events=False):
        with self._lock:
            self._check()
            if namespace in self._subscriptions:
                sub = self._subscriptions[namespace]
                if queue_events and sub._queue is None:
                    raise ValueError(
                        "subscribe with queue_events before first subscription"
                    )
                return sub
            sub = Subscription(namespace, self.capacity, queue_events)
            self._subscriptions[namespace] = sub
            self._enqueue({"type": "subscribe", "namespace": namespace})
            return sub

    def _initialize_watcher(self, watcher, namespace):
        watcher.initial_snapshot = copy.deepcopy(self._cache[namespace])
        watcher.initial_revision = self._revisions[namespace]
        watcher.changed.set()
        self._queue_completion(watcher.ready, {"ok": True})

    def watch(self, namespace):
        """Independent ordered delta queue, atomically based on local mirror.

        Wait ``ready`` once at startup, process ``initial_snapshot`` at
        ``initial_revision``, then consume this watcher's own queue. A second
        watcher cannot steal its events. Call ``close`` when no longer needed.
        """
        with self._lock:
            self._check()
            if sum(len(values) for values in self._watchers.values()) >= 128:
                raise ControlUnavailable("control watcher limit reached")
            self.subscribe(namespace)
            watcher = Subscription(namespace, self.capacity, True)
            self._watchers.setdefault(namespace, []).append(watcher)

            def close_watcher():
                with self._lock:
                    values = self._watchers.get(namespace, [])
                    if watcher in values:
                        values.remove(watcher)
                        if not watcher.ready.done():
                            self._queue_completion(
                                watcher.ready,
                                {
                                    "ok": False,
                                    "error": "control watcher closed before readiness",
                                },
                            )

            watcher._close = close_watcher
            if namespace in self._cache:
                self._initialize_watcher(watcher, namespace)
            return watcher

    def submit(self, service, method, *args, **kwargs):
        with self._lock:
            self._check()
            operation = self._operation + 1
            request = {
                "type": "call",
                "operation_id": operation,
                "service": service,
                "method": method,
                "args": list(args),
                "kwargs": kwargs,
            }
            # Snapshot arguments now, not after the caller mutates/reuses them.
            request = json.loads(_encode(request)[4:])
            future = self._submit_request(request)
            self._operation = operation
            return future

    def _submit_request(self, request):
        self._check()
        if len(self._pending) >= self.max_pending:
            raise ControlUnavailable("control pending request limit reached")
        future = RPCFuture(self, request["operation_id"], request)
        sequence = self._enqueue(request)
        self._pending[sequence] = future
        return future

    def retry(self, future):
        """Explicit same-connection retry; never allocate another operation ID."""
        if not isinstance(future, RPCFuture) or future._owner is not self:
            raise ValueError("retry requires this client's original RPCFuture")
        with self._lock:
            return self._submit_request(future._request)

    def call(self, service, method, *args, **kwargs):
        assert_control_rpc_blocking_allowed()
        future = self.submit(service, method, *args, **kwargs)
        try:
            return future.result(timeout=self.call_timeout)
        except TimeoutError as exc:
            # The operation may already be committed. A timeout is not a
            # negative ownership acknowledgement or permission to reroute.
            self._fail("control call outcome uncertain after timeout")
            raise ControlUnavailable(
                "control call outcome uncertain; retain ownership"
            ) from exc

    def cache_get(self, namespace, key, default=None):
        with self._lock:
            self._check()
            if namespace not in self._cache:
                raise ControlUnavailable("control subscription is not ready")
            return copy.deepcopy(self._cache[namespace].get(key, default))

    def wait_for_record(self, namespace, key, predicate, timeout=None):
        """IO-worker/startup wait on a cached record, with no scan or lost wakeup.

        The predicate must be short and side-effect free. Each waiter has its
        own predicate; another waiter cannot consume its wakeup. Timeout does
        not cancel any ownership operation.
        """
        assert_control_rpc_blocking_allowed()
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cache_changed:
            self.subscribe(namespace)
            while True:
                self._check()
                if namespace in self._cache:
                    value = copy.deepcopy(self._cache[namespace].get(key))
                    if predicate(value):
                        return value
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._cache_changed.wait(remaining)

    def cache_snapshot(self, namespace):
        with self._lock:
            self._check()
            if namespace not in self._cache:
                raise ControlUnavailable("control subscription is not ready")
            return copy.deepcopy(self._cache[namespace])

    def cache_revision(self, namespace):
        with self._lock:
            self._check()
            if namespace not in self._revisions:
                raise ControlUnavailable("control subscription is not ready")
            return self._revisions[namespace]

    def check_health(self):
        with self._lock:
            self._check()

    def wait_ready(self, timeout=10):
        if not self._ready.wait(timeout):
            self._fail("control handshake timeout")
            raise TimeoutError("control handshake timeout")
        with self._lock:
            self._check()

    def close(self):
        self._fail("control client shutdown")
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=2)


_SHARED_CLIENTS = {}
_SHARED_LOCK = threading.Lock()


def get_control_client():
    """Initialize once at process startup, then reuse; never touch a filesystem."""
    endpoint = os.environ.get("SGLANG_AGENTIC_CONTROL_ENDPOINT", "")
    run_id = os.environ.get("SGLANG_AGENTIC_CONTROL_RUN_ID", "")
    token = os.environ.get("SGLANG_AGENTIC_CONTROL_TOKEN", "")
    if not endpoint or not run_id or not token:
        raise ControlUnavailable("control endpoint/run/token are required")
    host, port = endpoint.removeprefix("tcp://").rsplit(":", 1)
    key = (os.getpid(), endpoint, run_id, token)
    with _SHARED_LOCK:
        if key not in _SHARED_CLIENTS:
            client = ControlRPCClient((host, int(port)), run_id=run_id, token=token)
            client.wait_ready()
            _SHARED_CLIENTS[key] = client
        return _SHARED_CLIENTS[key]
