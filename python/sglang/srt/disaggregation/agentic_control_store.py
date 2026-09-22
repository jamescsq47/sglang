"""Small record adapters for the socket control plane, not a KV data store.

The server owns metadata and exclusion keys in DRAM. Client reads use the
ordered push mirror; writes that change ownership require an acknowledged RPC
on an IO worker. Advisory notifications can be queued without blocking Forward.
No filesystem path is opened: legacy directory strings are only scope labels.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import os
import threading
from collections import deque


def control_enabled():
    return bool(os.environ.get("SGLANG_AGENTIC_CONTROL_ENDPOINT"))


def record_namespace(kind, scope="default"):
    digest = hashlib.sha256(str(scope).encode()).hexdigest()[:24]
    return f"records:{kind}:{digest}"


class MemoryControlStore:
    """Broker-owned atomic operations; publisher must only enqueue messages."""

    def __init__(self, publish, *, max_records=1_000_000):
        self._publish = publish
        self._lock = threading.RLock()
        self._data = {}
        self._size = 0
        self._max_records = max_records
        self._ready_acks = {}
        self._retired_ready = set()

    def _set(self, namespace, key, value):
        records = self._data.setdefault(namespace, {})
        if key not in records:
            if self._size >= self._max_records:
                raise RuntimeError("control metadata capacity exhausted; retaining ownership")
        value = copy.deepcopy(value)
        # Broker validates/serializes before committing a publication. Failed
        # validation must not leave a hidden claim in the authoritative map.
        self._publish(namespace, key, value)
        if key not in records:
            self._size += 1
        records[key] = value

    def upsert(self, namespace, key, value):
        if value is None:
            raise ValueError("None is reserved for deletion notifications")
        with self._lock:
            if (str(namespace), str(key)) in self._retired_ready:
                raise RuntimeError("cannot republish an admitted P-ready generation")
            self._set(str(namespace), str(key), value)
        return 0

    def put(self, namespace, key, value):
        if value is None:
            raise ValueError("None is reserved for deletion notifications")
        with self._lock:
            if str(key) in self._data.get(str(namespace), {}):
                return -1
            self._set(str(namespace), str(key), value)
        return 0

    def claim(self, namespace, key, owner):
        if not owner:
            raise ValueError("claim owner is required")
        with self._lock:
            current = self._data.get(str(namespace), {}).get(str(key))
            if current is not None:
                return [current == owner, False]
            self._set(str(namespace), str(key), owner)
            return [True, True]

    def upsert_if_owner(self, namespace, key, value, claim_namespace, claim_key, owner):
        """Commit lifecycle metadata under the authoritative generation fence.

        A client's push mirror is advisory; it cannot authorize a write after
        another owner acquired the generation. Existing lifecycle code retains
        the exclusive owner through the acknowledged transition.
        """
        if not owner or value is None:
            raise ValueError("claim owner and metadata are required")
        with self._lock:
            current = self._data.get(str(claim_namespace), {}).get(str(claim_key))
            if current != owner:
                raise RuntimeError("snapshot lifecycle owner changed")
            self._set(str(namespace), str(key), value)
        return 0

    def remove(self, namespace, key, expected=None):
        with self._lock:
            records = self._data.get(str(namespace), {})
            if str(key) not in records:
                return -1
            if expected is not None and records[str(key)] != expected:
                return -1
            self._publish(str(namespace), str(key), None)
            del records[str(key)]
            self._size -= 1
        return 0

    def methods(self):
        return {name: getattr(self, name) for name in (
            "upsert", "put", "claim", "remove", "admit_ready", "upsert_if_owner",
        )}

    def admit_ready(self, namespace, key, group, rank, size):
        """Retire a notification only after every selected D shard admitted it.

        This is an admission latch, not a DMA/ownership fence. Source KV is
        still governed by the existing transport-completion protocol.
        """
        if (type(rank) is not int or type(size) is not int
                or not 0 <= rank < size <= 256 or not group):
            raise ValueError("invalid P-ready admission group")
        identity = (str(namespace), str(key))
        with self._lock:
            if identity in self._retired_ready:
                return True
            if str(key) not in self._data.get(str(namespace), {}):
                return False
            entry = self._ready_acks.setdefault(identity, (str(group), size, set()))
            if entry[:2] != (str(group), size):
                raise RuntimeError("P-ready admitted by two different D groups")
            entry[2].add(rank)
            if len(entry[2]) != size:
                return False
            if len(self._retired_ready) >= self._max_records:
                raise RuntimeError("P-ready tombstone capacity exhausted")
            self.remove(namespace, key)
            self._retired_ready.add(identity)
            del self._ready_acks[identity]
            return True


class ControlKV:
    def __init__(self, namespace, *, client=None):
        if client is None:
            from sglang.srt.disaggregation.agentic_control_rpc import get_control_client
            client = get_control_client()
        self.client = client
        self.namespace = str(namespace)
        self.subscription = client.subscribe(self.namespace)
        # One-time initialization, never a recurring synchronization. Engine
        # owners construct adapters at startup, before starting Forward.
        if not self.subscription.ready.done():
            from sglang.srt.disaggregation.agentic_control_rpc import assert_control_rpc_blocking_allowed
            assert_control_rpc_blocking_allowed()
        self.subscription.ready.result(timeout=30)
        self._pending = deque()
        self._lock = threading.Lock()

    def check(self):
        with self._lock:
            while self._pending and self._pending[0].done():
                self._pending[0].result()
                self._pending.popleft()

    def get(self, key):
        self.check()
        return self.client.cache_get(self.namespace, str(key))

    def snapshot(self):
        self.check()
        return self.client.cache_snapshot(self.namespace)

    def call(self, method, key, *args):
        self.check()
        return self.client.call("records", method, self.namespace, str(key), *args)

    def notify(self, method, key, *args):
        """Queue only non-owning notifications; check every eventual reply."""
        self.check()
        with self._lock:
            if len(self._pending) >= 4096:
                raise RuntimeError("control notification backlog full")
            future = self.client.submit("records", method, self.namespace, str(key), *args)
            self._pending.append(future)
        return future


_stores = {}
_stores_lock = threading.Lock()


def control_kv(kind, scope="default"):
    namespace = record_namespace(kind, scope)
    identity = (os.getpid(), namespace)
    with _stores_lock:
        store = _stores.get(identity)
    if store is not None:
        return store
    # Subscription setup may wait for the initial mirror. Never hold the
    # global cache lock across it: Forward's already-initialized namespace
    # must remain usable while an unrelated IO worker starts a subscription.
    candidate = ControlKV(namespace)
    with _stores_lock:
        return _stores.setdefault(identity, candidate)


def read_pressure_record(path):
    """Advisory load snapshot; never an ownership or capacity grant."""
    import json
    if control_enabled():
        value = control_kv("prefill-pressure", str(path)).get("latest")
        if value is None:
            raise FileNotFoundError(str(path))
        return value
    with open(path, encoding="utf-8") as source:
        return json.load(source)


class BrokerRawStore:
    """The existing snapshot metadata byte API over an in-memory record service."""

    def __init__(self, directory):
        self.directory = str(directory)
        self.records = control_kv("snapshot-metadata", self.directory)

    @staticmethod
    def _encode(value):
        return base64.b64encode(bytes(value)).decode("ascii")

    def put(self, key, value):
        return self.records.call("put", key, self._encode(value))

    def upsert(self, key, value):
        return self.records.call("upsert", key, self._encode(value))

    def get(self, key):
        value = self.records.get(key)
        return b"" if value is None else base64.b64decode(value, validate=True)

    def is_exist(self, key):
        return int(self.records.get(key) is not None)

    def batch_is_exist(self, keys):
        return [self.is_exist(key) for key in keys]

    def remove(self, key, force=False):
        return self.records.call("remove", key)

    def batch_remove(self, keys, force=False):
        return [self.remove(key, force=force) for key in keys]


class ReadySignals:
    """Non-owning P-ready/HTTP admission signals with pushed local reads."""

    def __init__(self, scope):
        self.records = control_kv("p-ready", scope)

    def get(self, room, kind="ready"):
        return self.records.get(f"{room}.{kind}")

    def publish(self, room, metadata, kind="ready"):
        return self.records.notify("upsert", f"{room}.{kind}", metadata)

    def remove(self, room, kind="ready"):
        return self.records.notify("remove", f"{room}.{kind}")

    def admit(self, room, *, group, rank, size):
        return self.records.notify("admit_ready", f"{room}.ready", group, rank, size)

    def snapshot(self):
        return self.records.snapshot()
