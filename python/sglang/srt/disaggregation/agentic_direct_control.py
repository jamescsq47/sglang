"""Poll existing lifecycle RPCs without blocking the Direct transport worker.

No executor or transport callback mutates lifecycle state here. Each operation
retains its original RPCFuture and advances only when the Direct worker polls
it. Cancellation stops new ownership transitions; already-submitted writes
must settle before the original all-rank abort can retire the attempt.
"""

from dataclasses import replace
import os
import time

from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    SnapshotLifecycleError, SnapshotNotReadyError, SnapshotState,
)
from sglang.srt.disaggregation.agentic_tp_events import ControlUnavailable


class DirectControlCall:
    def __init__(self, identity, operation, *args):
        self.identity = identity
        self.cancelled = False
        self.started_at = time.monotonic()
        self.finished_at = None
        self.future = None
        self.done = False
        self.result = None
        self.error = None
        self.uncertain = False
        self.committed_manifest = None
        self._generator = operation(self, *args)

    @property
    def settled(self):
        return (self.done and not self.uncertain
                and not (self.error is not None and self.committed_manifest is not None))

    def poll(self, *, cancel=False):
        self.cancelled |= cancel
        if self.done:
            if self.error is not None:
                raise self.error
            return True, self.result
        # Bounded progress; no result() is allowed on an unfinished future.
        for _ in range(8):
            value = None
            if self.future is not None:
                if not self.future.done():
                    return False, None
                try:
                    value = self.future.result()
                except Exception as exc:
                    self.error = exc
                    self.uncertain = isinstance(exc, ControlUnavailable)
                    self.done = True
                    self.finished_at = time.monotonic()
                    raise
                self.future = None
            try:
                self.future = self._generator.send(value)
            except StopIteration as stop:
                self.done = True
                self.result = stop.value
                self.finished_at = time.monotonic()
                return True, self.result
            except Exception as exc:
                self.error = exc
                self.uncertain = isinstance(exc, ControlUnavailable)
                self.done = True
                self.finished_at = time.monotonic()
                raise
            if self.future is None:
                return False, None  # A pushed mirror has not arrived yet.
        return False, None


def _submit(records, method, key, *args):
    records.check()
    return records.client.submit("records", method, records.namespace, str(key), *args)


def _fences():
    from sglang.srt.disaggregation.agentic_lifecycle_control import lifecycle_records
    return lifecycle_records()


def _same_claim(manifest, claim_id, room):
    return (manifest is not None and manifest.claim_id == claim_id
            and manifest.direct_room == room)


def claim_direct(call, store, request, claim_id, room):
    """The existing local fence -> claim key -> DIRECT_LOADING sequence."""
    if call.cancelled:
        return None
    fences = _fences()
    path = store._local_claim_path(request)
    acquired, created = yield _submit(fences, "claim", path, claim_id)
    if call.cancelled:
        return None
    if not acquired:
        raise SnapshotNotReadyError("Direct lifecycle fence belongs to another owner")
    if not created:
        # Join the same logical TP claim through its pushed manifest. No
        # sleep, repeated RPC, or independent follower routing decision.
        while True:
            if call.cancelled:
                return None
            current = store.load(request, require_ready=False)
            if (current is not None and current.state is SnapshotState.DIRECT_LOADING
                    and _same_claim(current, claim_id, room)):
                return current
            if current is not None and (
                    current.direct_room != room or current.state not in {
                        SnapshotState.DIRECT_READY, SnapshotState.DIRECT_LOADING}):
                raise SnapshotNotReadyError("Direct offer changed while joining claim")
            yield None
    records = store.store.records
    encoded_claim = store.store._encode(f"direct:{claim_id}".encode())
    result = yield _submit(records, "put", request.claim_key, encoded_claim)
    if call.cancelled:
        return None
    current = store.load(request, require_ready=False)
    if result != 0:
        if (store.store.get(request.claim_key) == f"direct:{claim_id}".encode()
                and current is not None and current.state is SnapshotState.DIRECT_LOADING
                and _same_claim(current, claim_id, room)):
            return current
        raise SnapshotNotReadyError("Direct claim key belongs to another owner")
    if (current is not None and current.state is SnapshotState.DIRECT_LOADING
            and _same_claim(current, claim_id, room)):
        return current
    if (current is None or current.state is not SnapshotState.DIRECT_READY
            or current.direct_room != room):
        raise SnapshotNotReadyError("Direct offer changed before claim commit")
    claimed = replace(current, claim_id=claim_id).transition(SnapshotState.DIRECT_LOADING)
    yield _submit(records, "upsert_if_owner", claimed.manifest_key,
                  store.store._encode(claimed.to_bytes()), fences.namespace, path, claim_id)
    return claimed


def complete_direct(call, store, manifest, claim_id):
    current = store.load(manifest.request, require_ready=False)
    if call.cancelled:
        return current
    if current is not None and current.state in {SnapshotState.P_RECEIVED, SnapshotState.CONSUMED}:
        if current.direct_room != manifest.direct_room:
            raise SnapshotLifecycleError("Direct completion room changed")
        return current
    if (current is None or current.state is not SnapshotState.DIRECT_LOADING
            or not _same_claim(current, claim_id, manifest.direct_room)):
        raise SnapshotLifecycleError("Direct group claim changed before completion")
    store._require_local_claim_owner(manifest.request, claim_id)
    received = current.transition(SnapshotState.P_RECEIVED)
    fences = _fences()
    yield _submit(store.store.records, "upsert_if_owner", received.manifest_key,
                  store.store._encode(received.to_bytes()), fences.namespace,
                  store._local_claim_path(manifest.request), claim_id)
    return received


def commit_bound(call, store, manifest, claim_id):
    if call.cancelled:
        return store.load(manifest.request, require_ready=False)
    if manifest.state is not SnapshotState.P_RECEIVED or manifest.claim_id != claim_id:
        raise SnapshotLifecycleError("Direct bind commit lost its claim")
    store._require_local_claim_owner(manifest.request, claim_id)
    terminal = replace(manifest.transition(SnapshotState.CONSUMED), terminal_at=time.time())
    fences = _fences()
    records = store.store.records
    yield _submit(records, "upsert_if_owner", terminal.manifest_key,
                  store.store._encode(terminal.to_bytes()), fences.namespace,
                  store._local_claim_path(manifest.request), claim_id)
    call.committed_manifest = terminal
    # CONSUMED is irreversible. Even if cancel arrived while this CAS was
    # pending, finish only its claim cleanup; never transfer ownership back.
    yield _submit(records, "remove", terminal.request.claim_key)
    yield _submit(fences, "remove", store._local_claim_path(manifest.request), claim_id)
    return terminal


def publish_route(call, marker_store, request, domain, tokens):
    if call.cancelled:
        return None
    from sglang.srt.disaggregation.agentic_early_claim import _VERSION
    payload = {
        "version": _VERSION, "kind": "route", "snapshot_id": request.snapshot_id,
        "request_id": request.request_id, "generation": request.generation,
        "route": "direct_complete", "prefill_domain": int(domain),
        "arena_numa_node": None, "snapshot_tokens": int(tokens),
        "published_at": time.time(), "publisher_pid": os.getpid(),
    }
    yield _submit(marker_store._records, "upsert",
                  marker_store._record_key(marker_store.route_path(request)), payload)
    return payload


def release_unstarted_claim(call, store, request, claim_id):
    """After ALL rollback ACKs, remove an aborted pre-manifest claim fence."""
    fences = _fences()
    path = store._local_claim_path(request)
    if fences.get(path) != claim_id:
        return
    records = store.store.records
    expected = store.store._encode(f"direct:{claim_id}".encode())
    yield _submit(records, "remove", request.claim_key, expected)
    yield _submit(fences, "remove", path, claim_id)
