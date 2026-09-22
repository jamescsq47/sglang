"""Event-backed Host ledger adapter; no runtime file discovery or file locks.

The server retains the original lifecycle transitions in a pure-memory core.
Clients read pushed local mirrors. Mutations return only committed RPC results;
scheduler callers use the resumable, nonblocking transition scope below.
"""

from copy import deepcopy
from functools import partial
import json
import os
import queue
import threading


HOST_METHODS = frozenset({
    "offer", "claim", "claim_rank", "prepare_p2d_write_rank", "claim_p2d_write_rank",
    "reject_unclaimed_offer", "abort_unsubmitted_p2d", "arbitrate_p2d_release",
    "arbitrate_p2d_native", "publish_rank_grant", "publish_grants", "complete_host_write",
    "complete_host_load_rank", "request_host_load_failure", "mark_host_load_rank_drained",
    "complete_d2p_host_load_rank", "complete_host_bind_rank", "request_d2p_retry",
    "complete_d2p_retry_rank", "begin_host_load_rank", "assign_d2p_recovery_domain",
    "claim_d2p_recovery_rank", "attach_d2p_recovery_lease_rank",
    "mark_d2p_recovery_phase_rank", "cancel_d2p_recovery_rank", "prepare_tp_host_load_rank",
    "complete_p2d_host_write_rank", "mark_writer_drained", "mark_writer_rank_drained",
    "fail_host_write", "mark_sent", "ack_chunk", "mark_host_ready", "begin_host_eviction",
    "complete_host_eviction_rank", "transition", "set_source_host_pressure_rank",
    "complete_source_host_release_rank", "prune", "apply_recovery_event",
    "apply_p2d_recovery_event",
})


def _names(direction):
    if direction not in {"d2p", "p2d"}:
        raise ValueError("Host control direction must be d2p or p2d")
    return "host-" + direction, "host/" + direction


class HostLedgerService:
    """Register one authoritative direction with the authenticated broker."""

    def __init__(self, server, direction, *, max_generations=1_000_000):
        from sglang.srt.disaggregation.agentic_host_control import InMemoryHostStagingLedger
        self.server = server
        self.direction = direction
        self.service, self.namespace = _names(direction)
        self.ledger = InMemoryHostStagingLedger(max_generations=max_generations)
        server.register_service(self.service, {
            name: partial(self._invoke, name) for name in HOST_METHODS
        })

    def _invoke(self, method, *args, **kwargs):
        if method not in HOST_METHODS:
            raise ValueError("unsupported Host control method")
        try:
            p2d_event = {
                "begin_host_load_rank": "begin", "complete_host_load_rank": "loaded",
                "request_host_load_failure": "failed", "mark_host_load_rank_drained": "drained",
            }.get(method) if self.direction == "p2d" else None
            if p2d_event is not None:
                required = {"tp_rank", "tp_size", "decode_domain", "attempt_id"}
                if not required <= kwargs.keys():
                    raise ValueError("P2D completion requires exact destination/attempt identity")
                return self.ledger.apply_p2d_recovery_event(
                    *args, event=p2d_event, **kwargs,
                )
            if method == "transition":
                from sglang.srt.disaggregation.agentic_host_staging import HostStageState
                if len(args) > 1:
                    args = (args[0], HostStageState(args[1]), *args[2:])
                else:
                    kwargs = dict(kwargs, state=HostStageState(kwargs["state"]))
            event = {
                "complete_d2p_host_load_rank": "loaded",
                "complete_host_bind_rank": "bound",
            }.get(method)
            if method == "mark_d2p_recovery_phase_rank" and kwargs.get("phase") == "handed":
                kwargs = dict(kwargs)
                kwargs.pop("phase")
                event = "handed"
            if event is not None:
                # Never infer identity from the latest mirror: a delayed old
                # DMA receipt must carry the attempt that actually performed
                # the copy, not accidentally acknowledge a successor.
                required = {"claim_id", "lease_id", "remote_read_epoch", "tp_rank", "tp_size"}
                if not required <= kwargs.keys():
                    raise ValueError("Host completion requires exact claim/lease/epoch identity")
                return self.ledger.apply_recovery_event(*args, event=event, **kwargs)
            return getattr(self.ledger, method)(*args, **kwargs)
        finally:
            # publish only enqueues into the broker event stream. The broker
            # applies these pushes before completing the invoking RPC reply.
            for change in self.ledger.drain_changes():
                self.server.publish(self.namespace, change["snapshot_id"], change["entry"])


class RemoteHostStagingLedger:
    """File-ledger API with local reads and explicit asynchronous mutations."""

    is_event_control = True

    def __init__(self, client, direction, *, startup_timeout=30.0):
        self.client = client
        self.service, self.namespace = _names(direction)
        self._subscription = client.subscribe(self.namespace)
        # Constructor is startup-only. No runtime read ever waits for a sync.
        self._subscription.ready.result(timeout=startup_timeout)
        self._pending_lock = threading.Lock()
        self._pending = {}

    @classmethod
    def from_environment(cls, path):
        from sglang.srt.disaggregation.agentic_control_rpc import get_control_client
        paths = {
            "d2p": os.getenv("SGLANG_AGENTIC_KV_STAGING_LEDGER_PATH"),
            "p2d": os.getenv("SGLANG_AGENTIC_KV_P2D_STAGING_LEDGER_PATH"),
        }
        directions = [key for key, value in paths.items() if value and os.path.normpath(value) == os.path.normpath(path)]
        if len(directions) != 1:
            raise ValueError("Host RPC ledger path must identify exactly one configured direction")
        # The transport factory validates endpoint, token and run identity.
        return cls(get_control_client(), directions[0])

    def get(self, snapshot_id):
        return deepcopy(self.client.cache_get(self.namespace, str(snapshot_id)))

    def snapshot_entries(self, *, force_refresh=False):
        return deepcopy(self.client.cache_snapshot(self.namespace))

    def list_state(self, *states):
        wanted = {getattr(state, "value", str(state)) for state in states}
        values = [value for value in self.snapshot_entries().values() if value.get("state") in wanted]
        values.sort(key=lambda item: (item.get("created_at", 0.0), item["snapshot_id"]))
        return values

    def create_watcher(self):
        return HostLedgerWatcher(self.client, self.namespace)

    def submit(self, method, *args, **kwargs):
        if method not in HOST_METHODS:
            raise ValueError("unsupported Host control method")
        return self.client.submit(self.service, method, *args, **kwargs)

    def call(self, method, *args, **kwargs):
        if method not in HOST_METHODS:
            raise ValueError("unsupported Host control method")
        return self.client.call(self.service, method, *args, **kwargs)

    def poll_call(self, method, *args, wakeup=None, **kwargs):
        """Submit once, then consume only a committed result without waiting.

        ``(False, None)`` means pending, never failed or successful ownership.
        Callers place this at explicit safe continuation points; no allocator,
        Radix, queue or DMA operation is replayed by this adapter.
        """
        key = json.dumps([method, args, kwargs], sort_keys=True, separators=(",", ":"))
        with self._pending_lock:
            future = self._pending.get(key)
            if future is None:
                future = self.submit(method, *args, **kwargs)
                self._pending[key] = future
                if wakeup is not None:
                    future.add_done_callback(lambda _future: wakeup.set())
        if not future.done():
            return False, None
        # Ambiguous connection/commit failure remains attached to this exact
        # operation. Never turn a later scheduler tick into a new acquisition.
        result = future.result()
        with self._pending_lock:
            if self._pending.get(key) is future:
                self._pending.pop(key, None)
        return True, result

    def __getattr__(self, name):
        if name in HOST_METHODS:
            return partial(self.call, name)
        raise AttributeError(name)


class HostLedgerWatcher:
    """One independent pushed-event consumer, with no periodic file scan."""

    def __init__(self, client, namespace):
        self.client = client
        self._watch = client.watch(namespace)
        self._watch.ready.result(timeout=30.0)
        self.initial_entries = deepcopy(self._watch.initial_snapshot)
        self.healthy = True

    def wait_events(self):
        while self.healthy:
            self.client.check_health()
            self._watch.changed.wait()
            self._watch.changed.clear()
            self.client.check_health()
            events = []
            while True:
                try:
                    delta = self._watch.get_nowait()
                except queue.Empty:
                    break
                entry = deepcopy(delta["value"])
                events.append({
                    "snapshot_id": delta["key"], "entry": entry,
                    "revision": int(entry["_event_revision"]) if entry is not None else int(delta["revision"]),
                })
            if events:
                return events
        return []

    def close(self):
        self.healthy = False
        self._watch.close()
        self._watch.changed.set()
