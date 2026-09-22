"""Socket implementation of the existing TP mailbox interface.

Only the transport changes. Native rank-zero phase decisions and their actual
DMA/allocator fences remain with their callers. A retryable snapshot must use
an explicit EventKey; silently treating a bare snapshot as a unique I/O attempt
would let an old callback complete a replacement lease.
"""

import base64
import os
import threading

from sglang.srt.disaggregation.agentic_tp_events import (
    ControlUnavailable,
    EventKey,
    TPEventClient,
)
from sglang.srt.disaggregation.base import KVPoll


_CLIENTS = {}
_CLIENT_LOCK = threading.Lock()


def get_tp_event_client(tp_rank, tp_size):
    """Once-per-process startup connection; all namespaces share this session."""
    endpoint = os.getenv("SGLANG_AGENTIC_TP_EVENT_ENDPOINT", "")
    run_id = os.getenv("SGLANG_AGENTIC_CONTROL_RUN_ID", "")
    token = os.getenv("SGLANG_AGENTIC_CONTROL_TOKEN", "")
    group = os.getenv("SGLANG_AGENTIC_CONTROL_GROUP_ID", "")
    if not all((endpoint, run_id, token, group)):
        raise ControlUnavailable("TP event endpoint/run/token/group are required")
    host, port = endpoint.removeprefix("tcp://").rsplit(":", 1)
    identity = (os.getpid(), endpoint, run_id, token, group, tp_rank, tp_size)
    with _CLIENT_LOCK:
        if identity not in _CLIENTS:
            client = TPEventClient(
                (host, int(port)),
                run_id=run_id,
                token=token,
                group=group,
                rank=tp_rank,
                size=tp_size,
            )
            client.wait_ready()
            _CLIENTS[identity] = client
        return _CLIENTS[identity]


class SocketTPGroupMailbox:
    def __init__(
        self,
        namespace,
        *,
        tp_rank,
        tp_size,
        directory=None,
        group_local_directory=None,
        nnodes=1,
        client=None,
        source_group=None,
    ):
        self.namespace, self.tp_rank, self.tp_size = namespace, tp_rank, tp_size
        self.client = client or get_tp_event_client(tp_rank, tp_size)
        if self.client.rank != tp_rank or self.client.size != tp_size:
            raise ValueError("mailbox rank/size disagrees with TP session")
        self._bindings, self._hidden = {}, set()
        self._prepared = set()
        self._lock = threading.RLock()
        self.source_group = source_group
        if (
            namespace == "p2d-receiver"
            and os.getenv("SGLANG_AGENTIC_MULTINODE_ROLE") == "prefill"
        ):
            self.source_group = source_group or os.getenv(
                "SGLANG_AGENTIC_TP_RECEIPT_SOURCE_GROUP"
            )
            if not self.source_group:
                raise ValueError(
                    "P receiver mailbox requires explicit D receipt source group"
                )
        # Deliberately never create/read the legacy directory arguments.

    def bind_identity(self, raw_key, identity):
        """Bind once for legacy callsites; retries must pass explicit EventKey.

        A raw key cannot later change meaning, even after clear. This prevents
        a delayed callback holding that raw key from being relabeled as a newer
        attempt. New code should simply retain/pass the immutable EventKey.
        """
        if not isinstance(identity, EventKey):
            raise ValueError("explicit snapshot/attempt EventKey required")
        with self._lock:
            if raw_key in self._bindings and self._bindings[raw_key] != identity:
                raise ValueError("cannot rebind an existing TP callback identity")
            self._bindings[raw_key] = identity
        return identity

    def _key(self, key):
        if isinstance(key, EventKey):
            return key
        with self._lock:
            identity = self._bindings.get(key)
        if identity is not None:
            return identity
        if self.namespace.startswith("p2d-") and isinstance(key, str) and "@" in key:
            rid, room = key.rsplit("@", 1)
            if rid and room:
                return EventKey(key, "bootstrap-room:" + room)
        raise ValueError(
            f"{self.namespace} requires an explicit request-generation/attempt EventKey"
        )

    def publish_local(self, key, status):
        identity = self._key(key)
        self._hidden.discard(identity)
        if self.source_group:
            raise ValueError("receipt observer cannot publish producer shard reports")
        if self.namespace.startswith("p2d-"):
            self.client.report_transfer(
                self.namespace,
                identity,
                int(status),
                failed=int(KVPoll.Failed),
                success=int(KVPoll.Success),
                transferring=int(KVPoll.Transferring),
            )
        else:
            self.client.report_state(self.namespace, identity, int(status))

    def publish_prepare(self, request, payload, manifest):
        """Rank0 sends one metadata-only intent; this grants no pages or DMA."""
        identity = self._key(request.snapshot_id)
        with self._lock:
            if identity in self._prepared:
                return
            self.client.publish_command(
                self.namespace + ":prepare",
                identity,
                {
                    "op": "PREPARE",
                    "request_id": request.request_id,
                    "generation": request.generation,
                    "payload": payload,
                    "manifest": base64.b64encode(manifest.to_bytes()).decode("ascii"),
                },
                command_id=1,
            )
            self._prepared.add(identity)

    def drain_prepares(self):
        return self.client.drain_commands(self.namespace + ":prepare")

    def ack_prepare(self, identity, command_id):
        # Only acknowledges installing the metadata intent in the local
        # admission queue. Existing Direct report/fence still guards DMA.
        self.client.ack_command(self.namespace + ":prepare", identity, command_id)

    def publish_local_progress(self, key, status):
        if self.source_group:
            raise ValueError("receipt observer cannot publish producer shard reports")
        identity = self._key(key)
        self._hidden.discard(identity)
        self.client.report(self.namespace, identity, int(status))

    def local_status(self, key, rank=None):
        identity = self._key(key)
        if self.source_group:
            raise ValueError("receipt observer cannot inspect producer shard reports")
        if (rank is None or rank == self.tp_rank) and identity in self._hidden:
            return None
        return self.client.local_status(self.namespace, identity, rank)

    def group_status(self, key):
        if self.source_group:
            raise ValueError("receipt observer cannot reduce producer shard reports")
        return self.client.group_status(self.namespace, self._key(key))

    def any_negative_report(self, key):
        if self.source_group:
            raise ValueError("receipt observer cannot reduce producer shard reports")
        return self.client.any_negative_report(self.namespace, self._key(key))

    def transfer_group_status(self, key):
        if self.source_group:
            raise ValueError("receipt observer cannot reduce producer shard reports")
        return self.client.transfer_group_status(self.namespace, self._key(key))

    def publish_receipt(self, key, status):
        if self.source_group:
            raise ValueError("receipt observer cannot publish producer decision")
        self.client.publish_receipt(self.namespace, self._key(key), int(status))

    def receipt(self, key):
        identity = self._key(key)
        if self.source_group:
            return self.client.observed_receipt(
                self.source_group, self.namespace, identity
            )
        return self.client.receipt(self.namespace, identity)

    def publish_local_rollback_complete(self, key):
        if self.source_group:
            raise ValueError("receipt observer cannot report producer rollback")
        self.client.report(self.namespace + ":rollback", self._key(key), 1)

    def rollback_group_complete(self, key):
        return (
            self.client.group_status(self.namespace + ":rollback", self._key(key)) == 1
        )

    def clear_local_rollback(self, key):
        # Rollback is monotonic for this exact attempt. No physical operation
        # is undone by forgetting a local consumer's reference.
        self._key(key)

    def clear_group_rollback(self, key):
        if self.tp_rank != 0:
            raise ValueError("only TP rank zero may clear rollback")
        self.client.clear(self.namespace + ":rollback", self._key(key))

    def clear_local(self, key):
        self._hidden.add(self._key(key))

    def clear_group(self, key):
        identity = self._key(key)
        if self.source_group:
            self.client.clear_observed(self.source_group, self.namespace, identity)
        else:
            if self.tp_rank != 0:
                raise ValueError("only TP rank zero may clear group")
            self.client.clear(self.namespace, identity)
