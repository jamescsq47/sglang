"""Authoritative, process-local Host lifecycle state for a control broker.

This is a storage backend, not another transfer state machine. The public
request-generation transitions are inherited from SharedHostStagingLedger.
No engine switches to this backend merely by importing it. The broker must
serialize commands/replies and deduplicate transport requests; this class
neither performs network I/O nor treats a timeout as a physical DMA fence.
"""

from collections import OrderedDict
from copy import deepcopy
import threading
import time

from sglang.srt.disaggregation.agentic_host_staging import (
    HostStageState,
    SharedHostStagingLedger,
    _TERMINAL_STATES,
)


class InMemoryHostStagingLedger(SharedHostStagingLedger):
    """Copy-on-write lifecycle transactions with event-driven notifications.

    A single broker owns this instance. Short in-process transactions commit
    detached values, then wake the broker's event distributor. There are no
    user callbacks, sockets, files, CUDA calls or external locks in commit.
    Source KV can still only be released by the existing all-shard fences.
    Broker death is fail-closed: a new empty instance is not a recovery of an
    old live run, and must never authorize reuse of that run's allocations.
    """

    def __init__(self, *, max_generations=1_000_000):
        # Deliberately do not call the file-backed superclass constructor.
        if not isinstance(max_generations, int) or max_generations < 1:
            raise ValueError("max_generations must be a positive integer")
        # The broker's guarded event facade validates an attempt and then
        # invokes an inherited pure-state transition under this same lock.
        self._condition = threading.Condition(threading.RLock())
        self._entries = {}
        # A generation can never be resurrected by a delayed producer offer.
        # Keep compact tombstones for the entire broker/run lifetime; expiry
        # would reintroduce ABA. At the explicit bound fail new admission
        # closed, but continue all existing transfer/cancel/release progress.
        self._retired = set()
        # Eviction is also a notification to a later consumer, not just a
        # physical release. Keep its small terminal receipt for this run so a
        # delayed tool/Router/P cannot mistake pruned Host data for pending IO.
        # These IDs are already counted in the bounded _retired generation set.
        self._retired_receipts = {}
        self._max_generations = max_generations
        self._relays = {}
        self._sequence = 0
        # One latest change per generation, not a growing history of polls.
        # Only the broker distributor drains this queue; it fans out changes
        # to its subscribers. A newly connected peer takes a fresh snapshot.
        self._changes = OrderedDict()

    def _check_new_generations_locked(self, snapshot_ids):
        if self._retired.intersection(snapshot_ids):
            raise ValueError("retired request-generation cannot be offered again")
        if len(self._entries) + len(self._retired) + len(snapshot_ids) > self._max_generations:
            raise RuntimeError("Host control generation capacity exhausted; new admission is closed")

    def _record_changes_locked(self, changes, removed_revisions=None):
        for snapshot_id, value in changes.items():
            self._sequence += 1
            self._changes[snapshot_id] = {
                "version": self.VERSION,
                "sequence": self._sequence,
                "snapshot_id": snapshot_id,
                "revision": int(value["_event_revision"]) if value is not None else int(
                    (removed_revisions or {}).get(snapshot_id, 0)
                ),
                "entry": value,
            }
            self._changes.move_to_end(snapshot_id)
        if changes:
            self._condition.notify_all()

    def _mutate(self, callback, *, event_snapshot_id=None):
        if event_snapshot_id is None:
            raise ValueError("request-generation mutation requires snapshot id")
        snapshot_id = str(event_snapshot_id)
        with self._condition:
            previous = self._entries.get(snapshot_id)
            working = {} if previous is None else {snapshot_id: deepcopy(previous)}
            result, changed = callback(working)
            # Copy everything that can fail before publishing anything. In
            # particular an exception/rejected mutation cannot leak nested
            # dict/list edits into the authoritative record.
            result = deepcopy(result)
            if changed:
                if set(working) - {snapshot_id}:
                    raise ValueError("single-generation transaction changed another snapshot")
                current = deepcopy(working.get(snapshot_id))
                if current is not None:
                    if previous is None:
                        self._check_new_generations_locked({snapshot_id})
                    current["_event_revision"] = int(
                        (previous or {}).get("_event_revision", 0)
                    ) + 1
                    self._entries[snapshot_id] = current
                else:
                    self._entries.pop(snapshot_id, None)
                    if previous is not None:
                        self._retired.add(snapshot_id)
                self._record_changes_locked({snapshot_id: current}, {
                    snapshot_id: int((previous or {}).get("_event_revision", 0)) + 1,
                })
            return result

    def _mutate_document(self, callback, *, event_snapshot_id=None):
        """Compatibility transaction for the small in-memory registry.

        Normal request transitions use _mutate and copy only one generation.
        Legacy relay execution is intentionally unsupported by this backend.
        """
        if callable(event_snapshot_id):
            raise ValueError("dynamic relay claims are not supported")
        sid = None if event_snapshot_id is None else str(event_snapshot_id)
        with self._condition:
            selected = self._entries if sid is None else (
                {} if sid not in self._entries else {sid: self._entries[sid]}
            )
            working = deepcopy({
                "version": self.VERSION, "entries": selected, "relays": self._relays,
            })
            result, changed = callback(working)
            result = deepcopy(result)
            if not changed:
                return result
            entries = deepcopy(working["entries"])
            relays = deepcopy(working["relays"])
            if sid is not None and set(entries) - {sid}:
                raise ValueError("single-generation transaction changed another snapshot")
            self._check_new_generations_locked(set(entries) - set(selected))
            changed_entries = {}
            removed_revisions = {}
            for key in set(selected) | set(entries):
                value = entries.get(key)
                previous = selected.get(key)
                if value != previous:
                    if value is not None:
                        value["_event_revision"] = int(
                            (previous or {}).get("_event_revision", 0)
                        ) + 1
                    changed_entries[key] = value
                    if value is None:
                        removed_revisions[key] = int(
                            (previous or {}).get("_event_revision", 0)
                        ) + 1
            self._relays = relays
            self._retired.update(removed_revisions)
            if sid is None:
                self._entries = entries
            elif sid in entries:
                self._entries[sid] = entries[sid]
            else:
                self._entries.pop(sid, None)
            self._record_changes_locked(changed_entries, removed_revisions)
            return result

    def get(self, snapshot_id):
        with self._condition:
            sid = str(snapshot_id)
            return deepcopy(self._entries.get(sid, self._retired_receipts.get(sid)))

    def snapshot_entries(self, *, force_refresh=False):
        with self._condition:
            return deepcopy({**self._retired_receipts, **self._entries})

    def apply_recovery_event(
        self, snapshot_id, owner, *, tp_rank, tp_size, claim_id, lease_id,
        remote_read_epoch, event,
    ):
        """Apply one exact-attempt completion receipt, never a stale report.

        Network handlers use this facade instead of directly exposing the
        inherited completion methods, whose single-node call sites identify
        only the snapshot/owner. Validation and the existing transition are
        one in-process transaction; no network/allocator/CUDA action occurs
        while holding this lock. Physical completion must already be known
        to the reporting rank; this method cannot manufacture a DMA fence.
        """
        if event not in {"loaded", "bound", "handed"}:
            raise ValueError("unknown Host recovery event")
        if (
            not all(type(value) is int for value in (tp_rank, tp_size, lease_id, remote_read_epoch))
            or tp_size < 1 or not 0 <= tp_rank < tp_size
            or remote_read_epoch < 0 or not claim_id
        ):
            return False
        snapshot_id = str(snapshot_id)
        with self._condition:
            current = self._entries.get(snapshot_id)
            if (
                current is None
                or current.get("recovery_owner") != owner
                or current.get("recovery_claim_id") != claim_id
                or int(current.get("tp_size", 1)) != tp_size
                or int(current.get("remote_read_epoch", 0)) != remote_read_epoch
            ):
                return False
            claim = current.get("recovery_claims", {}).get(str(tp_rank))
            if (
                claim is None or claim.get("claim_id") != claim_id
                or claim.get("lease_id") != lease_id
                or claim.get("phase") not in {"io_inflight", "handed"}
            ):
                return False
            # A delayed duplicate after handoff is safe only if that exact
            # shard already supplied the corresponding completion receipt.
            if event == "loaded":
                if claim["phase"] == "handed" and tp_rank not in current.get("loader_acks", []):
                    return False
                return self.complete_d2p_host_load_rank(
                    snapshot_id, owner, tp_rank=tp_rank, tp_size=tp_size,
                )
            if tp_rank not in current.get("loader_acks", []):
                return False
            if event == "bound":
                if claim["phase"] == "handed" and tp_rank not in current.get("binder_acks", []):
                    return False
                return self.complete_host_bind_rank(
                    snapshot_id, owner, tp_rank=tp_rank, tp_size=tp_size,
                )
            if (
                current.get("state") != HostStageState.CONSUMED.value
                or tp_rank not in current.get("binder_acks", [])
            ):
                return False
            return self.mark_d2p_recovery_phase_rank(
                snapshot_id, owner, tp_rank=tp_rank, tp_size=tp_size,
                claim_id=claim_id, lease_id=lease_id, phase="handed",
            )

    def apply_p2d_recovery_event(
        self, snapshot_id, owner, *, tp_rank, tp_size, decode_domain, attempt_id, event,
        reason=None,
    ):
        """Fence one destination generation; P2D never retries a live room.

        Each rank owns a receiver nonce. The first rank fixes the destination
        TP group; another receiver or D cannot steal/ack that physical lease.
        A retry must be a fresh request-generation, as in the native P2D path.
        """
        if (event not in {"begin", "loaded", "failed", "drained"} or not isinstance(attempt_id, str)
                or not attempt_id or type(tp_rank) is not int or type(tp_size) is not int
                or type(decode_domain) is not int or decode_domain < 0
                or not 0 <= tp_rank < tp_size):
            return False
        with self._condition:
            entry = self._entries.get(str(snapshot_id))
            if (entry is None or entry.get("p_owner") != owner
                    or int(entry.get("tp_size", 1)) != tp_size
                    or entry.get("p2d_decode_domain", decode_domain) != decode_domain):
                return False
            claims = entry.get("p2d_receiver_attempts", {})
            previous = claims.get(str(tp_rank))
            if previous is not None and previous != attempt_id:
                return False
            if event in {"failed", "drained"}:
                if event == "drained" and not entry.get("h2d_abort_started", False):
                    return False
                if previous is None:
                    # A rank may observe peer cancellation before launching
                    # any DMA. Bind its no-READ receipt to this receiver once,
                    # without allowing a replacement to cancel a live lease.
                    allowed = ({HostStageState.ABORTING.value} if event == "drained" else {
                        HostStageState.HOST_READY.value, HostStageState.H2D_LOADING.value,
                        HostStageState.ABORTING.value,
                    })
                    if entry.get("state") not in allowed:
                        return False
                    def claim_abort(entries):
                        current = entries[snapshot_id]
                        current["p2d_decode_domain"] = decode_domain
                        current.setdefault("p2d_receiver_attempts", {})[str(tp_rank)] = attempt_id
                        return True, True
                    self._mutate(claim_abort, event_snapshot_id=snapshot_id)
                if event == "failed":
                    return self.request_host_load_failure(snapshot_id, owner, reason=reason or "p2d_h2d_failed")
                return self.mark_host_load_rank_drained(
                    snapshot_id, owner, tp_rank=tp_rank, tp_size=tp_size,
                )
            if event == "loaded":
                if previous != attempt_id or tp_rank not in entry.get("loading_ranks", []):
                    return False
                return self.complete_host_load_rank(
                    snapshot_id, owner, tp_rank=tp_rank, tp_size=tp_size,
                )
            if previous == attempt_id:
                return entry.get("state") in {
                    HostStageState.H2D_LOADING.value, HostStageState.CONSUMED.value,
                }
            if not self.begin_host_load_rank(
                snapshot_id, owner, tp_rank=tp_rank, tp_size=tp_size,
            ):
                return False
            def attach(entries):
                current = entries[snapshot_id]
                current["p2d_decode_domain"] = decode_domain
                current.setdefault("p2d_receiver_attempts", {})[str(tp_rank)] = attempt_id
                return True, True
            return self._mutate(attach, event_snapshot_id=snapshot_id)

    def drain_changes(self, *, timeout=0.0):
        """Get coalesced committed deltas; wait without periodic polling.

        This is a single-consumer broker API. ``timeout=None`` waits until a
        mutation arrives. A finite timeout is useful for orderly shutdown;
        it does not expire or release any ownership state.
        """
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be nonnegative or None")
        with self._condition:
            self._condition.wait_for(lambda: bool(self._changes), timeout=timeout)
            values = deepcopy(list(self._changes.values()))
            self._changes.clear()
            return values

    @staticmethod
    def _can_prune(entry, cutoff, consumed_cutoff):
        size = int(entry.get("tp_size", 1))
        if entry.get("source_host_node") and set(
            entry.get("source_host_released_ranks", [])
        ) != set(range(size)):
            return False
        claims = entry.get("recovery_claims", {})
        claim_id = entry.get("recovery_claim_id")
        if entry.get("state") == HostStageState.CONSUMED.value:
            # Radix-bound and source-released are NOT scheduler handoff.
            if claims or claim_id is not None:
                if not claim_id or set(claims) != {str(rank) for rank in range(size)}:
                    return False
                if any(
                    claim.get("phase") != "handed"
                    or claim.get("claim_id") != claim_id
                    or claim.get("lease_id") is None
                    for claim in claims.values()
                ):
                    return False
            return float(entry.get("updated_at", 0)) < consumed_cutoff
        # Failure records with attached recovery leases remain until explicit
        # cancellation/fence handling clears those claims, regardless of age.
        if claims or claim_id is not None:
            return False
        return (
            entry.get("state") in _TERMINAL_STATES
            and float(entry.get("updated_at", 0)) < cutoff
        )

    def prune(self, older_than_seconds=600.0, consumed_older_than_seconds=5.0):
        now = time.time()
        cutoff = now - max(0.0, older_than_seconds)
        consumed_cutoff = now - max(0.0, consumed_older_than_seconds)
        # This is explicit broker maintenance, not a thread doing scans or a
        # timer that changes physical ownership. It removes terminal metadata
        # only, under the same lock used for handoff and cancellation.
        with self._condition:
            removed = {
                sid: None for sid, value in self._entries.items()
                if self._can_prune(value, cutoff, consumed_cutoff)
            }
            removed_revisions = {
                sid: int(self._entries[sid].get("_event_revision", 0)) + 1
                for sid in removed
            }
            for sid in removed:
                current = self._entries[sid]
                if current.get("state") in {
                    HostStageState.RECOMPUTE_REQUIRED.value, HostStageState.FAILED.value,
                }:
                    receipt = {
                        key: deepcopy(current[key]) for key in (
                            "snapshot_id", "request_id", "generation", "tp_size",
                            "p_owner", "source_host_node", "source_host_engine",
                            "source_host_released_ranks", "token_count", "reason",
                            "evicted_at", "updated_at", "state",
                        ) if key in current
                    }
                    receipt.update(_event_revision=removed_revisions[sid], terminal_receipt=True)
                    self._retired_receipts[sid] = receipt
                    removed[sid] = receipt
                del self._entries[sid]
            self._retired.update(removed)
            self._record_changes_locked(removed, removed_revisions)

    @staticmethod
    def _unsupported(*args, **kwargs):
        raise NotImplementedError("file/legacy relay API is unavailable in the in-memory broker")

    # Fail closed if a future inherited method accidentally reaches an old
    # filesystem helper. No fake paths, compatibility file writes or NFS.
    _write_locked = _unsupported
    _event_path = _unsupported
    _entry_lock_path = _unsupported
    _relay_marker_path = _unsupported
    _publish_relay_marker = _unsupported
    _is_relay_snapshot = _unsupported
    _entry_locked = _unsupported
    _entry_read_locked = _unsupported
    _publish_entry_event_locked = _unsupported
    read_entry_event = _unsupported
    _mutate_relay_entry = _unsupported
    register_relay = _unsupported
    heartbeat_relay = _unsupported
    list_relays = _unsupported
    assign_transfer_path = _unsupported
    claim_relay_job = _unsupported
    relay_prepare_chunk = _unsupported
    relay_mark_source_sent = _unsupported
    relay_complete_chunk = _unsupported
    relay_fail_to_direct = _unsupported
