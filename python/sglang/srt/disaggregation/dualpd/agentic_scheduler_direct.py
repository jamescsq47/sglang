"""AgenticDirectMixin: shared native Scheduler state; no independent state owner."""

import logging
from sglang.srt.disaggregation.agentic_workset import AgenticPWorksetLease
from sglang.srt.disaggregation.agentic_workset import AgenticPWorksetLeaseBroker
from typing import Any
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from typing import Optional
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration
from sglang.srt.disaggregation.agentic_kv_lifecycle import SnapshotLifecycleError
from sglang.srt.disaggregation.agentic_kv_lifecycle import SnapshotNotReadyError
from sglang.srt.disaggregation.agentic_kv_lifecycle import SnapshotState
from dataclasses import dataclass
from sglang.srt.disaggregation.agentic_direct_transfer import debug_kv_digest
from sglang.srt.environ import envs
from contextlib import nullcontext
import os
import time
from sglang.srt.disaggregation.agentic_kv_lifecycle import token_ids_digest
import torch
from sglang.srt.disaggregation.agentic_scheduler_compat import _native_scheduler_class

logger = logging.getLogger("sglang.srt.managers.scheduler")


@dataclass
class AgenticEarlyDirectReceive:
    """P-owned reverse transfer that exists before the tokenized Req."""

    request: RequestGeneration
    manifest: Any
    claim_id: str
    receiver: Any
    device_indices: Optional[torch.Tensor]
    started_at: float
    arrived_at: float
    prefill_domain: Optional[int] = None
    workset_lease: Optional[AgenticPWorksetLease] = None
    io_attempt: Optional[str] = None
    io_quiesced: bool = False
    completed_at: Optional[float] = None
    group_committed: bool = False
    abort_requested: bool = False
    abort_release_claim: bool = False
    abort_reason: Optional[str] = None
    route_published: bool = False
    # TP binds are two-phase.  A scheduler tick may install and pin the
    # received shard in the local Radix tree, but the request is not admitted
    # until every rank reports the same prepared state.  Keeping the Req here
    # also gives group abort a precise object whose branch must be rolled back.
    prepared_req: Optional[Any] = None
    # Transport progress is driven by a lightweight worker while a long
    # Prefill kernel owns the scheduler thread.  Group lifecycle completion is
    # metadata-only and also progresses there; HBM ownership and Radix
    # insertion remain scheduler-owned.
    transport_poll: Optional[Any] = None
    radix_prepared: bool = False
    existing_tokens: int = 0

AgenticEarlyDirectReceive.__module__ = "sglang.srt.managers.scheduler"


class AgenticDirectMixin:
    def _agentic_start_early_direct_receive(
        self,
        request: RequestGeneration,
        manifest,
        snapshot_store,
        *,
        arrived_at: float,
        prefill_domain: Optional[int] = None,
        workset_lease: Optional[AgenticPWorksetLease] = None,
    ) -> bool:
        """Reserve P pages and start reverse NIXL before a Req exists."""

        runtime = getattr(self, "agentic_direct_runtime", None)
        if runtime is None or getattr(self.tree_cache, "is_eagle", False):
            return False
        if manifest.tp_size != self.tp_size:
            logger.error(
                "AgenticKV Direct TP mismatch snapshot=%s source=%d destination=%d",
                manifest.snapshot_id,
                manifest.tp_size,
                self.tp_size,
            )
            if self.tp_rank == 0 and manifest.state is SnapshotState.DIRECT_READY:
                try:
                    snapshot_store.fail_direct_offer(
                        manifest,
                        owner_id=f"p-incompatible:{os.getpid()}",
                        reason="permanent_direct_tp_mismatch",
                    )
                except Exception:
                    logger.exception(
                        "AgenticKV permanent TP mismatch commit retry snapshot=%s",
                        manifest.snapshot_id,
                    )
            self.agentic_p_workset_broker.request_release(
                request.snapshot_id, workset_lease
            )
            return False
        if manifest.kv_layout_hash and manifest.kv_layout_hash != runtime.layout_hash:
            logger.error(
                "AgenticKV Direct layout mismatch snapshot=%s source=%s destination=%s",
                manifest.snapshot_id,
                manifest.kv_layout_hash,
                runtime.layout_hash,
            )
            if self.tp_rank == 0 and manifest.state is SnapshotState.DIRECT_READY:
                try:
                    snapshot_store.fail_direct_offer(
                        manifest,
                        owner_id=f"p-incompatible:{os.getpid()}",
                        reason="permanent_direct_layout_mismatch",
                    )
                except Exception:
                    logger.exception(
                        "AgenticKV permanent layout mismatch commit retry "
                        "snapshot=%s",
                        manifest.snapshot_id,
                    )
            self.agentic_p_workset_broker.request_release(
                request.snapshot_id, workset_lease
            )
            return False
        if workset_lease is None:
            return False
        if workset_lease.parent_tokens < manifest.token_count:
            raise RuntimeError("Direct workset parent slice is too small")
        device_indices = workset_lease.parent_indices[: manifest.token_count]
        if self.tp_size == 1:
            claim_id = (
                f"direct-early-p:{os.getpid()}:{request.snapshot_id}:"
                f"{time.monotonic_ns()}"
            )
        else:
            # Every TP rank must join the same logical claim while receiving
            # only its own physical KV-head shard.
            engine_id = os.getenv("SGLANG_AGENTIC_KV_ENGINE_ID", "prefill")
            claim_id = f"direct-early-tp:{engine_id}:{request.snapshot_id}"
        with getattr(self, "agentic_early_direct_poll_lock", nullcontext()):
            if request.snapshot_id in getattr(
                self, "agentic_early_direct_terminal", ()
            ):
                return False
        if not self.agentic_p_workset_broker.begin_io_attempt(
            request.snapshot_id, workset_lease, claim_id
        ):
            # Another Direct attempt already owns this exact workset.  It is
            # the only attempt allowed to claim, quiesce, or release the DMA
            # destination; this caller leaves both lifecycle and pages alone.
            return False
        receiver = None
        claimed = None
        direct_requested = getattr(self, "agentic_direct_poll_requested", None)
        nixl_lock = getattr(self, "agentic_nixl_control_lock", nullcontext())
        try:
            claimed = snapshot_store.claim_direct(request, claim_id)
            if direct_requested is not None:
                direct_requested.set()
            with nixl_lock:
                if not runtime.manager.try_ensure_parallel_info(
                    claimed.direct_bootstrap_addr
                ):
                    raise SnapshotNotReadyError("reverse bootstrap is not ready")
                receiver = runtime.receiver_class(
                    mgr=runtime.manager,
                    bootstrap_addr=claimed.direct_bootstrap_addr,
                    bootstrap_room=claimed.direct_room,
                )
                receiver.init(prefill_dp_rank=0)
                if receiver.poll() == KVPoll.Failed:
                    raise SnapshotLifecycleError("reverse receiver init failed")
                self.agentic_p_workset_broker.mark_io_inflight(
                    request.snapshot_id, workset_lease, claim_id
                )
                if workset_lease.state_device_indices:
                    from sglang.srt.disaggregation.agentic_hybrid_transfer import submit_reverse_receive
                    submit_reverse_receive(receiver, workset_lease, ("mamba",))
                else:
                    receiver.send_metadata(workset_lease.parent_page_indices, aux_index=0)
        except Exception as exc:
            transport_may_write = bool(
                receiver is not None
                and getattr(receiver, "started_transfer", False)
                and workset_lease.io_attempt == claim_id
                and workset_lease.state in {"io_inflight", "release_pending"}
            )
            if transport_may_write:
                self.agentic_p_workset_broker.request_release(
                    request.snapshot_id,
                    workset_lease,
                    io_attempt=claim_id,
                )
                entry = AgenticEarlyDirectReceive(
                    request=request,
                    manifest=claimed if claimed is not None else manifest,
                    claim_id=claim_id,
                    receiver=receiver,
                    device_indices=device_indices,
                    started_at=time.monotonic(),
                    arrived_at=arrived_at,
                    prefill_domain=prefill_domain,
                    workset_lease=workset_lease,
                    io_attempt=claim_id,
                    abort_requested=True,
                    abort_release_claim=True,
                    abort_reason="metadata_publication_failed",
                )
                with getattr(self, "agentic_early_direct_poll_lock", nullcontext()):
                    self.agentic_early_direct_receives[request.snapshot_id] = entry
                logger.exception(
                    "Direct metadata publication may be partially visible; "
                    "quarantining workset snapshot=%s",
                    request.snapshot_id,
                )
                return True
            if (
                workset_lease.state in {"io_inflight", "release_pending"}
                and workset_lease.io_attempt == claim_id
            ):
                self.agentic_p_workset_broker.mark_io_quiesced(
                    request.snapshot_id, workset_lease, claim_id
                )
            else:
                self.agentic_p_workset_broker.cancel_io_attempt(
                    request.snapshot_id, workset_lease, claim_id
                )
            self.agentic_p_workset_broker.request_release(
                request.snapshot_id, workset_lease
            )
            if receiver is not None:
                try:
                    receiver.clear()
                    if claimed is not None:
                        self._agentic_clear_direct_receiver(receiver, claimed)
                except Exception:
                    logger.exception(
                        "Failed to clear early Direct receiver for %s",
                        request.snapshot_id,
                    )
            current = snapshot_store.load(request, require_ready=False)
            # In TP mode the deterministic claim is group-owned.  A follower
            # that fails to initialize its local receiver must only roll back
            # its own pages/transport; releasing the shared claim here would
            # invalidate TP0 and every already-started peer.  TP0 (or TP=1)
            # Only TP=1 can release here; TP groups use the unified abort path.
            owns_group_claim = self.tp_size == 1
            if (
                owns_group_claim
                and current is not None
                and current.state is SnapshotState.DIRECT_LOADING
                and current.claim_id == claim_id
            ):
                try:
                    snapshot_store.release_direct_claim(current, claim_id)
                except Exception:
                    logger.exception(
                        "Failed to release early Direct claim for %s",
                        request.snapshot_id,
                    )
            if not isinstance(exc, SnapshotNotReadyError):
                if getattr(self, "tp_size", 1) > 1:
                    failed = getattr(self, "agentic_tp_direct_local_failed", None)
                    if failed is not None:
                        failed.add(request.snapshot_id)
                logger.exception(
                    "Could not start early Direct D->P receive for %s",
                    request.snapshot_id,
                )
            return False
        finally:
            if direct_requested is not None:
                direct_requested.clear()

        entry = AgenticEarlyDirectReceive(
            request=request,
            manifest=claimed,
            claim_id=claim_id,
            receiver=receiver,
            device_indices=device_indices,
            started_at=time.monotonic(),
            arrived_at=arrived_at,
            prefill_domain=prefill_domain,
            workset_lease=workset_lease,
            io_attempt=claim_id,
        )
        with getattr(self, "agentic_early_direct_poll_lock", nullcontext()):
            cancelled = request.snapshot_id in getattr(
                self, "agentic_early_direct_terminal", ()
            )
            if cancelled:
                entry.abort_requested = True
                entry.abort_release_claim = True
                entry.abort_reason = "tp_control_cancelled_during_start"
                self.agentic_p_workset_broker.request_release(
                    request.snapshot_id,
                    workset_lease,
                    io_attempt=claim_id,
                )
            self.agentic_early_direct_receives[request.snapshot_id] = entry
        logger.info(
            "AgenticKV early_direct_start snapshot=%s tokens=%d "
            "arrival_to_start_ms=%.3f workset_tokens=%d",
            request.snapshot_id,
            claimed.token_count,
            max(0.0, (time.time() - arrived_at) * 1000.0),
            workset_lease.allocated_tokens,
        )
        return True

    def _agentic_drop_early_direct_receive(
        self,
        entry: AgenticEarlyDirectReceive,
        snapshot_store,
        *,
        release_claim: bool,
        reason: str,
    ) -> None:
        transport_terminal = entry.completed_at is not None or entry.transport_poll in {
            KVPoll.Success,
            KVPoll.Failed,
        }
        if (
            not transport_terminal
            and entry.workset_lease is not None
            and entry.workset_lease.state in {"io_inflight", "release_pending"}
        ):
            # Logical timeout/TP abort is not a DMA fence. Keep the receiver
            # pollable and quarantine its pages until NIXL reports a terminal
            # result; otherwise a late remote WRITE could corrupt a new Req.
            entry.abort_requested = True
            entry.abort_release_claim = entry.abort_release_claim or release_claim
            entry.abort_reason = reason
            self.agentic_p_workset_broker.request_release(
                entry.request.snapshot_id,
                entry.workset_lease,
                io_attempt=entry.io_attempt,
            )
            logger.warning(
                "AgenticKV early_direct_abort_deferred snapshot=%s reason=%s",
                entry.request.snapshot_id,
                reason,
            )
            return
        if entry.completed_at is None:
            try:
                entry.receiver.clear()
                self._agentic_clear_direct_receiver(entry.receiver, entry.manifest)
            except Exception:
                logger.exception(
                    "Failed to clear early Direct receiver for %s",
                    entry.request.snapshot_id,
                )
        # A completed negative-send/terminal fence makes this Direct owner
        # permanently obsolete for the request-generation.  A plain release
        # is insufficient here: an arrival already queued by the inotify
        # worker can recreate the same owner after the allocator services the
        # release.  Install the owner-scoped tombstone and retire both a
        # pending intent and an unstarted active lease atomically.  Slow uses
        # a distinct owner, so this cannot reclaim or reject its workset.
        direct_owner_fn = getattr(
            self.agentic_p_workset_broker, "direct_owner", None
        )
        supersede = getattr(
            self.agentic_p_workset_broker, "supersede_unstarted", None
        )
        if callable(direct_owner_fn) and callable(supersede):
            supersede(
                entry.request.snapshot_id,
                owner=direct_owner_fn(entry.request.snapshot_id),
            )
        else:
            # Focused tests and older embedders may provide the historical
            # minimal broker surface.  The production broker always supports
            # the tombstone path above.
            self.agentic_p_workset_broker.request_release(
                entry.request.snapshot_id, entry.workset_lease
            )
        if release_claim:
            try:
                current = snapshot_store.load(entry.request, require_ready=False)
                if (
                    current is not None
                    and current.state in {
                        SnapshotState.DIRECT_LOADING,
                        SnapshotState.P_RECEIVED,
                    }
                    and current.claim_id == entry.claim_id
                ):
                    if current.state is SnapshotState.P_RECEIVED:
                        snapshot_store.release_received_direct(
                            current, entry.claim_id
                        )
                    else:
                        snapshot_store.release_direct_claim(
                            current, entry.claim_id
                        )
            except Exception:
                logger.exception(
                    "Failed to release early Direct claim for %s",
                    entry.request.snapshot_id,
                )
        with getattr(self, "agentic_early_direct_poll_lock", nullcontext()):
            if (
                self.agentic_early_direct_receives.get(entry.request.snapshot_id)
                is entry
            ):
                self.agentic_early_direct_receives.pop(entry.request.snapshot_id, None)
            self.agentic_early_direct_terminal[entry.request.snapshot_id] = (
                time.monotonic()
            )
            if getattr(self, "tp_size", 1) > 1:
                failed = getattr(self, "agentic_tp_direct_local_failed", None)
                if failed is not None:
                    failed.add(entry.request.snapshot_id)
        logger.warning(
            "AgenticKV early_direct_drop snapshot=%s reason=%s",
            entry.request.snapshot_id,
            reason,
        )

    def _agentic_handle_unstarted_direct_abort(
        self,
        entry: AgenticEarlyDirectReceive,
        snapshot_store,
        marker_store,
        poll,
        direct_timeout: float,
    ) -> bool:
        """Apply D's negative-send fence and return its claim for Slow.

        For TP=1 this is either a negative-send proof or a physical terminal
        proof. TP groups publish only after every D shard is no-future-write;
        P then uses its own rank mailbox before the shared claim is returned.
        A contradictory ``Transferring`` observation under an unstarted proof
        remains quarantined until a real physical terminal notification.
        """

        now = time.monotonic()
        if now < float(getattr(entry, "direct_abort_retry_at", 0.0)):
            return True
        def defer_retry() -> None:
            delay = max(
                0.05,
                float(getattr(entry, "direct_abort_retry_delay", 0.05)),
            )
            entry.direct_abort_retry_at = now + delay
            entry.direct_abort_retry_delay = min(1.0, delay * 2.0)

        if not getattr(entry, "direct_abort_marker_seen", False):
            read_abort = getattr(marker_store, "read_direct_abort", None)
            if not callable(read_abort):
                return False
            pending_names = getattr(
                self, "agentic_direct_abort_pending_names", set()
            )
            abort_path = getattr(marker_store, "direct_abort_path", None)
            marker_name = (
                abort_path(entry.request).name
                if callable(abort_path)
                else entry.request.snapshot_id
            )
            event_pending = marker_name in pending_names
            # Before the ordinary shared deadline, only an inotify edge can
            # trigger an exact read. Thus explicit init/terminal failures are
            # handled immediately while the common no-marker case performs
            # zero per-request filesystem polling.
            if (
                not event_pending
                and time.time()
                < float(entry.arrived_at) + float(direct_timeout)
            ):
                return False
            if now < float(getattr(entry, "direct_abort_next_poll_at", 0.0)):
                return False
            absent_delay = max(
                0.05,
                float(getattr(entry, "direct_abort_absent_delay", 0.05)),
            )
            entry.direct_abort_next_poll_at = now + absent_delay
            marker = read_abort(
                entry.request,
                claim_id=entry.claim_id,
                not_before=float(getattr(entry.manifest, "created_at", 0.0)),
                # Claim-id matching makes a stale marker unable to cancel a
                # newer Direct session. This exact live claim's negative-send
                # fence must not expire during an arbitrarily long P control
                # stall; successful cleanup removes it explicitly.
                max_age_seconds=float("inf"),
            )
            pending_names.discard(marker_name)
            if marker is None:
                # D's control worker may observe the deadline slightly after
                # P. Back off missing-marker reads to a bounded 2 Hz rather
                # than doing O(active requests) I/O every transport tick.
                entry.direct_abort_absent_delay = min(0.5, absent_delay * 2.0)
                return False
            entry.direct_abort_marker_seen = True
            entry.direct_abort_fence_kind = marker.get(
                "fence_kind", "unstarted"
            )
            entry.direct_abort_absent_delay = 0.05
        terminal_fence = (
            getattr(entry, "direct_abort_fence_kind", "unstarted")
            == "terminal"
        )
        if poll == KVPoll.Transferring and not terminal_fence:
            # This should be unreachable because D publishes the marker only
            # before sender submission.  Fail closed if transport disagrees:
            # request release, but keep polling until the real DMA fence.
            if not entry.abort_requested:
                entry.abort_requested = True
                entry.abort_release_claim = True
                entry.abort_reason = "direct_abort_after_transfer_started"
                self.agentic_p_workset_broker.request_release(
                    entry.request.snapshot_id,
                    entry.workset_lease,
                    io_attempt=entry.io_attempt,
                )
                logger.error(
                    "AgenticKV direct_abort_transport_conflict snapshot=%s "
                    "claim=%s",
                    entry.request.snapshot_id,
                    entry.claim_id,
                )
            return True
        if (
            not terminal_fence
            and poll not in {KVPoll.Bootstrapping, KVPoll.WaitingForInput}
        ):
            # A terminal transport result is stronger than the cancellation
            # marker.  Let the normal Success/Failed path settle it.
            return False
        # Check the exact lifecycle owner before making allocator I/O
        # quiescent.  A contradictory P_RECEIVED/CONSUMED state is stronger
        # evidence than this marker and must finish through the normal Direct
        # completion path instead of recycling its pages.
        try:
            current = snapshot_store.load(entry.request, require_ready=False)
            release_claim = bool(
                current is not None
                and current.state is SnapshotState.DIRECT_LOADING
                and current.claim_id == entry.claim_id
            )
            if not release_claim and (
                current is None or current.state is not SnapshotState.DIRECT_READY
            ):
                return False
        except Exception:
            defer_retry()
            if not getattr(entry, "direct_abort_error_logged", False):
                entry.direct_abort_error_logged = True
                logger.exception(
                    "AgenticKV direct_abort_claim_read_retry snapshot=%s "
                    "claim=%s action=retain_and_retry",
                    entry.request.snapshot_id,
                    entry.claim_id,
                )
            return True

        if not entry.io_quiesced:
            if entry.workset_lease is not None and not (
                entry.io_attempt is not None
                and self.agentic_p_workset_broker.mark_io_quiesced(
                    entry.request.snapshot_id,
                    entry.workset_lease,
                    entry.io_attempt,
                )
            ):
                defer_retry()
                if not getattr(entry, "direct_abort_error_logged", False):
                    entry.direct_abort_error_logged = True
                    logger.error(
                        "AgenticKV direct_abort_io_attempt_mismatch snapshot=%s "
                        "claim=%s action=retain_and_retry",
                        entry.request.snapshot_id,
                        entry.claim_id,
                    )
                return True
            entry.io_quiesced = True

        tp_size = int(getattr(self, "tp_size", 1))
        if tp_size > 1 and release_claim:
            mailbox = getattr(self, "agentic_tp_direct_abort_mailbox", None)
            if mailbox is None:
                return True
            # The D marker is already a group source fence. Before TP0 returns
            # the shared lifecycle claim, every P rank must also prove its
            # destination workset I/O quiescent.
            if not _native_scheduler_class()._agentic_tp_abort_mailbox_write(
                self,
                entry,
                lambda: mailbox.publish_local_progress(
                    entry.request.snapshot_id, 1
                ),
            ):
                return True
            statuses = [
                mailbox.local_status(entry.request.snapshot_id, rank)
                for rank in range(tp_size)
            ]
            if any(status is None or int(status) < 1 for status in statuses):
                return True
            if getattr(self, "tp_rank", 0) != 0:
                try:
                    current = snapshot_store.load(
                        entry.request, require_ready=False
                    )
                except Exception:
                    return True
                if (
                    current is None
                    or current.state is not SnapshotState.DIRECT_READY
                ):
                    return True
                release_claim = False

        # Publish the lifecycle acknowledgement before releasing destination
        # pages.  If metadata is transiently unavailable, retain the entry and
        # retry; D keeps its source pinned and cannot enter Slow yet.
        try:
            if release_claim:
                current = snapshot_store.release_direct_claim(
                    current, entry.claim_id
                )
            if current is None or current.state is not SnapshotState.DIRECT_READY:
                return False
        except Exception:
            defer_retry()
            if not getattr(entry, "direct_abort_error_logged", False):
                entry.direct_abort_error_logged = True
                logger.exception(
                    "AgenticKV direct_abort_claim_release_retry snapshot=%s "
                    "claim=%s action=retain_and_retry",
                    entry.request.snapshot_id,
                    entry.claim_id,
                )
            return True

        # Cleanup of the TP coordination epoch is part of the phase boundary,
        # not best-effort garbage collection. Keep the entry/workset as the
        # retry carrier until clear succeeds; D cannot receive a contradictory
        # command epoch while stale rank files remain visible.
        if tp_size > 1 and getattr(self, "tp_rank", 0) == 0:
            mailbox = getattr(self, "agentic_tp_direct_abort_mailbox", None)
            if mailbox is not None and not _native_scheduler_class()._agentic_tp_abort_mailbox_write(
                self,
                entry,
                lambda: mailbox.clear_group(entry.request.snapshot_id),
            ):
                return True

        # The negative-send marker plus DIRECT_READY transition is the full
        # cancellation fence.  Marking the synthetic local poll terminal now
        # only authorizes cleanup of this exact attempt; it does not claim a
        # transport failure or transfer ownership to P.
        entry.transport_poll = KVPoll.Failed
        self._agentic_drop_early_direct_receive(
            entry,
            snapshot_store,
            release_claim=False,
            reason=(
                "direct_transport_terminal_failed"
                if terminal_fence
                else "direct_setup_deadline_unstarted"
            ),
        )
        remove_abort = getattr(marker_store, "remove_direct_abort", None)
        if callable(remove_abort) and (
            tp_size == 1 or getattr(self, "tp_rank", 0) == 0
        ):
            remove_abort(entry.request)
        logger.warning(
            "AgenticKV direct_abort_complete snapshot=%s claim=%s "
            "action=return_to_decode",
            entry.request.snapshot_id,
            entry.claim_id,
        )
        return True

    def _agentic_collect_direct_abort_events(self) -> None:
        """Collect rare Direct fence edges without scanning active entries."""

        watcher = getattr(self, "agentic_direct_abort_watcher", None)
        if watcher is None:
            return
        paths, overflow = watcher.poll(0.0)
        pending = self.agentic_direct_abort_pending_names
        pending.update(path.name for path in paths)
        if overflow:
            marker_store = getattr(self, "agentic_early_claim_store", None)
            if marker_store is not None:
                pending.update(
                    marker_store.direct_abort_path(entry.request).name
                    for entry in self.agentic_early_direct_receives.values()
                )

    def _agentic_return_early_direct_to_slow(
        self,
        entry: AgenticEarlyDirectReceive,
        req: Req,
        *,
        reason: str,
    ) -> bool:
        """Return a quiescent TP1 Direct session to D without recomputing.

        D retains the authoritative source until lifecycle CONSUMED.  Once P
        has received but cannot bind the parent, returning P_RECEIVED to
        DIRECT_READY lets D's independent progress worker stage that intact
        source through the Slow queue.  The child stays deferred throughout.
        """

        self._agentic_drop_early_direct_receive(
            entry,
            self._agentic_snapshot_store(),
            release_claim=True,
            reason=reason,
        )
        req._agentic_kv_queue_class = "slow"
        return True

    def _agentic_early_direct_progress_worker(self) -> None:
        """Own Direct discovery and transport progress off the GPU scheduler.

        The worker consumes scheduler-granted complete-workset leases; it
        never mutates the shared SGLang allocator or Radix tree. The scheduler
        later performs only the completed-request binding step.
        """

        try:
            interval = max(
                0.001,
                float(
                    os.environ.get(
                        "SGLANG_AGENTIC_KV_P_DIRECT_PROGRESS_INTERVAL_SECONDS",
                        "0.005",
                    )
                ),
            )
        except ValueError:
            logger.exception("Invalid P Direct progress interval")
            return
        cycles = 0
        total_seconds = 0.0
        max_seconds = 0.0
        last_stats = time.monotonic()
        while not self.agentic_early_direct_progress_stop.is_set():
            started = time.monotonic()
            try:
                self._agentic_poll_early_direct_receives()
            except Exception:
                logger.exception("P Direct ingress worker failed")
            elapsed = time.monotonic() - started
            cycles += 1
            total_seconds += elapsed
            max_seconds = max(max_seconds, elapsed)
            now = time.monotonic()
            if now - last_stats >= 30.0:
                with getattr(self, "agentic_early_direct_poll_lock", nullcontext()):
                    active = sum(
                        entry.completed_at is None
                        and entry.transport_poll not in {KVPoll.Success, KVPoll.Failed}
                        for entry in self.agentic_early_direct_receives.values()
                    )
                    ready = sum(
                        entry.completed_at is not None
                        for entry in self.agentic_early_direct_receives.values()
                    )
                    admission_pending = len(self.agentic_early_direct_admission_queue)
                workset_pending, workset_grants, workset_alloc_misses = (
                    self.agentic_p_workset_broker.stats
                )
                logger.info(
                    "Agentic P Direct progress stats cycles=%d avg_us=%.1f "
                    "max_ms=%.3f admission_pending=%d active=%d ready=%d "
                    "leased_workset_tokens=%d workset_pending=%d "
                    "workset_grants=%d workset_alloc_misses=%d "
                    "lease_states=%s active_leases=%s",
                    cycles,
                    total_seconds / max(cycles, 1) * 1e6,
                    max_seconds * 1e3,
                    admission_pending,
                    active,
                    ready,
                    self.agentic_p_workset_broker.leased_tokens,
                    workset_pending,
                    workset_grants,
                    workset_alloc_misses,
                    self.agentic_p_workset_broker.lease_state_summary,
                    self.agentic_p_workset_broker.active_lease_summary,
                )
                cycles = 0
                total_seconds = 0.0
                max_seconds = 0.0
                last_stats = now
            self.agentic_early_direct_progress_stop.wait(interval)

    def _agentic_poll_early_direct_receives(self, now: Optional[float] = None) -> None:
        cycle_lock = getattr(self, "agentic_early_direct_cycle_lock", None)
        if cycle_lock is None:
            return self._agentic_poll_early_direct_receives_once(now)
        with cycle_lock:
            return self._agentic_poll_early_direct_receives_once(now)

    def _agentic_collect_direct_arrivals(self, poll_lock) -> None:
        """Move paths reported by inotify into the Direct admission FIFO."""

        watcher = getattr(self, "agentic_early_direct_arrival_watcher", None)
        if watcher is None:
            return
        arrivals = watcher.poll(0.0)
        if not arrivals:
            return
        with poll_lock:
            queue = self.agentic_early_direct_admission_queue
            pending = self.agentic_early_direct_admission_ids
            for request, payload in arrivals:
                snapshot_id = request.snapshot_id
                if (
                    snapshot_id in pending
                    or snapshot_id in self.agentic_early_direct_receives
                    or snapshot_id in self.agentic_early_direct_terminal
                ):
                    continue
                queue.append((request, payload, None))
                pending.add(snapshot_id)

    def _agentic_early_direct_slots_used(self) -> int:
        """Count physical or TP-granted reverse transfers on this P rank.

        Completed receives no longer consume a NIXL lane even if their
        workset remains pinned until the model request binds.  TP grants do
        consume a lane before the local receiver becomes visible, so count
        their request-generation exactly once as well.
        """

        poll_lock = getattr(self, "agentic_early_direct_poll_lock", nullcontext())
        with poll_lock:
            receives = dict(
                getattr(self, "agentic_early_direct_receives", {})
            )
            physical = {
                snapshot_id
                for snapshot_id, entry in receives.items()
                if entry.completed_at is None
                and entry.transport_poll not in {KVPoll.Success, KVPoll.Failed}
            }
            # TP active state outlives local DMA completion until the logical
            # group is bound.  Count only a grant that has not produced its
            # local receiver yet; otherwise the receive state is the physical
            # truth and a completed DMA immediately returns its lane.
            granted = set(
                getattr(self, "agentic_tp_direct_admission_active", {}).keys()
            ) - set(receives)
        return len(physical | granted)

    def _agentic_admit_queued_direct_receives(
        self,
        snapshot_store,
        direct_timeout: float,
        poll_lock,
    ) -> None:
        """Claim queued arrivals immediately when exact-size credit is free."""

        queue = getattr(self, "agentic_early_direct_admission_queue", None)
        pending = getattr(self, "agentic_early_direct_admission_ids", None)
        if queue is None or pending is None:
            return
        marker_max_age = max(
            5.0,
            envs.SGLANG_AGENTIC_KV_FAST_TOOL_THRESHOLD.get() + direct_timeout + 1.0,
        )
        dynamic_domains = os.environ.get(
            "SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        configured_domain = int(
            os.environ.get("SGLANG_AGENTIC_KV_PREFILL_DOMAIN", "-1")
        )
        marker_store = getattr(self, "agentic_early_claim_store", None)
        tp_size = int(getattr(self, "tp_size", 1))
        tp_rank = int(getattr(self, "tp_rank", 0))
        if tp_size > 1 and marker_store is None:
            raise RuntimeError("TP Direct admission lost its early-claim store")
        tp_active = getattr(self, "agentic_tp_direct_admission_active", None)
        if not isinstance(tp_active, dict):
            tp_active = {}
            self.agentic_tp_direct_admission_active = tp_active
        direct_io_cap = max(
            1, int(os.environ.get("SGLANG_AGENTIC_KV_DIRECT_IO_CAP", "4"))
        )
        direct_slots_used = _native_scheduler_class()._agentic_early_direct_slots_used(self)
        # Examine each currently queued request once.  A large snapshot with
        # insufficient credit is rotated behind smaller requests instead of
        # causing head-of-line blocking; FIFO order is otherwise preserved.
        with poll_lock:
            attempts = len(queue)
        for _ in range(attempts):
            with poll_lock:
                if not queue:
                    break
                request, payload, manifest = queue.popleft()
                pending.discard(request.snapshot_id)
                if (
                    request.snapshot_id in self.agentic_early_direct_receives
                    or request.snapshot_id in self.agentic_early_direct_terminal
                ):
                    continue

            arrived_at = float(payload["arrived_at"])
            if arrived_at + marker_max_age < time.time():
                self.agentic_p_workset_broker.cancel_unstarted(
                    request.snapshot_id,
                    owner=AgenticPWorksetLeaseBroker.direct_owner(request.snapshot_id),
                )
                continue
            target_domain = payload.get("target_prefill_domain")
            if dynamic_domains:
                if target_domain is None or int(target_domain) != configured_domain:
                    continue
            else:
                # Preserve the established 1P behavior: its arrival markers
                # are untargeted and require no route-resolution handshake.
                target_domain = None

            broker = self.agentic_p_workset_broker
            workset_owner = AgenticPWorksetLeaseBroker.direct_owner(
                request.snapshot_id
            )
            if broker.owner_is_superseded(
                request.snapshot_id, owner=workset_owner
            ):
                # HOST_READY won the Direct/Slow race before this inotify
                # marker reached the admission queue.  Consume the stale
                # marker exactly once; the complete parent is already owned
                # by Shared Host and will enter through the independent Slow
                # recovery queue.
                with poll_lock:
                    self.agentic_early_direct_terminal[request.snapshot_id] = (
                        time.time()
                    )
                continue

            if manifest is None:
                manifest = snapshot_store.load(request, require_ready=False)
            if manifest is None:
                # The marker and lifecycle manifest are written by different
                # processes.  Retain the event briefly if publication order is
                # observed in reverse; no directory rescan is required.
                with poll_lock:
                    queue.append((request, payload, None))
                    pending.add(request.snapshot_id)
                continue
            if arrived_at + 0.05 < manifest.created_at:
                continue
            prompt_tokens = payload.get("prompt_token_count")
            if prompt_tokens is None:
                # Without the next prompt length P cannot atomically reserve
                # parent+tool suffix.  Leave DIRECT_READY unclaimed so D takes
                # the ordinary timeout-to-Slow path rather than overcommitting.
                continue
            prompt_tokens = int(prompt_tokens)
            if prompt_tokens < int(manifest.token_count):
                logger.warning(
                    "AgenticKV invalid workset marker snapshot=%s parent=%d prompt=%d",
                    request.snapshot_id,
                    int(manifest.token_count),
                    prompt_tokens,
                )
                continue
            eligible_states = (
                {SnapshotState.DIRECT_READY, SnapshotState.DIRECT_LOADING}
                if tp_size > 1 and tp_rank != 0
                else {SnapshotState.DIRECT_READY}
            )
            if manifest.state not in eligible_states:
                # A stale arrival can outlive D's Direct->Slow transition.
                # The previous worker pass may already have published a
                # Direct intent which the scheduler granted before D crossed
                # the timeout boundary.  Cancel that exact Direct owner here;
                # otherwise an unstarted ``active`` workset can permanently
                # occupy P HBM while the authoritative parent is waiting in
                # Shared Host.  Owner scoping keeps the Slow restore lease
                # independent, and TP brokers retire the stale grant through
                # their normal group-synchronous plan.
                self.agentic_p_workset_broker.cancel_unstarted(
                    request.snapshot_id,
                    owner=AgenticPWorksetLeaseBroker.direct_owner(
                        request.snapshot_id
                    ),
                )
                continue
            # A TP follower is an executor, never an admission authority.
            # Wait for rank0's exact generation grant before reserving any P
            # pages; otherwise a burst of not-yet-granted arrivals can bypass
            # the physical lane cap by accumulating invisible workset leases.
            if tp_size > 1 and tp_rank != 0:
                receipt = self.agentic_tp_direct_mailbox.receipt(
                    request.snapshot_id
                )
                if receipt is None:
                    with poll_lock:
                        queue.append((request, payload, manifest))
                        pending.add(request.snapshot_id)
                    continue
                if int(receipt) < 0 or int(receipt) >= 4:
                    broker.cancel_unstarted(
                        request.snapshot_id,
                        owner=workset_owner,
                    )
                    self.agentic_early_direct_terminal[request.snapshot_id] = (
                        time.monotonic()
                    )
                    continue
            if direct_slots_used >= direct_io_cap:
                # Keep the request metadata-only until a physical Direct lane
                # reaches DONE/ERR.  In particular, do not acquire a workset
                # lease here: a queued transfer must not consume P HBM merely
                # because the shared NIXL agent is at its safe concurrency.
                with poll_lock:
                    queue.append((request, payload, manifest))
                    pending.add(request.snapshot_id)
                continue
            broker.request(
                request.snapshot_id,
                int(manifest.token_count),
                prompt_tokens,
                owner=workset_owner,
            )
            workset_lease = broker.get(request.snapshot_id, owner=workset_owner)
            if workset_lease is None:
                with poll_lock:
                    queue.append((request, payload, manifest))
                    pending.add(request.snapshot_id)
                continue
            # A queue entry deliberately caches its manifest while waiting
            # for allocator credit.  Once credit is actually granted, refresh
            # the authoritative lifecycle exactly once before starting I/O:
            # D may have crossed DIRECT_READY -> SLOW_FALLBACK between worker
            # passes.  Checking only the cached object would strand this new
            # active lease in P HBM even though Shared Host now owns the
            # parent.  Avoid refreshing every metadata-only queue rotation;
            # only a physical grant pays this store read.
            authoritative_manifest = snapshot_store.load(
                request, require_ready=False
            )
            authoritative_eligible_states = (
                {SnapshotState.DIRECT_READY, SnapshotState.DIRECT_LOADING}
                if tp_size > 1 and tp_rank != 0
                else {SnapshotState.DIRECT_READY}
            )
            if (
                authoritative_manifest is None
                or authoritative_manifest.state
                not in authoritative_eligible_states
            ):
                broker.cancel_unstarted(
                    request.snapshot_id,
                    owner=workset_owner,
                )
                continue
            manifest = authoritative_manifest
            if tp_size > 1 and tp_rank != 0:
                # TP0 publishes one exact request-generation grant through
                # the dedicated tmpfs mailbox.  Followers never choose work
                # independently; they merely mirror that grant and let their
                # background progress worker start the local KV-head shard.
                receipt = self.agentic_tp_direct_mailbox.receipt(request.snapshot_id)
                if receipt is None:
                    broker.cancel_unstarted(
                        request.snapshot_id,
                        owner=workset_owner,
                    )
                    with poll_lock:
                        queue.append((request, payload, manifest))
                        pending.add(request.snapshot_id)
                    continue
                if int(receipt) < 0 or int(receipt) >= 4:
                    broker.cancel_unstarted(
                        request.snapshot_id,
                        owner=workset_owner,
                    )
                    self.agentic_early_direct_terminal[request.snapshot_id] = (
                        time.monotonic()
                    )
                    continue
                if manifest.state not in {
                    SnapshotState.DIRECT_READY,
                    SnapshotState.DIRECT_LOADING,
                }:
                    broker.request_release(request.snapshot_id, workset_lease)
                    self.agentic_tp_direct_local_failed.add(request.snapshot_id)
                    continue
                with poll_lock:
                    if request.snapshot_id in self.agentic_early_direct_terminal:
                        broker.request_release(request.snapshot_id, workset_lease)
                        continue
                    tp_active[request.snapshot_id] = (
                        request,
                        arrived_at,
                        None if target_domain is None else int(target_domain),
                        prompt_tokens,
                        workset_lease,
                    )
                direct_slots_used += 1
                continue
            if manifest.state is not SnapshotState.DIRECT_READY:
                broker.request_release(request.snapshot_id, workset_lease)
                continue
            if tp_size > 1:
                # TP0 owns admission order and publishes the grant before any
                # model-scheduler interaction.  All ranks then start their
                # physical shards from their independent progress workers.
                active_item = (
                    request,
                    arrived_at,
                    None if target_domain is None else int(target_domain),
                    prompt_tokens,
                    workset_lease,
                )
                with poll_lock:
                    if request.snapshot_id in self.agentic_early_direct_terminal:
                        broker.request_release(request.snapshot_id, workset_lease)
                        continue
                    self.agentic_tp_direct_admission_active[request.snapshot_id] = (
                        active_item
                    )
                direct_slots_used += 1
                try:
                    self.agentic_tp_direct_mailbox.publish_receipt(
                        request.snapshot_id, 1
                    )
                except Exception:
                    # No rank may start before the receipt is visible.  Undo
                    # the logical reservation and retain the exact arrival in
                    # FIFO order instead of stranding Direct page credit.
                    logger.exception(
                        "AgenticKV failed to publish TP Direct grant snapshot=%s",
                        request.snapshot_id,
                    )
                    with poll_lock:
                        if (
                            self.agentic_tp_direct_admission_active.get(
                                request.snapshot_id
                            )
                            == active_item
                        ):
                            self.agentic_tp_direct_admission_active.pop(
                                request.snapshot_id, None
                            )
                            direct_slots_used -= 1
                        queue.appendleft((request, payload, manifest))
                        pending.add(request.snapshot_id)
                continue

            if self._agentic_start_early_direct_receive(
                request,
                manifest,
                snapshot_store,
                arrived_at=arrived_at,
                prefill_domain=(None if target_domain is None else int(target_domain)),
                workset_lease=workset_lease,
            ):
                direct_slots_used += 1
                continue

            # Credit exhaustion and transient bootstrap setup both leave the
            # manifest DIRECT_READY. Requeue only while D still offers it.
            current = snapshot_store.load(request, require_ready=False)
            if current is not None and current.state is SnapshotState.DIRECT_READY:
                with poll_lock:
                    queue.append((request, payload, current))
                    pending.add(request.snapshot_id)
            else:
                # Validation can reject a Direct attempt before begin_io_attempt
                # mutates the lease.  Once D no longer offers DIRECT_READY,
                # there is no future queue entry that could own that grant.
                self.agentic_p_workset_broker.cancel_unstarted(
                    request.snapshot_id,
                    owner=workset_owner,
                )

    def _agentic_poll_early_direct_receives_once(
        self, now: Optional[float] = None
    ) -> None:
        """Discover arrival markers and progress async reverse transfers."""

        marker_store = getattr(self, "agentic_early_claim_store", None)
        runtime = getattr(self, "agentic_direct_runtime", None)
        if marker_store is None or runtime is None:
            return
        now = time.monotonic() if now is None else now
        snapshot_store = self._agentic_snapshot_store()
        if snapshot_store is None:
            return

        direct_timeout = max(0.1, envs.SGLANG_AGENTIC_KV_DIRECT_HANDSHAKE_TIMEOUT.get())
        bind_timeout = max(
            direct_timeout,
            envs.SGLANG_AGENTIC_KV_READY_TIMEOUT.get(),
            120.0,
        )
        poll_lock = getattr(self, "agentic_early_direct_poll_lock", nullcontext())
        # Event ingestion and admission precede transport completion work, so
        # an older completion burst cannot consume a fast tool's two-second
        # claim window.
        self._agentic_collect_direct_arrivals(poll_lock)
        self._agentic_collect_direct_abort_events()
        self._agentic_admit_queued_direct_receives(
            snapshot_store, direct_timeout, poll_lock
        )
        if getattr(self, "tp_size", 1) > 1:
            self._agentic_progress_tp_direct_grants(snapshot_store)
        with poll_lock:
            receive_entries = tuple(self.agentic_early_direct_receives.items())

        # NIXL notifications are manager-wide. Polling every active receiver
        # separately drains and parses the same notification queue once per
        # request, which becomes expensive during a Direct burst. Reuse the
        # transport's batch API so each manager is progressed once per cycle;
        # receivers without a batch API retain their original behavior.
        batched_groups = {}
        for snapshot_id, entry in receive_entries:
            if entry.completed_at is not None or entry.transport_poll in {
                KVPoll.Success,
                KVPoll.Failed,
            }:
                continue
            poll_many = getattr(
                type(entry.receiver),
                "poll_many_agentic",
                getattr(type(entry.receiver), "poll_many", None),
            )
            if not callable(poll_many):
                continue
            group_key = (
                type(entry.receiver),
                id(getattr(entry.receiver, "kv_mgr", None)),
            )
            batched_groups.setdefault(group_key, []).append(
                (snapshot_id, entry, poll_many)
            )

        batched_polls = {}
        for grouped_entries in batched_groups.values():
            batch_started = time.monotonic()
            direct_requested = getattr(
                self, "agentic_direct_poll_requested", nullcontext()
            )
            nixl_lock = getattr(self, "agentic_nixl_control_lock", nullcontext())
            try:
                if hasattr(direct_requested, "set"):
                    direct_requested.set()
                with nixl_lock:
                    polls = grouped_entries[0][2](
                        [entry.receiver for _, entry, _ in grouped_entries]
                    )
                if len(polls) != len(grouped_entries):
                    raise RuntimeError(
                        "Direct transport batch poll returned the wrong result count"
                    )
            except Exception:
                logger.exception(
                    "Early Direct batch poll failed for %d receivers",
                    len(grouped_entries),
                )
                # A control-plane exception is not a DMA completion fence.
                # Retain and retry every receiver instead of reusing pages
                # that a remote WRITE may still target.
                polls = [KVPoll.WaitingForInput] * len(grouped_entries)
            finally:
                if hasattr(direct_requested, "clear"):
                    direct_requested.clear()
            batch_elapsed = time.monotonic() - batch_started
            if batch_elapsed >= 0.25:
                logger.warning(
                    "Agentic P Direct batch poll slow elapsed_ms=%.3f "
                    "active_receivers=%d",
                    batch_elapsed * 1000.0,
                    len(grouped_entries),
                )
            for (snapshot_id, entry, _), poll in zip(grouped_entries, polls):
                batched_polls[snapshot_id] = (entry, poll)

        # A burst can complete dozens of rooms together. Ledger publication,
        # route publication and receiver teardown are request-local but not
        # free; processing the whole burst before the next arrival scan caused
        # multi-second admission gaps. Time-slice terminal bookkeeping while
        # continuing to poll every transport room each cycle.
        terminal_commit_budget = 8
        terminal_commits = 0
        for snapshot_id, entry in receive_entries:
            if entry.completed_at is not None:
                if (
                    self.tp_size == 1
                    and entry.prefill_domain is not None
                    and not entry.route_published
                ):
                    try:
                        marker_store.publish_route(
                            entry.request,
                            route="direct_complete",
                            prefill_domain=entry.prefill_domain,
                            snapshot_tokens=entry.manifest.token_count,
                        )
                        entry.route_published = True
                    except OSError:
                        logger.exception(
                            "Failed to publish Direct route for %s", snapshot_id
                        )
                if (
                    now - entry.completed_at >= bind_timeout
                    and not getattr(entry, "bind_wait_warned", False)
                ):
                    # The P workset is now the complete authoritative copy.
                    # Request binding latency is not a validity deadline.
                    entry.bind_wait_warned = True
                    logger.warning(
                        "AgenticKV direct_bind_wait snapshot=%s waited_s=%.3f "
                        "action=retain_parent",
                        snapshot_id,
                        now - entry.completed_at,
                    )
                continue
            try:
                poll = entry.transport_poll
                if poll not in {KVPoll.Success, KVPoll.Failed}:
                    # NIXL polling can occasionally take seconds under a
                    # burst. Never hold the state lock across transport calls;
                    # the scheduler needs it to inspect completed entries.
                    batched = batched_polls.get(snapshot_id)
                    if batched is not None and batched[0] is entry:
                        poll = batched[1]
                    else:
                        poll_agentic = getattr(
                            entry.receiver, "poll_agentic", entry.receiver.poll
                        )
                        poll = poll_agentic()
                    with poll_lock:
                        if (
                            self.agentic_early_direct_receives.get(snapshot_id)
                            is not entry
                        ):
                            continue
                        entry.transport_poll = poll
            except Exception:
                logger.exception("Early Direct D->P receive failed for %s", snapshot_id)
                # Retrying is safe; treating a Python/NIXL polling exception
                # as terminal is not, because the remote WRITE may continue.
                poll = KVPoll.WaitingForInput
            if self._agentic_handle_unstarted_direct_abort(
                entry,
                snapshot_store,
                marker_store,
                poll,
                direct_timeout,
            ):
                continue
            if poll in {KVPoll.Success, KVPoll.Failed}:
                if entry.workset_lease is not None and not entry.io_quiesced:
                    if entry.io_attempt is None or not (
                        self.agentic_p_workset_broker.mark_io_quiesced(
                            snapshot_id,
                            entry.workset_lease,
                            entry.io_attempt,
                        )
                    ):
                        # A terminal notification without the matching
                        # attempt token cannot authorize page reuse or bind.
                        entry.abort_requested = True
                        entry.abort_release_claim = True
                        entry.abort_reason = "direct_io_attempt_mismatch"
                        continue
                    entry.io_quiesced = True
                if entry.abort_requested:
                    self._agentic_drop_early_direct_receive(
                        entry,
                        snapshot_store,
                        release_claim=entry.abort_release_claim,
                        reason=entry.abort_reason or "deferred_abort",
                    )
                    continue
                if terminal_commits >= terminal_commit_budget:
                    continue
                terminal_commits += 1
            if poll == KVPoll.Success:
                try:
                    debug_settle = float(
                        os.getenv(
                            "SGLANG_AGENTIC_KV_DEBUG_RECEIVE_SETTLE_SECONDS",
                            "0",
                        )
                    )
                    if debug_settle > 0:
                        time.sleep(debug_settle)
                        torch.cuda.synchronize()
                    if self.tp_size > 1:
                        # Physical completion is rank-local.  Record it in
                        # memory only; the scheduler's TP status reduction
                        # fences all shards and rank 0 alone commits CONSUMED.
                        completed_at = time.monotonic()
                        entry.receiver.clear()
                        self._agentic_clear_direct_receiver(
                            entry.receiver, entry.manifest
                        )
                        with poll_lock:
                            if (
                                self.agentic_early_direct_receives.get(snapshot_id)
                                is not entry
                            ):
                                continue
                            entry.completed_at = completed_at
                            self.agentic_early_direct_completion_queue.append(
                                snapshot_id
                            )
                        continue

                    current = snapshot_store.load(entry.request, require_ready=False)
                    if current is None:
                        raise SnapshotLifecycleError(
                            "early Direct claim disappeared before completion"
                        )
                    elif current.state in {
                        SnapshotState.P_RECEIVED,
                        SnapshotState.CONSUMED,
                    }:
                        completed = current
                    elif (
                        current.state is SnapshotState.DIRECT_LOADING
                        and current.claim_id == entry.claim_id
                    ):
                        completed = snapshot_store.complete_direct(
                            current, entry.claim_id
                        )
                    else:
                        raise SnapshotLifecycleError(
                            "early Direct group claim changed before completion"
                        )
                    if completed.state not in {
                        SnapshotState.P_RECEIVED,
                        SnapshotState.CONSUMED,
                    }:
                        # This rank's bytes are resident, but the logical
                        # request-generation is not visible until every TP
                        # shard has acknowledged the same claim.
                        continue
                    if entry.prefill_domain is not None:
                        try:
                            marker_store.publish_route(
                                entry.request,
                                route="direct_complete",
                                prefill_domain=entry.prefill_domain,
                                snapshot_tokens=entry.manifest.token_count,
                            )
                            entry.route_published = True
                        except OSError:
                            logger.exception(
                                "Failed to publish Direct route for %s; retrying",
                                snapshot_id,
                            )
                    # Do not launch debug GPU work from the independent
                    # ingress thread. The token digest is validated when the
                    # tokenized Req binds on the scheduler thread.
                    entry.receiver.clear()
                    self._agentic_clear_direct_receiver(entry.receiver, entry.manifest)
                    completed_at = time.monotonic()
                    with poll_lock:
                        if (
                            self.agentic_early_direct_receives.get(snapshot_id)
                            is not entry
                        ):
                            continue
                        entry.completed_at = completed_at
                        self.agentic_early_direct_completion_queue.append(snapshot_id)
                    logger.info(
                        "AgenticKV early_direct_complete snapshot=%s tokens=%d "
                        "transfer_ms=%.3f",
                        snapshot_id,
                        entry.manifest.token_count,
                        (completed_at - entry.started_at) * 1000.0,
                    )
                except Exception:
                    logger.exception(
                        "Could not complete early Direct receive for %s",
                        snapshot_id,
                    )
                    if self.tp_size > 1:
                        self._agentic_mark_tp_direct_failed(
                            entry, reason="completion_failed"
                        )
                    else:
                        self._agentic_drop_early_direct_receive(
                            entry,
                            snapshot_store,
                            release_claim=True,
                            reason="completion_failed",
                        )
            elif poll == KVPoll.Failed:
                if self.tp_size > 1:
                    self._agentic_mark_tp_direct_failed(
                        entry, reason="transfer_failed_or_timeout"
                    )
                else:
                    self._agentic_drop_early_direct_receive(
                        entry,
                        snapshot_store,
                        release_claim=True,
                        reason="transfer_failed_or_timeout",
                    )

        if getattr(self, "tp_size", 1) > 1:
            self._agentic_commit_tp_direct_groups(snapshot_store)

        # Retain short-lived terminal ids only to avoid repeatedly reopening a
        # marker while Decode is about to remove it.
        with poll_lock:
            for snapshot_id, terminal_at in tuple(
                self.agentic_early_direct_terminal.items()
            ):
                if now - terminal_at >= 10.0:
                    self.agentic_early_direct_terminal.pop(snapshot_id, None)

    def _agentic_bind_early_direct_receive(
        self,
        req: Req,
        request: RequestGeneration,
        *,
        allow_tp_commit: bool = True,
    ) -> Optional[bool]:
        """Bind already-received KV to the real Req; return defer decision."""

        receives = getattr(self, "agentic_early_direct_receives", None)
        snapshot_id = getattr(request, "snapshot_id", None)
        if not receives or snapshot_id is None:
            return None
        with getattr(self, "agentic_early_direct_poll_lock", nullcontext()):
            entry = receives.get(snapshot_id)
        if entry is None:
            return None
        req._agentic_kv_queue_class = "fast"
        if entry.completed_at is None:
            return True

        tp_size = getattr(self, "tp_size", 1)
        marker_store = None
        if tp_size == 1 and entry.prepared_req is req and entry.radix_prepared:
            if entry.workset_lease is None:
                return self._agentic_admit_early_direct_bind(
                    req,
                    request,
                    entry,
                    tp_size=tp_size,
                    marker_store=marker_store,
                )
            return self._agentic_try_finalize_early_direct_bind(
                req,
                request,
                entry,
                existing_tokens=entry.existing_tokens,
                tp_size=tp_size,
                marker_store=marker_store,
                admit=True,
            )
        if tp_size > 1:
            direct_actions = getattr(self, "_agentic_tp_direct_actions", {})
            action = direct_actions.get(request.snapshot_id)
            if action == "commit_bind":
                if entry.prepared_req is not req:
                    return True
                return self._agentic_admit_early_direct_bind(
                    req,
                    request,
                    entry,
                    tp_size=tp_size,
                    marker_store=marker_store,
                )
            if action != "prepare_bind":
                return True
            if entry.prepared_req is req:
                return self._agentic_try_finalize_early_direct_bind(
                    req,
                    request,
                    entry,
                    existing_tokens=entry.existing_tokens,
                    tp_size=tp_size,
                    marker_store=marker_store,
                    admit=False,
                )

        if entry.device_indices is None or entry.workset_lease is None:
            raise RuntimeError("completed Direct receive lost its workset lease")

        # The independent ingress worker deliberately avoids launching GPU
        # diagnostics.  Run the opt-in exact byte digest here, on the
        # scheduler thread, before the restored parent enters model work.
        direct_runtime = getattr(self, "agentic_direct_runtime", None)
        restored_digest = (
            None
            if direct_runtime is None
            else debug_kv_digest(direct_runtime.kv_pool, entry.device_indices)
        )
        if restored_digest is not None:
            logger.info(
                "AgenticKV p_restored_digest snapshot=%s digest=%s",
                request.snapshot_id,
                restored_digest,
            )

        if entry.workset_lease.state_device_indices:
            from sglang.srt.disaggregation.agentic_hybrid_transfer import log_mamba_digest
            log_mamba_digest(self.req_to_token_pool, entry.workset_lease.state_device_indices,
                             phase="d2p_direct_received", snapshot_id=request.snapshot_id)
        parent_tokens = req.origin_input_ids[: entry.manifest.token_count]
        if (
            len(parent_tokens) != entry.manifest.token_count
            or token_ids_digest(parent_tokens) != entry.manifest.token_digest
        ):
            if tp_size > 1:
                return self._agentic_fail_tp_direct_bind(
                    entry,
                    req,
                    reason="token_digest_mismatch",
                )
            snapshot_store = self._agentic_snapshot_store()
            try:
                current = snapshot_store.load(request, require_ready=False)
                if current is not None and current.state is not SnapshotState.FAILED:
                    snapshot_store.mark_failed(
                        current,
                        reason="permanent_token_digest_mismatch",
                        owner_claim_id=entry.claim_id,
                    )
            except Exception:
                logger.exception(
                    "AgenticKV permanent Direct failure commit retry "
                    "snapshot=%s",
                    request.snapshot_id,
                )
                return True
            self._agentic_drop_early_direct_receive(
                entry,
                snapshot_store,
                release_claim=False,
                reason="token_digest_mismatch",
            )
            req._agentic_kv_gate_complete = True
            req._agentic_kv_fallback = "early_direct_token_digest_mismatch"
            return False
        if len(req.origin_input_ids) > entry.workset_lease.prompt_tokens:
            if tp_size > 1:
                return self._agentic_fail_tp_direct_bind(
                    entry, req, reason="workset_marker_underestimated"
                )
            return self._agentic_return_early_direct_to_slow(
                entry, req, reason="workset_marker_underestimated"
            )

        if not self.agentic_p_workset_broker.begin_bind(
            request.snapshot_id, entry.workset_lease
        ):
            if tp_size > 1:
                return self._agentic_fail_tp_direct_bind(
                    entry,
                    req,
                    reason="workset_ownership_lost",
                )
            return self._agentic_return_early_direct_to_slow(
                entry, req, reason="workset_ownership_lost"
            )

        # Record ownership before the first Radix mutation so every exception
        # path can remove the exact request-generation branch.
        entry.prepared_req = req
        req._agentic_direct_parent_token_count = len(parent_tokens)
        try:
            result = self.tree_cache.insert(
                InsertParams(
                    key=RadixKey(parent_tokens, req.extra_key),
                    value=entry.device_indices,
                    mamba_value=(entry.workset_lease.state_device_indices[0].clone()
                                 if entry.workset_lease.state_device_indices else None),
                    priority=getattr(req, "priority", 0) or 0,
                )
            )
        except Exception:
            self.agentic_p_workset_broker.abort_bind(
                request.snapshot_id,
                entry.workset_lease,
                parent_bound=False,
            )
            # insert() raised before a Radix branch existed.  Do not let the
            # TP-wide rollback path call release_agentic_request_cache() for
            # an unrelated/native prefix merely because the Req object was
            # recorded before the attempted mutation.
            entry.prepared_req = None
            entry.radix_prepared = False
            entry.existing_tokens = 0
            if hasattr(req, "_agentic_direct_parent_token_count"):
                delattr(req, "_agentic_direct_parent_token_count")
            logger.exception("Failed to bind early Direct KV for %s", req.rid)
            if tp_size > 1:
                return self._agentic_fail_tp_direct_bind(
                    entry,
                    req,
                    reason="radix_insert_failed",
                )
            return self._agentic_return_early_direct_to_slow(
                entry, req, reason="radix_insert_failed"
            )
        self.agentic_p_workset_broker.commit_parent_bound(
            request.snapshot_id, entry.workset_lease,
            state_donated_to_radix=(bool(entry.workset_lease.state_device_indices) and not result.mamba_exist),
            state_duplicate=(bool(entry.workset_lease.state_device_indices) and result.mamba_exist),
        )
        if entry.workset_lease.runtime_state_device_indices:
            self.agentic_p_workset_broker.attach_runtime_state_for_bind(
                request.snapshot_id, req, entry.workset_lease
            )
        # insert() makes the restored parent visible to the Radix LRU.  Pin the
        # exact request-generation before returning to the queue.  Native
        # Prefill acquires its ordinary request lock first and only then drops
        # this temporary pin, so there is no evictable gap between ownerships.
        try:
            parent_match = self.tree_cache.match_prefix(
                MatchPrefixParams(
                    key=RadixKey(parent_tokens, req.extra_key),
                    req=req,
                )
            )
            if len(parent_match.device_indices) != len(parent_tokens):
                raise RuntimeError(
                    "Early Direct parent disappeared before request protection"
                )
            self.tree_cache.inc_lock_ref(parent_match.last_device_node)
            req._agentic_direct_parent_pin_node = parent_match.last_device_node
            req._agentic_direct_parent_token_count = len(parent_tokens)
            entry.radix_prepared = True
            entry.existing_tokens = int(result.prefix_len)
        except Exception:
            logger.exception("Failed to prepare Direct bind for %s", req.rid)
            if tp_size > 1:
                return self._agentic_fail_tp_direct_bind(
                    entry,
                    req,
                    reason="radix_prepare_failed",
                )
            release = getattr(self.tree_cache, "release_agentic_request_cache", None)
            if release is not None:
                release(
                    req,
                    committed_len=len(parent_tokens),
                    _defer_if_blocked=False,
                )
            self.agentic_p_workset_broker.abort_bind(
                request.snapshot_id,
                entry.workset_lease,
                parent_bound=True,
            )
            return self._agentic_return_early_direct_to_slow(
                entry, req, reason="radix_prepare_failed"
            )
        return self._agentic_try_finalize_early_direct_bind(
            req,
            request,
            entry,
            existing_tokens=entry.existing_tokens,
            tp_size=tp_size,
            marker_store=marker_store,
            admit=tp_size == 1,
        )

    def _agentic_try_finalize_early_direct_bind(
        self,
        req: Req,
        request: RequestGeneration,
        entry: AgenticEarlyDirectReceive,
        *,
        existing_tokens: int,
        tp_size: int,
        marker_store,
        admit: bool = True,
    ) -> bool:
        """Retry final admission without unwinding a pinned Radix parent."""

        try:
            return self._agentic_finalize_early_direct_bind(
                req,
                request,
                entry,
                existing_tokens=existing_tokens,
                tp_size=tp_size,
                marker_store=marker_store,
                admit=admit,
            )
        except Exception:
            logger.exception(
                "AgenticKV direct_finalize_retry snapshot=%s req=%s",
                request.snapshot_id,
                req.rid,
            )
            return True

    def _agentic_finalize_early_direct_bind(
        self,
        req: Req,
        request: RequestGeneration,
        entry: AgenticEarlyDirectReceive,
        *,
        existing_tokens: int,
        tp_size: int,
        marker_store,
        admit: bool = True,
    ) -> bool:
        """Commit one prepared Direct shard after every TP rank is ready."""

        if existing_tokens:
            # A trajectory-unique extra_key normally makes this zero.  Keep
            # duplicate-prefix handling correct with ordinary allocator pages.
            entry.existing_tokens = 0
            # MambaRadixCache.insert already frees duplicate Attention pages.
            if not self.tree_cache.supports_mamba():
                self.token_to_kv_pool_allocator.free(entry.device_indices[:existing_tokens])
        if admit:
            self.agentic_p_workset_broker.handoff_to_req(
                request.snapshot_id, req, entry.workset_lease
            )
            entry.workset_lease = None
        logger.info(
            "AgenticKV early_direct_bind snapshot=%s tokens=%d existing_tokens=%d "
            "arrival_to_bind_ms=%.3f workset_committed=true req=%s",
            request.snapshot_id,
            entry.manifest.token_count,
            existing_tokens,
            max(0.0, (time.time() - entry.arrived_at) * 1000.0),
            req.rid,
        )
        if not admit:
            entry.prepared_req = req
            self.agentic_tp_direct_mailbox.publish_local_progress(
                request.snapshot_id, 4
            )
            logger.info(
                "AgenticKV early_direct_bind_prepared snapshot=%s req=%s",
                request.snapshot_id,
                req.rid,
            )
            return True
        return self._agentic_admit_early_direct_bind(
            req,
            request,
            entry,
            tp_size=tp_size,
            marker_store=marker_store,
        )

    def _agentic_admit_early_direct_bind(
        self,
        req: Req,
        request: RequestGeneration,
        entry: AgenticEarlyDirectReceive,
        *,
        tp_size: int,
        marker_store,
    ) -> bool:
        """Expose one already group-committed Direct parent to Prefill."""

        if tp_size == 1:
            snapshot_store = self._agentic_snapshot_store()
            try:
                current = snapshot_store.load(request, require_ready=False)
                if current is None:
                    raise SnapshotNotReadyError(
                        f"Direct bind manifest is not visible for "
                        f"{request.snapshot_id}"
                    )
                if current.state is SnapshotState.P_RECEIVED:
                    snapshot_store.commit_direct_bound(current, entry.claim_id)
                elif current.state is not SnapshotState.CONSUMED:
                    raise SnapshotNotReadyError(
                        f"Direct bind observed {current.state.value} for "
                        f"{request.snapshot_id}"
                    )
            except Exception:
                # Radix is already inserted and pinned and the complete
                # workset lease remains owned by this entry.  Retry only the
                # idempotent lifecycle ACK; never roll back to recompute.
                logger.exception(
                    "AgenticKV direct_bind_commit_retry snapshot=%s req=%s",
                    request.snapshot_id,
                    req.rid,
                )
                return True

        if getattr(entry, "workset_lease", None) is not None:
            self.agentic_p_workset_broker.handoff_to_req(
                request.snapshot_id, req, entry.workset_lease
            )
            entry.workset_lease = None

        with getattr(self, "agentic_early_direct_poll_lock", nullcontext()):
            if self.agentic_early_direct_receives.get(request.snapshot_id) is entry:
                self.agentic_early_direct_receives.pop(request.snapshot_id, None)
            self.agentic_early_direct_terminal[request.snapshot_id] = time.monotonic()
        req._agentic_kv_gate_complete = True
        req._agentic_kv_direct_hit_tokens = entry.manifest.token_count
        entry.prepared_req = None
        entry.radix_prepared = False
        entry.existing_tokens = 0
        if tp_size > 1:
            self.agentic_tp_direct_local_admitted.add(request.snapshot_id)
        logger.info(
            "AgenticKV early_direct_admit snapshot=%s tokens=%d req=%s",
            request.snapshot_id,
            entry.manifest.token_count,
            req.rid,
        )
        return False

    def _agentic_rollback_prepared_direct_bind(
        self, entry: AgenticEarlyDirectReceive
    ) -> None:
        """Remove one locally prepared TP Radix branch before group abort."""

        req = entry.prepared_req
        if req is None:
            return
        pin = getattr(req, "_agentic_direct_parent_pin_node", None)
        if pin is not None:
            self.tree_cache.dec_lock_ref(pin)
            del req._agentic_direct_parent_pin_node
        committed_len = int(getattr(req, "_agentic_direct_parent_token_count", 0))
        release = getattr(self.tree_cache, "release_agentic_request_cache", None)
        if committed_len and release is not None:
            release(
                req,
                committed_len=committed_len,
                _defer_if_blocked=False,
            )
        if getattr(entry, "workset_lease", None) is not None:
            self.agentic_p_workset_broker.abort_bind(
                entry.request.snapshot_id,
                entry.workset_lease,
                parent_bound=True,
            )
        entry.prepared_req = None
        entry.radix_prepared = False
        entry.existing_tokens = 0

    def _agentic_start_direct_load(self, req: Req, snapshot_store, manifest) -> bool:
        runtime = getattr(self, "agentic_direct_runtime", None)
        if runtime is None or getattr(self.tree_cache, "is_eagle", False):
            return False
        if self.tp_size > 1:
            # TP Direct is admitted before the tokenized Req through the
            # group-atomic early receiver.  The legacy request-bound loader
            # inserts one rank into Radix before its peers finish and cannot
            # safely roll that partial prefix back after a peer failure.
            req._agentic_direct_disabled = True
            return False
        if manifest.tp_size != self.tp_size or (
            manifest.kv_layout_hash and manifest.kv_layout_hash != runtime.layout_hash
        ):
            req._agentic_kv_fallback = "direct_tp_layout_mismatch"
            logger.error(
                "AgenticKV Direct layout mismatch snapshot=%s "
                "source_tp=%d destination_tp=%d source_layout=%s destination_layout=%s",
                manifest.snapshot_id,
                manifest.tp_size,
                self.tp_size,
                manifest.kv_layout_hash,
                runtime.layout_hash,
            )
            if manifest.state is SnapshotState.DIRECT_READY:
                try:
                    snapshot_store.fail_direct_offer(
                        manifest,
                        owner_id=f"p-legacy-incompatible:{os.getpid()}",
                        reason="permanent_direct_layout_mismatch",
                    )
                except Exception:
                    logger.exception(
                        "AgenticKV permanent legacy layout mismatch commit retry "
                        "snapshot=%s",
                        manifest.snapshot_id,
                    )
                    return True
            return False
        parent_tokens = req.origin_input_ids[: manifest.token_count]
        if (
            len(parent_tokens) != manifest.token_count
            or token_ids_digest(parent_tokens) != manifest.token_digest
        ):
            req._agentic_kv_fallback = "direct_token_digest_mismatch"
            if manifest.state is SnapshotState.DIRECT_READY:
                try:
                    snapshot_store.fail_direct_offer(
                        manifest,
                        owner_id=f"p-legacy-incompatible:{os.getpid()}",
                        reason="permanent_token_digest_mismatch",
                    )
                except Exception:
                    logger.exception(
                        "AgenticKV permanent legacy digest mismatch commit retry "
                        "snapshot=%s",
                        manifest.snapshot_id,
                    )
                    return True
            return False

        workset_owner = AgenticPWorksetLeaseBroker.direct_owner(manifest.snapshot_id)
        self.agentic_p_workset_broker.request(
            manifest.snapshot_id,
            manifest.token_count,
            len(req.origin_input_ids),
            owner=workset_owner,
        )
        workset_lease = self.agentic_p_workset_broker.get(
            manifest.snapshot_id, owner=workset_owner
        )
        if workset_lease is None:
            # The scheduler services the physical intent on its next safe
            # boundary.  D retains source KV and may independently time out to
            # Slow if a complete workset cannot be granted in time.
            return True
        device_indices = workset_lease.parent_indices[: manifest.token_count]

        claim_id = f"direct-p:{req.rid}"
        if not self.agentic_p_workset_broker.begin_io_attempt(
            manifest.snapshot_id, workset_lease, claim_id
        ):
            # Another concrete Direct attempt owns this destination.  This
            # compatibility path must not claim lifecycle or recycle pages
            # belonging to that attempt.
            return True
        receiver = None
        claimed = None
        try:
            claimed = snapshot_store.claim_direct(manifest.request, claim_id)
            if not runtime.manager.try_ensure_parallel_info(
                claimed.direct_bootstrap_addr
            ):
                raise SnapshotNotReadyError("reverse bootstrap is not ready")
            receiver = runtime.receiver_class(
                mgr=runtime.manager,
                bootstrap_addr=claimed.direct_bootstrap_addr,
                bootstrap_room=claimed.direct_room,
            )
            receiver.init(prefill_dp_rank=0)
            if receiver.poll() == KVPoll.Failed:
                raise SnapshotLifecycleError("reverse receiver init failed")
            self.agentic_p_workset_broker.mark_io_inflight(
                manifest.snapshot_id, workset_lease, claim_id
            )
            if workset_lease.state_device_indices:
                from sglang.srt.disaggregation.agentic_hybrid_transfer import submit_reverse_receive
                submit_reverse_receive(receiver, workset_lease, ("mamba",))
            else:
                receiver.send_metadata(workset_lease.parent_page_indices, aux_index=0)
        except Exception as exc:
            transport_may_write = bool(
                receiver is not None
                and getattr(receiver, "started_transfer", False)
                and workset_lease.io_attempt == claim_id
                and workset_lease.state in {"io_inflight", "release_pending"}
            )
            if transport_may_write:
                self.agentic_p_workset_broker.request_release(
                    manifest.snapshot_id,
                    workset_lease,
                    io_attempt=claim_id,
                )
                entry = AgenticEarlyDirectReceive(
                    request=manifest.request,
                    manifest=claimed if claimed is not None else manifest,
                    claim_id=claim_id,
                    receiver=receiver,
                    device_indices=device_indices,
                    started_at=time.monotonic(),
                    arrived_at=time.time(),
                    workset_lease=workset_lease,
                    io_attempt=claim_id,
                    abort_requested=True,
                    abort_release_claim=True,
                    abort_reason="request_bound_metadata_failed",
                )
                with getattr(self, "agentic_early_direct_poll_lock", nullcontext()):
                    self.agentic_early_direct_receives[manifest.snapshot_id] = entry
                req._agentic_direct_disabled = True
                logger.exception(
                    "Request-bound Direct metadata may be partially visible; "
                    "quarantining workset snapshot=%s",
                    manifest.snapshot_id,
                )
                return True
            if (
                workset_lease.state in {"io_inflight", "release_pending"}
                and workset_lease.io_attempt == claim_id
            ):
                self.agentic_p_workset_broker.mark_io_quiesced(
                    manifest.snapshot_id, workset_lease, claim_id
                )
            else:
                self.agentic_p_workset_broker.cancel_io_attempt(
                    manifest.snapshot_id, workset_lease, claim_id
                )
            self.agentic_p_workset_broker.request_release(
                manifest.snapshot_id, workset_lease
            )
            # The producer will cross the fast-tool threshold and publish a
            # complete Host snapshot.  Retrying a failed metadata transition
            # every scheduler pass only churns Mooncake and P HBM.
            req._agentic_direct_disabled = True
            current = snapshot_store.load(manifest.request, require_ready=False)
            if (
                current is not None
                and current.state is SnapshotState.DIRECT_LOADING
                and current.claim_id == claim_id
            ):
                try:
                    snapshot_store.release_direct_claim(current, claim_id)
                except Exception:
                    logger.exception("Failed to release direct claim for %s", req.rid)
            if isinstance(exc, SnapshotNotReadyError):
                # This is normal contention at the fast-window boundary: D
                # acquired the fallback claim first or the offer already
                # advanced.  The request will rematch the Mooncake path.
                logger.info(
                    "AgenticKV direct_claim_missed req=%s reason=%s",
                    req.rid,
                    exc,
                )
            else:
                logger.exception("Could not start direct D->P load for %s", req.rid)
            return True

        req._agentic_direct_receiver = receiver
        req._agentic_direct_indices = device_indices
        req._agentic_direct_workset_lease = workset_lease
        req._agentic_direct_manifest = claimed
        req._agentic_direct_claim_id = claim_id
        req._agentic_direct_io_attempt = claim_id
        req._agentic_kv_snapshot_store = snapshot_store
        req._agentic_direct_started_at = time.monotonic()
        logger.info(
            "AgenticKV direct_load_start snapshot=%s tokens=%d req=%s",
            claimed.snapshot_id,
            claimed.token_count,
            req.rid,
        )
        return True

    def _agentic_return_request_direct_to_decode(
        self, req: Req, *, reason: str
    ) -> bool:
        """Return a quiescent, unbound Direct attempt to D for Slow staging.

        This never admits the child for recompute.  If the lifecycle CAS is
        temporarily unavailable, P keeps its complete workset and retries.
        """

        snapshot_store = req._agentic_kv_snapshot_store
        manifest = req._agentic_direct_manifest
        claim_id = req._agentic_direct_claim_id
        current = snapshot_store.load(manifest.request, require_ready=False)
        try:
            if current is not None and current.claim_id == claim_id:
                if current.state is SnapshotState.P_RECEIVED:
                    snapshot_store.release_received_direct(current, claim_id)
                elif current.state is SnapshotState.DIRECT_LOADING:
                    snapshot_store.release_direct_claim(current, claim_id)
                elif current.state is not SnapshotState.DIRECT_READY:
                    return False
        except Exception:
            logger.exception(
                "AgenticKV direct_return_retry snapshot=%s req=%s reason=%s",
                manifest.snapshot_id,
                req.rid,
                reason,
            )
            return False

        receiver = req._agentic_direct_receiver
        receiver.clear()
        self._agentic_clear_direct_receiver(receiver, manifest)
        self.agentic_p_workset_broker.request_release(
            manifest.snapshot_id,
            req._agentic_direct_workset_lease,
        )
        for name in (
            "_agentic_direct_receiver",
            "_agentic_direct_indices",
            "_agentic_direct_manifest",
            "_agentic_direct_claim_id",
            "_agentic_direct_io_attempt",
            "_agentic_direct_started_at",
            "_agentic_direct_workset_lease",
            "_agentic_direct_radix_bound",
            "_agentic_direct_rank_received",
        ):
            if hasattr(req, name):
                delattr(req, name)
        req._agentic_direct_disabled = True
        logger.warning(
            "AgenticKV direct_return_to_slow snapshot=%s req=%s reason=%s",
            manifest.snapshot_id,
            req.rid,
            reason,
        )
        return True

    def _agentic_poll_direct_load(self, req: Req) -> bool:
        """Advance request-owned Direct receive without losing its D fallback.

        The early-Direct worker is the normal production path.  This legacy
        request-owned path remains for deployments without arrival markers,
        so it follows the same two-phase rule: transport, Radix bind, then
        lifecycle commit.  Every phase is explicitly retryable.
        """

        receiver = getattr(req, "_agentic_direct_receiver", None)
        if receiver is None:
            return False
        if getattr(req, "_agentic_direct_radix_bound", False):
            snapshot_store = req._agentic_kv_snapshot_store
            manifest = req._agentic_direct_manifest
            current = snapshot_store.load(manifest.request, require_ready=False)
            try:
                if (
                    current is not None
                    and current.state is SnapshotState.DIRECT_LOADING
                ):
                    current = snapshot_store.complete_direct(
                        current, req._agentic_direct_claim_id
                    )
                if current is not None and current.state is SnapshotState.P_RECEIVED:
                    current = snapshot_store.commit_direct_bound(
                        current, req._agentic_direct_claim_id
                    )
            except Exception:
                logger.exception(
                    "AgenticKV direct_bind_commit_retry snapshot=%s req=%s",
                    manifest.snapshot_id,
                    req.rid,
                )
                return True
            if current is None or current.state is not SnapshotState.CONSUMED:
                return True
            self.agentic_p_workset_broker.handoff_to_req(
                manifest.snapshot_id,
                req,
                req._agentic_direct_workset_lease,
            )
            req._agentic_kv_gate_complete = True
            req._agentic_kv_direct_hit_tokens = manifest.token_count
            for name in (
                "_agentic_direct_receiver",
                "_agentic_direct_indices",
                "_agentic_direct_manifest",
                "_agentic_direct_claim_id",
                "_agentic_direct_io_attempt",
                "_agentic_direct_started_at",
                "_agentic_direct_workset_lease",
                "_agentic_direct_radix_bound",
                "_agentic_direct_rank_received",
            ):
                if hasattr(req, name):
                    delattr(req, name)
            return False
        if getattr(req, "_agentic_direct_rank_received", False):
            snapshot_store = req._agentic_kv_snapshot_store
            manifest = req._agentic_direct_manifest
            current = snapshot_store.load(manifest.request, require_ready=False)
            if current is not None and current.state is SnapshotState.P_RECEIVED:
                try:
                    current = snapshot_store.commit_direct_bound(
                        current, req._agentic_direct_claim_id
                    )
                except Exception:
                    logger.exception(
                        "AgenticKV direct_bind_commit_retry snapshot=%s req=%s",
                        manifest.snapshot_id,
                        req.rid,
                    )
                    return True
            if current is None or current.state is not SnapshotState.CONSUMED:
                return True
            self.agentic_p_workset_broker.handoff_to_req(
                manifest.snapshot_id,
                req,
                req._agentic_direct_workset_lease,
            )
            req._agentic_kv_gate_complete = True
            req._agentic_kv_direct_hit_tokens = manifest.token_count
            for name in (
                "_agentic_direct_receiver",
                "_agentic_direct_indices",
                "_agentic_direct_manifest",
                "_agentic_direct_claim_id",
                "_agentic_direct_io_attempt",
                "_agentic_direct_started_at",
                "_agentic_direct_rank_received",
                "_agentic_direct_workset_lease",
                "_agentic_direct_radix_bound",
                "_agentic_direct_rank_received",
            ):
                if hasattr(req, name):
                    delattr(req, name)
            return False
        try:
            poll_agentic = getattr(receiver, "poll_agentic", receiver.poll)
            poll = poll_agentic()
        except Exception:
            logger.exception("Direct D->P receive failed for %s", req.rid)
            poll = KVPoll.WaitingForInput
        if poll not in {KVPoll.Success, KVPoll.Failed}:
            return True

        snapshot_store = req._agentic_kv_snapshot_store
        manifest = req._agentic_direct_manifest
        claim_id = req._agentic_direct_claim_id
        device_indices = req._agentic_direct_indices
        io_attempt = req._agentic_direct_io_attempt
        if not self.agentic_p_workset_broker.mark_io_quiesced(
            manifest.snapshot_id,
            req._agentic_direct_workset_lease,
            io_attempt,
        ):
            # A terminal callback carrying the wrong attempt token has no
            # authority over this destination.  Abort cleanup quarantines the
            # concrete receiver instead of binding or recycling its pages.
            req._agentic_kv_fallback = "direct_io_attempt_mismatch"
            self._agentic_abort_cleanup(req)
            return True
        if poll == KVPoll.Success:
            debug_settle = float(
                os.getenv("SGLANG_AGENTIC_KV_DEBUG_RECEIVE_SETTLE_SECONDS", "0")
            )
            if debug_settle > 0:
                time.sleep(debug_settle)
                torch.cuda.synchronize()
            restored_digest = debug_kv_digest(
                self.agentic_direct_runtime.kv_pool, device_indices
            )
            if restored_digest is not None:
                logger.info(
                    "AgenticKV p_restored_digest snapshot=%s digest=%s",
                    manifest.snapshot_id,
                    restored_digest,
                )
            keys = req.origin_input_ids[: manifest.token_count]
            workset_lease = req._agentic_direct_workset_lease
            if not self.agentic_p_workset_broker.begin_bind(
                manifest.snapshot_id, workset_lease
            ):
                self._agentic_return_request_direct_to_decode(
                    req, reason="workset_ownership_lost"
                )
                return True
            inserted = False
            req._agentic_direct_parent_token_count = len(keys)
            try:
                result = self.tree_cache.insert(
                    InsertParams(
                        key=RadixKey(keys, req.extra_key),
                        value=device_indices,
                        mamba_value=(workset_lease.state_device_indices[0].clone()
                                     if workset_lease.state_device_indices else None),
                        priority=getattr(req, "priority", 0) or 0,
                    )
                )
                inserted = True
                self.agentic_p_workset_broker.commit_parent_bound(
                    manifest.snapshot_id, workset_lease,
                    state_donated_to_radix=(bool(workset_lease.state_device_indices) and not result.mamba_exist),
                    state_duplicate=(bool(workset_lease.state_device_indices) and result.mamba_exist),
                )
                if workset_lease.runtime_state_device_indices:
                    self.agentic_p_workset_broker.attach_runtime_state_for_bind(
                        manifest.snapshot_id, req, workset_lease
                    )
            except Exception:
                self.agentic_p_workset_broker.abort_bind(
                    manifest.snapshot_id,
                    workset_lease,
                    parent_bound=inserted,
                )
                logger.exception(
                    "Failed to insert direct KV for %s; returning parent to Slow",
                    req.rid,
                )
                self._agentic_return_request_direct_to_decode(
                    req, reason="radix_insert_failed"
                )
                return True
            if result.prefix_len and not self.tree_cache.supports_mamba():
                self.token_to_kv_pool_allocator.free(
                    device_indices[: result.prefix_len]
                )
            logger.info(
                "AgenticKV direct_radix_insert snapshot=%s inserted_tokens=%d "
                "existing_tokens=%d extra_key=%s",
                manifest.snapshot_id,
                manifest.token_count - result.prefix_len,
                result.prefix_len,
                req.extra_key,
            )
            try:
                immediate_match = self.tree_cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(keys, req.extra_key), req=req)
                )
                if len(immediate_match.device_indices) != len(keys):
                    raise RuntimeError(
                        "Direct parent disappeared before request protection"
                    )
                self.tree_cache.inc_lock_ref(immediate_match.last_device_node)
                req._agentic_direct_parent_pin_node = immediate_match.last_device_node
            except Exception:
                if inserted:
                    release = getattr(
                        self.tree_cache, "release_agentic_request_cache", None
                    )
                    if release is not None:
                        release(
                            req,
                            committed_len=len(keys),
                            _defer_if_blocked=False,
                        )
                self.agentic_p_workset_broker.abort_bind(
                    manifest.snapshot_id,
                    workset_lease,
                    parent_bound=True,
                )
                logger.exception(
                    "Failed to protect direct KV for %s; returning parent to Slow",
                    req.rid,
                )
                self._agentic_return_request_direct_to_decode(
                    req, reason="radix_prepare_failed"
                )
                return True
            logger.info(
                "AgenticKV direct_radix_verify snapshot=%s device_tokens=%d "
                "host_tokens=%d",
                manifest.snapshot_id,
                len(immediate_match.device_indices),
                immediate_match.host_hit_length,
            )
            req._agentic_direct_radix_bound = True
            current = snapshot_store.load(manifest.request, require_ready=False)
            completed = current
            if current is not None and current.state is SnapshotState.DIRECT_LOADING:
                try:
                    if self.tp_size != 1:
                        raise RuntimeError(
                            "request-owned Direct completion is disabled for TP; "
                            "rank 0 must use the early-Direct group command"
                        )
                    completed = snapshot_store.complete_direct(current, claim_id)
                except Exception:
                    logger.exception(
                        "Direct KV is resident but completion marker failed for %s",
                        req.rid,
                    )
                    # Radix is already bound and D still retains its source.
                    # Retry the metadata transition without discarding either
                    # valid copy.
                    return True
            if completed is None:
                raise SnapshotLifecycleError(
                    f"Direct manifest disappeared for {manifest.snapshot_id}"
                )
            logger.info(
                "AgenticKV direct_rank_complete snapshot=%s tokens=%d "
                "rank=%d/%d group_state=%s req=%s",
                manifest.snapshot_id,
                manifest.token_count,
                self.tp_rank,
                self.tp_size,
                completed.state.value,
                req.rid,
            )
            receiver.clear()
            self._agentic_clear_direct_receiver(receiver, manifest)
            post_clear_match = self.tree_cache.match_prefix(
                MatchPrefixParams(key=RadixKey(keys, req.extra_key), req=req)
            )
            logger.info(
                "AgenticKV direct_post_clear_verify snapshot=%s device_tokens=%d "
                "host_tokens=%d",
                manifest.snapshot_id,
                len(post_clear_match.device_indices),
                post_clear_match.host_hit_length,
            )
            if completed.state is SnapshotState.P_RECEIVED:
                try:
                    completed = snapshot_store.commit_direct_bound(
                        completed, claim_id
                    )
                except Exception:
                    # The Radix branch is already pinned and the workset lease
                    # remains owned.  Retry the idempotent metadata commit on
                    # the next scheduler visit; never convert this to a full
                    # recompute.
                    req._agentic_direct_radix_bound = True
                    logger.exception(
                        "AgenticKV direct_bind_commit_retry snapshot=%s req=%s",
                        manifest.snapshot_id,
                        req.rid,
                    )
                    return True
            if completed.state is not SnapshotState.CONSUMED:
                req._agentic_direct_radix_bound = True
                return True
            req._agentic_kv_gate_complete = True
            req._agentic_kv_direct_hit_tokens = manifest.token_count
            self.agentic_p_workset_broker.handoff_to_req(
                manifest.snapshot_id,
                req,
                req._agentic_direct_workset_lease,
            )
            for name in (
                "_agentic_direct_receiver",
                "_agentic_direct_indices",
                "_agentic_direct_manifest",
                "_agentic_direct_claim_id",
                "_agentic_direct_io_attempt",
                "_agentic_direct_started_at",
                "_agentic_direct_workset_lease",
            ):
                if hasattr(req, name):
                    delattr(req, name)
            return False

        self.agentic_p_workset_broker.request_release(
            manifest.snapshot_id,
            getattr(req, "_agentic_direct_workset_lease", None),
        )
        current = snapshot_store.load(manifest.request, require_ready=False)
        if (
            current is not None
            and current.state is SnapshotState.DIRECT_LOADING
            and current.claim_id == claim_id
        ):
            try:
                snapshot_store.release_direct_claim(current, claim_id)
            except Exception:
                logger.exception(
                    "Failed to release failed direct claim for %s", req.rid
                )
        receiver.clear()
        self._agentic_clear_direct_receiver(receiver, manifest)
        for name in (
            "_agentic_direct_receiver",
            "_agentic_direct_indices",
            "_agentic_direct_manifest",
            "_agentic_direct_claim_id",
            "_agentic_direct_io_attempt",
            "_agentic_direct_started_at",
            "_agentic_direct_workset_lease",
        ):
            if hasattr(req, name):
                delattr(req, name)
        return True

    def _agentic_clear_direct_receiver(self, receiver, manifest) -> None:
        manager = receiver.kv_mgr
        room = manifest.direct_room
        manager.request_status.pop(room, None)
        manager.failure_records.pop(room, None)
        manager.required_prefill_response_num_table.pop(room, None)
        manager.prefill_response_tracker.pop(room, None)
        transfer_statuses = getattr(manager, "transfer_statuses", None)
        if transfer_statuses is not None:
            transfer_statuses.pop(room, None)
        rooms = manager.addr_to_rooms_tracker.get(manifest.direct_bootstrap_addr)
        if rooms is not None:
            rooms.discard(room)
