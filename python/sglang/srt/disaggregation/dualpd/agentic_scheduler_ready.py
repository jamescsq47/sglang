"""AgenticReadyMixin: shared native Scheduler state; no independent state owner."""

import logging
from sglang.srt.disaggregation.agentic_kv_lifecycle import AgenticRequestMetadata
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from typing import Optional
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.disaggregation.agentic_kv_lifecycle import SnapshotLifecycleError
from sglang.srt.disaggregation.agentic_kv_lifecycle import SnapshotNotReadyError
from sglang.srt.disaggregation.agentic_kv_lifecycle import SnapshotState
from sglang.srt.environ import envs
import json
from contextlib import nullcontext
import os
from sglang.srt.disaggregation.agentic_kv_lifecycle import page_namespace
import time

logger = logging.getLogger("sglang.srt.managers.scheduler")


class AgenticReadyMixin:
    def _agentic_refill_host_events(self, *, reserve_tokens: int = 0) -> None:
        """One bounded FIFO refill after completion handoff frees capacity.

        Events are hints, never ownership/fence authority. Use the original
        admission routine for fresh work (no source-class priority), then
        service its intents and start selected loads at this SAME allocator
        boundary. Retain the ordinary level-trigger path for missed/remote
        events, failed claims and absent HTTP waiters. No wait for DMA here.
        """
        manager = getattr(self, "agentic_host_staging_manager", None)
        if (
            getattr(self, "tp_size", 1) != 1
            or manager is None
            or not getattr(manager, "h2d_decoupled", False)
            or not getattr(manager, "h2d_event_progress", False)
        ):
            return
        events = manager.drain_scheduler_events(max_events=64)
        if not events or not self.agentic_kv_waiting_queue:
            return
        started = time.monotonic()
        # The initial completion sweep may just have released the resident
        # credit. Previously no fresh waiter could use it until another loop.
        self._drain_agentic_kv_waiting_queue(continue_tick=True)
        self.agentic_p_workset_broker.service(
            self.token_to_kv_pool_allocator, reserve_tokens=reserve_tokens
        )
        self._agentic_progress_granted_slow(owned_only=True)
        if not getattr(self, "_agentic_host_event_refill_logged", False):
            self._agentic_host_event_refill_logged = True
            logger.info(
                "AgenticKV host_event_refill_enabled events=%d tp_size=1",
                len(events),
            )
        # DEBUG avoids per-pass production logging on the Forward path.
        logger.debug(
            "AgenticKV host_event_refill events=%d elapsed_ms=%.3f waiting=%d",
            len(events), (time.monotonic() - started) * 1000,
            len(self.agentic_kv_waiting_queue),
        )

    def _agentic_should_defer(
        self, req: Req, started_at: float, *, allow_start_io: bool = True
    ) -> bool:
        """Claim a committed parent snapshot, or keep the request metadata-only."""

        if getattr(req, "_agentic_direct_receiver", None) is not None:
            req._agentic_kv_queue_class = "fast"
            return self._agentic_poll_direct_load(req)
        if getattr(req, "_agentic_kv_gate_complete", False):
            return False
        metadata = AgenticRequestMetadata.from_req(req)
        if metadata is None or metadata.parent is None:
            req._agentic_kv_gate_complete = True
            return False
        if (
            getattr(self, "tp_size", 1) > 1
            and metadata.parent.snapshot_id
            in getattr(self, "agentic_tp_host_cancelled_requests", {})
        ):
            # A new HTTP attempt must not start I/O under the old attempt's
            # cancellation command. All-rank CLEAR removes this tombstone.
            return True
        # A terminal application ACK is authoritative even when D deliberately
        # skipped snapshot publication.  A later repair/retry request must
        # recompute immediately instead of waiting the generic snapshot-ready
        # timeout for data that can never appear.
        marker_store = getattr(self, "agentic_early_claim_store", None)
        read_final = getattr(marker_store, "read_final", None)
        if read_final is not None:
            final_marker = read_final(
                metadata.parent,
                not_before=0.0,
                max_age_seconds=max(
                    600.0,
                    envs.SGLANG_AGENTIC_KV_READY_TIMEOUT.get() + 5.0,
                ),
            )
            if final_marker is not None:
                req._agentic_kv_gate_complete = True
                req._agentic_kv_fallback = "application_final"
                logger.info(
                    "AgenticKV parent_terminal_recompute parent=%s req=%s",
                    metadata.parent.snapshot_id,
                    req.rid,
                )
                return False
        # The router marker may already have caused P to receive this
        # generation while the full request was still being tokenized.  Bind
        # that allocator-owned KV before consulting Host/Mooncake state (the
        # Direct manifest is intentionally CONSUMED as soon as D may release
        # its source pages).
        early_direct = self._agentic_bind_early_direct_receive(req, metadata.parent)
        if early_direct is not None:
            return early_direct
        host_staging = getattr(self, "agentic_host_staging_manager", None)
        if host_staging is not None:
            if getattr(self, "tp_size", 1) > 1:
                host_action = getattr(self, "_agentic_tp_host_actions", {}).get(
                    metadata.parent.snapshot_id
                )
                allow_prepare_io = bool(
                    allow_start_io
                    and host_action is not None
                    and host_action in {"prepare", "start", "bind", "commit"}
                )
                allow_start_io = bool(
                    allow_start_io
                    and host_action is not None
                    and host_action in {"start", "bind", "commit"}
                )
                allow_bind_io = bool(
                    host_action is not None and host_action in {"bind", "commit"}
                )
            else:
                allow_prepare_io = allow_start_io
                allow_bind_io = True
            host_gate = host_staging.gate_request(
                req,
                metadata.parent,
                allow_prepare=allow_prepare_io,
                allow_start=allow_start_io,
                allow_bind=allow_bind_io,
            )
            if host_gate is not None:
                req._agentic_kv_queue_class = "slow"
                if host_gate is False and getattr(self, "tp_size", 1) > 1:
                    self.agentic_tp_host_local_admitted.add(metadata.parent.snapshot_id)
                # A shared-Host entry is an authoritative recoverable parent.
                # Scheduler delay or temporary capacity pressure is not an
                # eviction policy and therefore must never turn it into a
                # complete recompute.  READY_TIMEOUT is diagnostics only.
                timeout = max(0.0, envs.SGLANG_AGENTIC_KV_READY_TIMEOUT.get())
                if (
                    host_gate
                    and not self._agentic_io_active(req)
                    and not host_staging.snapshot_ready(metadata.parent)
                    and time.monotonic() - started_at >= timeout
                    and not getattr(req, "_agentic_host_wait_warned", False)
                ):
                    req._agentic_host_wait_warned = True
                    logger.warning(
                        "AgenticKV shared_host_wait snapshot=%s req=%s "
                        "waited_s=%.1f action=retain_parent",
                        metadata.parent.snapshot_id,
                        req.rid,
                        timeout,
                    )
                return host_gate
        snapshot_store = self._agentic_snapshot_store()
        if snapshot_store is None:
            logger.warning(
                "Agentic KV metadata found for %s, but P has no Mooncake snapshot store; "
                "falling back to recompute",
                req.rid,
            )
            req._agentic_kv_gate_complete = True
            req._agentic_kv_fallback = "no_snapshot_store"
            return False

        manifest = snapshot_store.load(metadata.parent, require_ready=False)
        if (
            manifest is not None
            and manifest.state is SnapshotState.DIRECT_READY
            and not getattr(req, "_agentic_direct_disabled", False)
        ):
            if envs.SGLANG_AGENTIC_KV_FORCE_SLOW_PATH.get():
                # D owns the parent until its independent worker commits the
                # complete snapshot to Shared Host.  Do not claim Direct in
                # this ablation; the next scheduler pass observes HOST_READY.
                req._agentic_kv_queue_class = "slow"
                return True
            req._agentic_kv_queue_class = "fast"
            if getattr(self, "tp_size", 1) > 1:
                # TP Direct admission is exclusively driven by rank 0's
                # inotify FIFO and native broadcast command.  Never let an
                # individual rank enter the legacy request-owned receiver.
                return True
            # When the router marker exists, the scheduler-independent
            # receiver is authoritative.  Do not let the legacy Req-owned
            # path bypass its Direct I/O cap merely because tokenization was
            # unusually fast; wait for the early receiver to claim/bind it.
            marker_store = getattr(self, "agentic_early_claim_store", None)
            if marker_store is not None:
                marker = marker_store.read_arrival(
                    metadata.parent,
                    not_before=manifest.created_at,
                    max_age_seconds=max(
                        5.0,
                        envs.SGLANG_AGENTIC_KV_FAST_TOOL_THRESHOLD.get()
                        + envs.SGLANG_AGENTIC_KV_DIRECT_HANDSHAKE_TIMEOUT.get()
                        + 1.0,
                    ),
                )
                if marker is not None:
                    if allow_start_io:
                        early_direct = self._agentic_bind_early_direct_receive(
                            req, metadata.parent
                        )
                        if early_direct is not None:
                            return early_direct
                    return True
            if not allow_start_io:
                return True
            started = self._agentic_start_direct_load(req, snapshot_store, manifest)
            if started:
                return True
            req._agentic_kv_gate_complete = True
            return False
        stale_seconds = max(0.0, envs.SGLANG_AGENTIC_KV_STALE_SECONDS.get())
        recoverable_states = {
            SnapshotState.OFFLOADING,
            SnapshotState.P_LOADING,
            SnapshotState.P_HOST,
            SnapshotState.P_GPU,
            SnapshotState.DELETE_PENDING,
        }
        if (
            stale_seconds > 0
            and manifest is not None
            and manifest.state in recoverable_states
            and time.time() - manifest.updated_at >= stale_seconds
        ):
            try:
                result = snapshot_store.recover_stale(manifest)
                if not result.removed:
                    retry = getattr(self.tree_cache, "queue_agentic_delete_retry", None)
                    if retry is not None:
                        retry(snapshot_store, manifest.request)
            except Exception:
                logger.exception(
                    "Failed to recover stale agentic snapshot %s",
                    manifest.snapshot_id,
                )
            req._agentic_kv_gate_complete = True
            req._agentic_kv_fallback = f"stale:{manifest.state.value}"
            return False
        if manifest is not None and manifest.state is SnapshotState.MOONCAKE_READY:
            req._agentic_kv_queue_class = "slow"
            if not allow_start_io:
                return True
            claim_id = f"p:{req.rid}"
            claimed = snapshot_store.claim_for_load(metadata.parent, claim_id)
            req._agentic_kv_snapshot_store = snapshot_store
            req._agentic_kv_manifest = claimed
            req._agentic_kv_claim_id = claim_id
            req._agentic_kv_storage_namespace = page_namespace(metadata.parent)
            req._agentic_kv_gate_complete = True
            logger.info(
                "AgenticKV mooncake_load_claim snapshot=%s tokens=%d bytes=%d req=%s",
                claimed.snapshot_id,
                claimed.token_count,
                claimed.byte_size,
                req.rid,
            )
            return False

        timeout = max(0.0, envs.SGLANG_AGENTIC_KV_READY_TIMEOUT.get())
        if manifest is not None and manifest.state in {
            SnapshotState.DIRECT_LOADING,
            SnapshotState.P_RECEIVED,
            SnapshotState.P_LOADING,
        }:
            # These are producer/receiver progress states, not evidence that
            # the parent KV is unavailable.  Under c640 the Direct manager can
            # remain in one of them for several seconds; admitting the child
            # here silently turns transport congestion into a full recompute.
            if (
                time.monotonic() - started_at >= timeout
                and not getattr(req, "_agentic_parent_wait_warned", False)
            ):
                req._agentic_parent_wait_warned = True
                logger.warning(
                    "AgenticKV parent_wait snapshot=%s req=%s state=%s "
                    "waited_s=%.1f action=retain_parent",
                    metadata.parent.snapshot_id,
                    req.rid,
                    manifest.state.value,
                    timeout,
                )
            return True

        if manifest is not None and manifest.state in {
            SnapshotState.P_HOST,
            SnapshotState.P_GPU,
            SnapshotState.TO_DECODE,
            SnapshotState.CONSUMED,
            SnapshotState.DELETE_PENDING,
            SnapshotState.EVICTED,
            SnapshotState.FINAL,
            SnapshotState.FAILED,
        }:
            req._agentic_kv_gate_complete = True
            req._agentic_kv_fallback = manifest.state.value
            logger.info(
                "AgenticKV parent_snapshot_terminal_fallback snapshot=%s "
                "state=%s req=%s",
                metadata.parent.snapshot_id,
                manifest.state.value,
                req.rid,
            )
            return False

        # A missing manifest is normal while D changes ownership from Direct
        # to Shared Host.  That transition can exceed the old 2s+8s shortcut
        # when several D workers publish concurrently.  The child must wait
        # for the request-level ready timeout instead of racing the producer
        # and recomputing an otherwise recoverable parent snapshot.
        if (
            time.monotonic() - started_at >= timeout
            and not getattr(req, "_agentic_parent_missing_warned", False)
        ):
            state = "missing" if manifest is None else manifest.state.value
            req._agentic_parent_missing_warned = True
            logger.warning(
                "AgenticKV parent_missing_wait snapshot=%s req=%s state=%s "
                "waited_s=%.1f action=retain_parent",
                metadata.parent.snapshot_id,
                req.rid,
                state,
                timeout,
            )
        return True

    @staticmethod
    def _agentic_queue_class(req: Req) -> str:
        value = getattr(req, "_agentic_kv_queue_class", "fast")
        return value if value in {"fast", "slow", "new"} else "fast"

    def _agentic_io_active(self, req: Req) -> bool:
        if getattr(req, "_agentic_direct_receiver", None) is not None:
            return True
        host_staging = getattr(self, "agentic_host_staging_manager", None)
        return host_staging is not None and req.rid in host_staging.loads

    def _agentic_io_kind(self, req: Req) -> Optional[str]:
        """Return the active receive class so Direct has independent credits."""

        if getattr(req, "_agentic_direct_receiver", None) is not None:
            return "direct"
        host_staging = getattr(self, "agentic_host_staging_manager", None)
        if host_staging is not None and req.rid in host_staging.loads:
            return "slow"
        if host_staging is not None and getattr(host_staging, "h2d_decoupled", False):
            metadata = AgenticRequestMetadata.from_req(req)
            parent = None if metadata is None else metadata.parent
            if parent is not None and parent.snapshot_id in host_staging.h2d_selected_snapshots():
                # A prestart workset intent already owns its physical lane.
                # It must progress even when NEW-I/O admission has no slots.
                return "slow"
        return None

    def _agentic_progress_granted_slow(self, *, owned_only: bool = False) -> None:
        """Consume bounded existing Slow grants at the allocator-safe boundary.

        No new request selection, class priority, allocator thread or TP phase:
        just start already-selected grants and bind completed worksets without
        forcing an unrelated Forward between grant and H2D start.
        """
        manager = getattr(self, "agentic_host_staging_manager", None)
        if manager is None or not getattr(manager, "h2d_decoupled", False):
            return
        selected = manager.h2d_selected_snapshots()
        if not selected:
            return
        remaining = []
        for req, arrived_at in self.agentic_kv_waiting_queue:
            metadata = AgenticRequestMetadata.from_req(req)
            parent = None if metadata is None else metadata.parent
            if parent is None or parent.snapshot_id not in selected:
                remaining.append((req, arrived_at))
                continue
            try:
                deferred = manager.gate_request(req, parent)
            except (SnapshotNotReadyError, SnapshotLifecycleError):
                # Same recoverable lifecycle races as the ordinary admission
                # loop: keep this exact waiter/ownership and progress others.
                remaining.append((req, arrived_at))
                continue
            if deferred is not False:
                remaining.append((req, arrived_at))
                continue
            if owned_only:
                lease = getattr(req, "_agentic_p_workset_lease", None)
                if not (
                    getattr(req, "_agentic_workset_backed", False)
                    and getattr(req, "_agentic_mamba_runtime_reserved", False)
                    and getattr(lease, "state", None) == "handed"
                ):
                    # False also covers explicit eviction/fallback. That
                    # metadata-only request must use ordinary FIFO admission,
                    # not the completion shortcut for an owned workset.
                    remaining.append((req, arrived_at))
                    continue
            # This workset is fully bound/handed, not a fresh compute admission
            # that can remain behind unallocated waiters and fill Mamba slots.
            req._agentic_kv_wait_enqueued = False
            self._agentic_publish_p_scheduled(req)
            self._add_request_to_queue(req)
        self.agentic_kv_waiting_queue = remaining

    def _agentic_bind_completed_waiters(self) -> None:
        """Bind completed Direct ingress independently of Prefill admission.

        The request may remain in the priority queue until a later admission
        batch. Binding now converts the full physical workset lease into a
        protected parent plus immediately usable Prefill suffix capacity.
        """

        receives = getattr(self, "agentic_early_direct_receives", None)
        completions = getattr(self, "agentic_early_direct_completion_queue", None)
        if not receives:
            return
        with getattr(self, "agentic_early_direct_poll_lock", nullcontext()):
            if completions is None:
                # Compatibility for embedders/tests constructing Scheduler
                # without init_running_status(). Production always uses the
                # completion queue and never scans all receivers here.
                completed = {
                    snapshot_id
                    for snapshot_id, entry in receives.items()
                    if entry.completed_at is not None
                }
            else:
                completed = set(completions)
                completions.clear()
        if not completed:
            return
        waiting_by_parent = {}
        for req, _ in self.agentic_kv_waiting_queue:
            if getattr(req, "_agentic_kv_gate_complete", False):
                continue
            metadata = AgenticRequestMetadata.from_req(req)
            parent = metadata.parent if metadata is not None else None
            if parent is not None:
                waiting_by_parent[parent.snapshot_id] = (req, parent)
        for snapshot_id in completed:
            waiter = waiting_by_parent.get(snapshot_id)
            if waiter is not None:
                self._agentic_bind_early_direct_receive(*waiter, allow_tp_commit=False)
            else:
                # Transport progress is independent of tokenizer/scheduler
                # progress.  In particular, a non-primary TP rank can finish
                # its shard before the broadcast request has entered that
                # rank's metadata queue.  Keep the edge-triggered completion
                # pending until the matching request is actually bindable;
                # dropping it here permanently strands that rank's Direct
                # reserve allocation.
                with getattr(self, "agentic_early_direct_poll_lock", nullcontext()):
                    entry = self.agentic_early_direct_receives.get(snapshot_id)
                    if entry is not None and entry.completed_at is not None:
                        self.agentic_early_direct_completion_queue.append(snapshot_id)

    def _drain_agentic_kv_waiting_queue_tp1(self) -> None:
        """Compatibility entry point using unified arrival/native admission."""

        self._drain_agentic_kv_waiting_queue()

    def _drain_agentic_kv_waiting_queue(self, *, continue_tick: bool = False) -> None:
        """Progress active KV I/O and admit pending work in arrival order.

        Unadmitted requests own metadata only, but a completed receive may
        already have handed its workset to a waiter. Each scheduler iteration
        first polls already-started transfers. New Direct, Slow, and initial
        requests then share one arrival-ordered compute-admission queue; their
        I/O engines and credits remain independent. A small admission batch
        amortizes scheduler ticks that contain long Prefill kernels.
        """
        if not continue_tick:
            self._agentic_admission_used_this_tick = 0
        # This sweep is intentionally outside admission_batch.  Admission
        # limits Prefill compute; it must not retain a completed workset lease.
        if getattr(self, "tp_size", 1) == 1:
            self._agentic_bind_completed_waiters()

        tp_bind_snapshots = (
            [
                snapshot_id
                for snapshot_id, action in getattr(
                    self, "_agentic_tp_direct_actions", {}
                ).items()
                if action in {"prepare_bind", "commit_bind"}
            ]
            if getattr(self, "tp_size", 1) > 1
            else []
        )
        if (
            getattr(self, "tp_size", 1) > 1
            and not tp_bind_snapshots
            and getattr(self, "_agentic_tp_selected_snapshot", None) is not None
            and getattr(self, "_agentic_tp_direct_action", "prepare_bind")
            in {"prepare_bind", "commit_bind"}
        ):
            tp_bind_snapshots = [self._agentic_tp_selected_snapshot]
        tp_host_timeout_snapshot = (
            getattr(self, "_agentic_tp_host_timeout_snapshot", None)
            if getattr(self, "tp_size", 1) > 1
            else None
        )
        tp_host_commit_snapshots = (
            list(getattr(self, "_agentic_tp_host_commit_snapshots", ()))
            if getattr(self, "tp_size", 1) > 1
            else []
        )
        if (
            getattr(self, "tp_size", 1) > 1
            and not tp_host_commit_snapshots
            and getattr(self, "_agentic_tp_host_commit_snapshot", None) is not None
        ):
            tp_host_commit_snapshots = [self._agentic_tp_host_commit_snapshot]
        if not self.agentic_kv_waiting_queue:
            return

        tp_host_snapshots = (
            list(getattr(self, "_agentic_tp_host_actions", ()))
            if getattr(self, "tp_size", 1) > 1
            else []
        )
        if (
            getattr(self, "tp_size", 1) > 1
            and not tp_host_snapshots
            and getattr(self, "_agentic_tp_host_selected_snapshot", None) is not None
        ):
            tp_host_snapshots = [self._agentic_tp_host_selected_snapshot]

        try:
            scan_limit = max(
                1, int(os.environ.get("SGLANG_AGENTIC_KV_ADMISSION_SCAN_LIMIT", "16"))
            )
            admission_batch = max(
                1, int(os.environ.get("SGLANG_AGENTIC_KV_ADMISSION_BATCH", "8"))
            )
            if continue_tick:
                admission_batch = max(
                    0, admission_batch - getattr(self, "_agentic_admission_used_this_tick", 0)
                )
            host_staging = getattr(self, "agentic_host_staging_manager", None)
            default_slow_io_cap = max(
                1, int(getattr(host_staging, "max_h2d_inflight", 1))
            )
            slow_io_cap = max(
                1,
                int(
                    os.environ.get(
                        "SGLANG_AGENTIC_KV_SELECTED_IO_CAP",
                        str(default_slow_io_cap),
                    )
                ),
            )
            direct_io_cap = max(
                1, int(os.environ.get("SGLANG_AGENTIC_KV_DIRECT_IO_CAP", "4"))
            )
        except ValueError:
            logger.exception("Invalid agentic KV admission setting")
            raise

        def owns_resident_mamba_workset(req):
            # TP1's completion sweep can commit ownership before this bounded
            # metadata scan reaches the waiter. Such a request already holds
            # parent + suffix KV and all runtime Mamba slots; it needs no new
            # I/O admission. Leaving it behind older unallocated waiters can
            # deadlock the state pool. Do not apply rank-local readiness to TP
            # (which uses the group commands below), or change Qwen3 policy.
            lease = getattr(req, "_agentic_p_workset_lease", None)
            return bool(
                getattr(self, "tp_size", 1) == 1
                and getattr(req, "_agentic_kv_gate_complete", False)
                and getattr(req, "_agentic_workset_backed", False)
                and getattr(req, "_agentic_mamba_runtime_reserved", False)
                and getattr(lease, "state", None) == "handed"
            )

        active = []
        resident = []
        inactive = []
        for entry in self.agentic_kv_waiting_queue:
            req = entry[0]
            if owns_resident_mamba_workset(req):
                resident.append(entry)
            elif self._agentic_io_active(req):
                active.append(entry)
            else:
                inactive.append(entry)

        # I/O ownership is independent, but Prefill admission is ordinary FIFO
        # across request classes. Already-active I/O remains first so a ready
        # Slow load can bind and immediately enter incremental Prefill.
        # Direct and Slow have independent I/O engines and neither class may
        # starve the other. Carry every exact TP group command visible on this
        # scheduler boundary; ordinary FIFO admission resumes after these
        # ownership transitions. The exact-snapshot filter still guarantees
        # that both ranks mutate the same request generations.
        forced_tp_snapshots = list(tp_bind_snapshots)
        for snapshot_id in (
            tp_host_commit_snapshots + tp_host_snapshots + [tp_host_timeout_snapshot]
        ):
            if snapshot_id is not None and snapshot_id not in forced_tp_snapshots:
                forced_tp_snapshots.append(snapshot_id)
        if forced_tp_snapshots:
            # The two TP ranks can receive tokenized HTTP requests in a
            # different order.  While one group bind is active, both queues
            # therefore advance only that exact parent generation.  No thread
            # blocks; a rank that has not received it yet simply retries on
            # the next scheduler tick.
            selected_by_snapshot = {}
            untouched = []
            for entry in active + inactive:
                req = entry[0]
                metadata = AgenticRequestMetadata.from_req(req)
                parent = metadata.parent if metadata is not None else None
                if (
                    parent is not None
                    and parent.snapshot_id in forced_tp_snapshots
                    and parent.snapshot_id not in selected_by_snapshot
                ):
                    selected_by_snapshot[parent.snapshot_id] = entry
                else:
                    untouched.append(entry)
            selected = [
                selected_by_snapshot[snapshot_id]
                for snapshot_id in forced_tp_snapshots
                if snapshot_id in selected_by_snapshot
            ]
            # HTTP arrival at TP ranks can be skewed.  Preparing local I/O for
            # one available snapshot is safe: the existing group-status
            # barrier still prevents model admission until every rank has
            # restored that same snapshot.  A missing command therefore must
            # not block unrelated restores that are already visible.
            if not selected:
                return
        else:
            inactive_by_arrival = sorted(inactive, key=lambda entry: entry[1])
            selected = (
                sorted(resident, key=lambda entry: entry[1])
                + active
                + inactive_by_arrival[:scan_limit]
            )
            untouched = inactive_by_arrival[scan_limit:]
        still_waiting = []
        new_io_started = 0
        newly_admitted = 0
        # A selected request may overlap its receive/load with the current
        # Prefill batch, but requests that have not been selected remain
        # metadata-only.  This is a global cap, not a per-tick cap: an active
        # receive from an earlier scheduler iteration consumes the slot.
        active_direct = sum(self._agentic_io_kind(req) == "direct" for req, _ in active)
        active_slow = sum(self._agentic_io_kind(req) == "slow" for req, _ in active)
        if host_staging is not None and getattr(host_staging, "h2d_decoupled", False):
            # A completed load still owns a workset, but no longer owns a DMA
            # lane. Counting it as in-flight would undo transport decoupling.
            active_slow = host_staging.h2d_physical_occupancy()
        direct_starts_left = max(0, direct_io_cap - active_direct)
        slow_starts_left = max(0, slow_io_cap - active_slow)
        for req, started_at in selected:
            previous_kind = self._agentic_io_kind(req)
            was_active = previous_kind is not None
            resident_workset = owns_resident_mamba_workset(req)
            if (
                not was_active
                and not resident_workset
                and newly_admitted >= admission_batch
            ):
                still_waiting.append((req, started_at))
                continue
            queue_class = self._agentic_queue_class(req)
            if was_active:
                allow_start_io = True
            elif queue_class == "slow":
                allow_start_io = slow_starts_left > 0
            elif queue_class == "fast":
                # A fresh parent request can discover either a DIRECT_READY
                # marker or a previously-fallen-back shared-Host record.
                # Probe without starting when Direct I/O slots are exhausted;
                # gate_request() will still reclassify an owned Host record as
                # slow, allowing it to use the slow budget on the next pass.
                allow_start_io = direct_starts_left > 0
            else:
                allow_start_io = True
            try:
                deferred = self._agentic_should_defer(
                    req,
                    started_at,
                    allow_start_io=allow_start_io,
                )
                if deferred:
                    still_waiting.append((req, started_at))
                else:
                    if not resident_workset:
                        newly_admitted += 1
                    req._agentic_kv_wait_enqueued = False
                    self._agentic_publish_p_scheduled(req)
                    direct_tokens = getattr(req, "_agentic_kv_direct_hit_tokens", 0)
                    if direct_tokens:
                        drain_match = self.tree_cache.match_prefix(
                            MatchPrefixParams(
                                key=RadixKey(
                                    req.origin_input_ids[:direct_tokens], req.extra_key
                                ),
                                req=req,
                            )
                        )
                        logger.info(
                            "AgenticKV direct_before_enqueue req=%s device_tokens=%d "
                            "host_tokens=%d",
                            req.rid,
                            len(drain_match.device_indices),
                            drain_match.host_hit_length,
                        )
                    self._add_request_to_queue(req)
                    if direct_tokens:
                        logger.info(
                            "AgenticKV direct_after_enqueue req=%s waiting=%d "
                            "kv_waiting=%d",
                            req.rid,
                            len(self.waiting_queue),
                            len(self.agentic_kv_waiting_queue),
                        )
                # A Mooncake-ready request is claimed here and its actual L3
                # prefetch is launched by _add_request_to_queue().  That
                # prefetch is tracked by the radix cache, not by
                # _agentic_io_active(), so looking only at direct/shared-host
                # receivers lets several Mooncake loads escape a cap=1 tick.
                # Treat either an active receiver or a claimed manifest as the
                # one selected I/O start for this scheduler pass.
                current_kind = self._agentic_io_kind(req)
                claimed_mooncake = (
                    getattr(req, "_agentic_kv_manifest", None) is not None
                )
                if not was_active and (current_kind is not None or claimed_mooncake):
                    new_io_started += 1
                    if current_kind == "direct":
                        direct_starts_left = max(0, direct_starts_left - 1)
                    else:
                        # Host and Mooncake recovery share the bounded slow
                        # ingress budget; they must never consume Direct's four
                        # page credits.
                        slow_starts_left = max(0, slow_starts_left - 1)
                    if deferred:
                        newly_admitted += 1
            except (SnapshotNotReadyError, SnapshotLifecycleError):
                # Another loop may be finishing the manifest transition.  Keep
                # this request metadata-only and retry; the timeout is the
                # explicit recompute fallback boundary.
                still_waiting.append((req, started_at))
        # Keep unscanned metadata waiters; the next pass applies native arrival
        # order again. Resident Mamba worksets must not depend on this rotation.
        self.agentic_kv_waiting_queue = untouched + still_waiting
        self._agentic_admission_used_this_tick = (
            getattr(self, "_agentic_admission_used_this_tick", 0) + newly_admitted
        )

    def _prioritize_agentic_prefill_ready(self) -> None:
        """Consume already-owned worksets without KV-source class priority.

        ``fast`` describes where a parent snapshot came from, not whether the
        request is currently runnable.  A failed Direct parent may retain the
        fast label while requiring ordinary KV for full recomputation.  It
        must never head-of-line block a handed Direct/Slow workset whose
        suffix pages are already owned by that exact request.
        """

        def owns_complete_workset(req):
            suffix = getattr(req, "_agentic_workset_suffix_indices", None)
            return bool(
                getattr(req, "_agentic_workset_backed", False)
                and suffix is not None
                and suffix.numel() > 0
            )

        # Physical ownership is the only leading key. TP>1 then uses the
        # broadcast ingress sequence so rank-local completion skew cannot
        # diverge collectives; TP=1 retains the native stable order. Neither
        # path includes Direct/Slow/New in its key.
        if getattr(self, "tp_size", 1) > 1:
            self.waiting_queue = sorted(
                self.waiting_queue,
                key=lambda req: (
                    0 if owns_complete_workset(req) else 1,
                    int(getattr(req, "_agentic_tp_prefill_sequence", 1 << 62)),
                    str(req.rid),
                ),
            )
        else:
            owned = [
                req for req in self.waiting_queue if owns_complete_workset(req)
            ]
            ordinary = [
                req for req in self.waiting_queue if not owns_complete_workset(req)
            ]
            self.waiting_queue = owned + ordinary

    def _agentic_mark_p_host(self, req: Req) -> None:
        manifest = getattr(req, "_agentic_kv_manifest", None)
        if manifest is None or manifest.state is not SnapshotState.P_LOADING:
            return
        # init_next_round_input has just rematched both GPU and Host HiCache.
        # Do not ACK a partial GET: the request-level contract is all-or-nothing.
        available_prefix = len(req.prefix_indices) + req.host_hit_length
        if available_prefix < manifest.token_count:
            return
        snapshot_store = req._agentic_kv_snapshot_store
        claim_id = req._agentic_kv_claim_id
        req._agentic_kv_manifest = snapshot_store.mark_p_host(manifest, claim_id)

    def _agentic_abandon_load(self, req: Req) -> None:
        manifest = getattr(req, "_agentic_kv_manifest", None)
        if manifest is None or manifest.state is not SnapshotState.P_LOADING:
            return
        snapshot_store = req._agentic_kv_snapshot_store
        result = snapshot_store.abandon_load(manifest, req._agentic_kv_claim_id)
        if not result.removed:
            retry = getattr(self.tree_cache, "queue_agentic_delete_retry", None)
            if retry is not None:
                retry(snapshot_store, manifest.request)
        req._agentic_kv_fallback = "partial_prefetch"
        req._agentic_kv_storage_namespace = None

    def _agentic_consume_if_already_on_gpu(self, req: Req) -> None:
        manifest = getattr(req, "_agentic_kv_manifest", None)
        if manifest is None or manifest.state is not SnapshotState.P_HOST:
            return
        if len(req.prefix_indices) < manifest.token_count:
            return
        snapshot_store = req._agentic_kv_snapshot_store
        claim_id = req._agentic_kv_claim_id
        p_gpu = snapshot_store.mark_p_gpu(manifest, claim_id)
        req._agentic_kv_manifest = p_gpu
        try:
            result = snapshot_store.delete_snapshot(
                p_gpu, final_state=SnapshotState.CONSUMED
            )
        except Exception:
            retry = getattr(self.tree_cache, "queue_agentic_delete_retry", None)
            if retry is not None:
                retry(snapshot_store, p_gpu.request)
            logger.exception(
                "Failed to delete already-resident agentic snapshot %s; queued retry",
                p_gpu.snapshot_id,
            )
            return
        if not result.removed:
            retry = getattr(self.tree_cache, "queue_agentic_delete_retry", None)
            if retry is not None:
                retry(snapshot_store, p_gpu.request)
            logger.info(
                "Agentic snapshot %s deletion is pending on %d leased pages",
                p_gpu.snapshot_id,
                len(result.remaining_keys),
            )

    @staticmethod
    def _agentic_publish_p_accepted(req: Req) -> None:
        """ACK that a request has entered P's scheduler-owned pipeline."""

        if getattr(req, "_agentic_p_accepted_notified", False):
            return
        ready_dir = os.environ.get("SGLANG_PD_P_READY_DIR", "")
        room = getattr(req, "bootstrap_room", None)
        if not ready_dir or room is None:
            return
        accepted_path = os.path.join(ready_dir, f"{room}.accepted")
        tmp_path = f"{accepted_path}.{os.getpid()}.tmp"
        os.makedirs(ready_dir, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump({"rid": req.rid}, handle, separators=(",", ":"))
        os.replace(tmp_path, accepted_path)
        req._agentic_p_accepted_notified = True

    @staticmethod
    def _agentic_publish_p_scheduled(req: Req) -> None:
        """Mark the boundary where queue wait ends and P processing begins."""

        if getattr(req, "_agentic_p_scheduled_notified", False):
            return
        ready_dir = os.environ.get("SGLANG_PD_P_READY_DIR", "")
        room = getattr(req, "bootstrap_room", None)
        if not ready_dir or room is None:
            return
        scheduled_path = os.path.join(ready_dir, f"{room}.scheduled")
        tmp_path = f"{scheduled_path}.{os.getpid()}.tmp"
        os.makedirs(ready_dir, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump({"rid": req.rid}, handle, separators=(",", ":"))
        os.replace(tmp_path, scheduled_path)
        req._agentic_p_scheduled_notified = True
