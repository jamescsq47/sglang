"""AgenticOwnershipHooks: shared native Scheduler state; no independent state owner."""

import logging
from sglang.srt.disaggregation.agentic_scheduler_direct import AgenticEarlyDirectReceive
from sglang.srt.disaggregation.agentic_kv_lifecycle import AgenticRequestMetadata
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.disaggregation.agentic_kv_lifecycle import SnapshotState
from sglang.srt.environ import envs
from contextlib import nullcontext
import time

logger = logging.getLogger("sglang.srt.managers.scheduler")


class AgenticOwnershipHooks:
    def _agentic_snapshot_store(self):
        if not envs.SGLANG_AGENTIC_KV_LIFECYCLE.get():
            return None
        controller = getattr(self, "agentic_storage_controller", None)
        if controller is None:
            controller = getattr(self.tree_cache, "cache_controller", None)
        backend = getattr(controller, "storage_backend", None)
        factory = getattr(backend, "agentic_snapshot_store", None)
        if factory is None:
            return None
        return factory()

    def _agentic_service_p_workset_leases(self) -> None:
        """Service background restore intents at an allocator-safe boundary."""

        broker = getattr(self, "agentic_p_workset_broker", None)
        if broker is not None:
            reserve_tokens = 0
            chunked_req = getattr(self, "chunked_req", None)
            private_suffix = (
                None
                if chunked_req is None
                or not getattr(chunked_req, "_agentic_workset_backed", False)
                else getattr(chunked_req, "_agentic_workset_suffix_indices", None)
            )
            if (
                chunked_req is not None
                and (private_suffix is None or private_suffix.numel() == 0)
            ):
                # ``fill_ids`` is the prefix through the chunk that just ran;
                # ``origin_input_ids + output_ids`` is the complete logical
                # prompt.  Page-round the unprocessed suffix exactly as the
                # allocator will do on subsequent chunks.
                remaining = max(
                    0,
                    len(chunked_req.origin_input_ids)
                    + len(chunked_req.output_ids)
                    - len(chunked_req.fill_ids),
                )
                reserve_tokens = (
                    (remaining + self.page_size - 1) // self.page_size
                ) * self.page_size
            broker.service(
                self.token_to_kv_pool_allocator,
                reserve_tokens=reserve_tokens,
            )
            # This wrapper is used by BOTH ordinary and disaggregated Prefill
            # safe boundaries. PD never calls get_next_batch_to_run().
            manager = getattr(self, "agentic_host_staging_manager", None)
            if manager is not None and getattr(manager, "h2d_decoupled", False):
                event_progress = (
                    getattr(self, "tp_size", 1) == 1
                    and getattr(manager, "h2d_event_progress", False)
                )
                self._agentic_progress_granted_slow(owned_only=event_progress)
                if event_progress:
                    self._agentic_refill_host_events(reserve_tokens=reserve_tokens)

    def _agentic_abort_cleanup(self, req: Req) -> None:
        """Release a P load claim and its complete snapshot on cancellation."""

        for pin_attr in (
            "_agentic_direct_parent_pin_node",
            "_agentic_kv_host_pin_node",
        ):
            parent_pin = getattr(req, pin_attr, None)
            if parent_pin is not None:
                self.tree_cache.dec_lock_ref(parent_pin)
                delattr(req, pin_attr)

        direct_parent_tokens = getattr(req, "_agentic_direct_parent_token_count", 0)
        release_agentic_cache = getattr(
            self.tree_cache, "release_agentic_request_cache", None
        )
        if direct_parent_tokens and release_agentic_cache is not None:
            release_agentic_cache(req, committed_len=direct_parent_tokens)
            del req._agentic_direct_parent_token_count

        release_prefetch = getattr(self.tree_cache, "release_aborted_request", None)
        if release_prefetch is not None:
            release_prefetch(req.rid)

        direct_receiver = getattr(req, "_agentic_direct_receiver", None)
        if direct_receiver is not None:
            direct_manifest = req._agentic_direct_manifest
            direct_workset = getattr(req, "_agentic_direct_workset_lease", None)
            self.agentic_p_workset_broker.request_release(
                direct_manifest.snapshot_id,
                direct_workset,
                io_attempt=getattr(req, "_agentic_direct_io_attempt", None),
            )
            direct_store = req._agentic_kv_snapshot_store
            if direct_workset is not None and direct_workset.state == "release_pending":
                # CommonKVReceiver.abort() only changes local bookkeeping; it
                # does not cancel or fence a remote NIXL WRITE.  Preserve the
                # receiver and its destination pages until the transport
                # itself reports a terminal state.
                entry = AgenticEarlyDirectReceive(
                    request=direct_manifest.request,
                    manifest=direct_manifest,
                    claim_id=req._agentic_direct_claim_id,
                    receiver=direct_receiver,
                    device_indices=req._agentic_direct_indices,
                    started_at=getattr(
                        req, "_agentic_direct_started_at", time.monotonic()
                    ),
                    arrived_at=time.time(),
                    workset_lease=direct_workset,
                    io_attempt=getattr(req, "_agentic_direct_io_attempt", None),
                    abort_requested=True,
                    abort_release_claim=True,
                    abort_reason="request_aborted",
                )
                with getattr(self, "agentic_early_direct_poll_lock", nullcontext()):
                    self.agentic_early_direct_receives[direct_manifest.snapshot_id] = (
                        entry
                    )
            else:
                try:
                    direct_receiver.abort()
                except Exception:
                    logger.exception(
                        "Failed to abort inactive direct receiver for %s", req.rid
                    )
                self._agentic_clear_direct_receiver(direct_receiver, direct_manifest)
                current = direct_store.load(
                    direct_manifest.request, require_ready=False
                )
                if (
                    current is not None
                    and current.state in {
                        SnapshotState.DIRECT_LOADING,
                        SnapshotState.P_RECEIVED,
                    }
                ):
                    try:
                        if current.state is SnapshotState.P_RECEIVED:
                            direct_store.release_received_direct(
                                current, req._agentic_direct_claim_id
                            )
                        else:
                            direct_store.release_direct_claim(
                                current, req._agentic_direct_claim_id
                            )
                    except Exception:
                        logger.exception(
                            "Failed to release aborted direct claim for %s", req.rid
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

        workset_lease = getattr(req, "_agentic_p_workset_lease", None)
        if workset_lease is not None:
            workset_owner = getattr(req, "_agentic_p_workset_broker", None)
            release_abort = getattr(workset_owner, "release_abort", None)
            if release_abort is not None:
                # Multi-node V2 owns physical pages through its sole memory
                # authority; the legacy scheduler broker is intentionally
                # absent and must not be used for this lease.
                release_abort(req, reason="request_aborted")
            else:
                legacy_broker = workset_owner or self.agentic_p_workset_broker
                if legacy_broker is None:
                    raise RuntimeError("workset lease has no releasing authority")
                legacy_broker.release_handed(
                    workset_lease.snapshot_id, workset_lease, req=req
                )
        for name in (
            "_agentic_workset_backed",
            "_agentic_p_workset_lease",
            "_agentic_p_workset_broker",
            "_agentic_workset_suffix_indices",
        ):
            if hasattr(req, name):
                delattr(req, name)

        metadata = AgenticRequestMetadata.from_req(req)
        host_staging = getattr(self, "agentic_host_staging_manager", None)
        if (
            host_staging is not None
            and metadata is not None
            and metadata.parent is not None
        ):
            host_staging.abort_request(req.rid, metadata.parent)
            if (
                getattr(self, "tp_size", 1) > 1
                and metadata.parent.snapshot_id
                in getattr(self, "agentic_tp_host_active_requests", {})
            ):
                cancelled = getattr(self, "agentic_tp_host_cancelled_requests", None)
                if cancelled is None:
                    cancelled = self.agentic_tp_host_cancelled_requests = {}
                cancelled.setdefault(metadata.parent.snapshot_id, req.rid)

        snapshot_store = getattr(req, "_agentic_kv_snapshot_store", None)
        manifest = getattr(req, "_agentic_kv_manifest", None)
        if snapshot_store is None or manifest is None:
            return
        observed = snapshot_store.load(manifest.request, require_ready=False)
        if observed is None or observed.state in {
            SnapshotState.CONSUMED,
            SnapshotState.EVICTED,
            SnapshotState.FAILED,
        }:
            return
        result = None
        try:
            if observed.state in {SnapshotState.P_LOADING, SnapshotState.P_HOST}:
                result = snapshot_store.abandon_load(observed, req._agentic_kv_claim_id)
            elif observed.state is SnapshotState.P_GPU:
                result = snapshot_store.delete_snapshot(
                    observed, final_state=SnapshotState.CONSUMED
                )
            elif observed.state is SnapshotState.DELETE_PENDING:
                retry = getattr(self.tree_cache, "queue_agentic_delete_retry", None)
                if retry is not None:
                    retry(snapshot_store, observed.request)
                return
        except Exception:
            retry = getattr(self.tree_cache, "queue_agentic_delete_retry", None)
            if retry is not None:
                retry(snapshot_store, observed.request)
            logger.exception("Agentic abort cleanup failed for %s", req.rid)
            return
        if result is not None and not result.removed:
            retry = getattr(self.tree_cache, "queue_agentic_delete_retry", None)
            if retry is not None:
                retry(snapshot_store, observed.request)

    def _record_agentic_forward_read_fence(self, batch: ScheduleBatch) -> None:
        """Fence background KV readers at the actual Forward submit point.

        ``GenerationBatchResult.copy_done`` is intentionally recorded only
        after delayed sampling/copy-to-CPU.  In structured-tool workloads that
        event can therefore still be unrecorded when the previous batch is
        retired.  Agentic egress needs a different boundary: the Forward that
        last read the request pages has been submitted on the current stream.
        Recording a dedicated event here preserves overlap without letting a
        transport worker read or release pages behind an unrecorded event.
        """

        bridge = getattr(self, "agentic_p_memory_v2_bridge", None)
        if bridge is None:
            bridge = getattr(self, "agentic_d_memory_v2_bridge", None)
        if bridge is None:
            return
        decode_bridge = getattr(self, "agentic_d_memory_v2_bridge", None)
        if decode_bridge is not None:
            request_bridges = [
                getattr(req, "_agentic_d_memory_bridge", None)
                for req in batch.reqs
            ]
            owned = [value for value in request_bridges if value is not None]
            if owned and (
                len(owned) != len(request_bridges)
                or any(value is not decode_bridge for value in owned)
            ):
                raise RuntimeError(
                    "V2 Decode forward batch mixes memory authorities"
                )
            # process_batch_result receives a shallow copy whose Req objects
            # remain shared with newer overlap batches.  Capture ownership on
            # the batch itself before an earlier result can release a finished
            # Req and remove its bridge attributes.
            batch.agentic_d_memory_v2_owned = bool(owned)
        event = self.device_module.Event()
        event.record()
        bridge.record_forward_fence(batch.reqs, event)
