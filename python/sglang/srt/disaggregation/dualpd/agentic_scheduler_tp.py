"""AgenticTPMixin: shared native Scheduler state; no independent state owner."""

import logging
from sglang.srt.disaggregation.agentic_scheduler_direct import AgenticEarlyDirectReceive
from sglang.srt.disaggregation.agentic_workset import AgenticPWorksetLeaseBroker
from sglang.srt.disaggregation.agentic_kv_lifecycle import AgenticRequestMetadata
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration
from sglang.srt.disaggregation.agentic_kv_lifecycle import SnapshotState
from torch.distributed import barrier
from collections import deque
from sglang.srt.environ import envs
from contextlib import nullcontext
import os
import queue
from sglang.srt.disaggregation.agentic_tp import request_generation_key
import time
from sglang.srt.disaggregation.agentic_scheduler_compat import _native_scheduler_class

logger = logging.getLogger("sglang.srt.managers.scheduler")


class AgenticTPMixin:
    def _agentic_tp_abort_mailbox_write(
        self,
        entry: AgenticEarlyDirectReceive,
        operation,
    ) -> bool:
        """Run a TP abort write with per-request bounded retry/logging."""

        now = time.monotonic()
        if now < float(getattr(entry, "tp_abort_mailbox_retry_at", 0.0)):
            return False
        try:
            operation()
        except OSError:
            delay = max(
                0.05,
                float(getattr(entry, "tp_abort_mailbox_retry_delay", 0.05)),
            )
            entry.tp_abort_mailbox_retry_at = now + delay
            entry.tp_abort_mailbox_retry_delay = min(1.0, delay * 2.0)
            if not getattr(entry, "tp_abort_mailbox_error_logged", False):
                entry.tp_abort_mailbox_error_logged = True
                logger.exception(
                    "AgenticKV P TP Direct abort mailbox unavailable "
                    "snapshot=%s action=retain_and_retry",
                    entry.request.snapshot_id,
                )
            return False
        entry.tp_abort_mailbox_retry_at = 0.0
        entry.tp_abort_mailbox_retry_delay = 0.05
        return True

    def _agentic_mark_tp_direct_failed(
        self,
        entry: AgenticEarlyDirectReceive,
        *,
        reason: str,
    ) -> None:
        """Defer TP Direct teardown to the scheduler-owner abort command.

        The ingress worker may poll NIXL while a long Prefill forward is in
        flight, but it must never mutate the SGLang GPU allocator or Radix
        tree.  Rank-local failure is reduced through the TP mailbox; rank 0
        then broadcasts one abort command and every rank tears down the same
        request-generation at its next scheduler-safe boundary.
        """

        snapshot_id = entry.request.snapshot_id
        failed = getattr(self, "agentic_tp_direct_local_failed", None)
        if failed is None or snapshot_id in failed:
            return
        failed.add(snapshot_id)
        logger.warning(
            "AgenticKV tp_direct_defer_abort snapshot=%s reason=%s",
            snapshot_id,
            reason,
        )

    def _agentic_progress_tp_direct_grants(self, snapshot_store) -> None:
        """Start TP Direct shards from TP0's background mailbox grant.

        This path intentionally uses no distributed collective and never
        waits for a Prefill scheduler iteration.  TP0 publishes an exact
        request-generation receipt in tmpfs; every rank independently starts
        only that granted shard.  The existing per-rank mailbox statuses form
        the completion barrier.
        """

        mailbox = getattr(self, "agentic_tp_direct_mailbox", None)
        if mailbox is None:
            return
        poll_lock = getattr(self, "agentic_early_direct_poll_lock", nullcontext())
        with poll_lock:
            active = tuple(self.agentic_tp_direct_admission_active.items())
        for snapshot_id, active_item in active:
            request, arrived_at, prefill_domain, _ = active_item[:4]
            workset_lease = active_item[4]
            receipt = mailbox.receipt(snapshot_id)
            if receipt is None:
                continue
            receipt = int(receipt)
            entry = self.agentic_early_direct_receives.get(snapshot_id)
            if receipt < 0:
                # A peer can fail after this rank has already inserted and
                # pinned its received pages in Radix.  The background worker
                # must not return those pages to the transit pool while the
                # branch still references them.  The next native TP control
                # boundary applies one ordered rollback+drop on every rank.
                if snapshot_id not in getattr(
                    self, "agentic_tp_direct_local_rolled_back", ()
                ):
                    self.agentic_tp_direct_local_failed.add(snapshot_id)
                continue
            if receipt >= 3:
                if entry is not None:
                    entry.group_committed = True
                continue
            start_timeout = max(
                0.1, envs.SGLANG_AGENTIC_KV_DIRECT_HANDSHAKE_TIMEOUT.get()
            )
            if self.tp_rank == 0 and time.time() - arrived_at >= start_timeout:
                # A worker may have completed its last local DMA immediately
                # before this timeout observation.  The rank files are the
                # physical truth; never roll back a fully received group just
                # because TP0 has not published the logical receipt yet.
                group_status = mailbox.group_status(snapshot_id)
                if group_status is not None and int(group_status) >= 3:
                    continue
                self._agentic_abort_tp_direct_grant(
                    request,
                    snapshot_store,
                    reason="background_start_timeout",
                )
                # Whether the lifecycle was released, already CONSUMED, or
                # temporarily unreadable, never start a new receiver from the
                # same stale timeout observation.  A store error is retried
                # from a fresh authoritative read on the next worker cycle.
                continue
            if (
                snapshot_id in self.agentic_early_direct_receives
                or snapshot_id in self.agentic_tp_direct_local_failed
                or snapshot_id in self.agentic_tp_direct_local_admitted
            ):
                continue
            self._agentic_tp_start_direct_shard(
                request,
                arrived_at=arrived_at,
                prefill_domain=prefill_domain,
            )

    def _agentic_abort_tp_direct_grant(
        self,
        request: RequestGeneration,
        snapshot_store,
        *,
        reason: str,
        rolled_back: bool = False,
    ) -> bool:
        """Request or finalize one TP-wide Direct abort.

        The first phase only publishes receipt -1 so every native scheduler
        rolls back its local Radix/pin/workset.  The second phase runs after
        the separate rollback ACK proves every rank quiescent; only then may TP0
        return P_RECEIVED to D and publish terminal receipt -2.
        """

        snapshot_id = request.snapshot_id
        mailbox = self.agentic_tp_direct_mailbox
        poll_lock = getattr(self, "agentic_early_direct_poll_lock", nullcontext())
        with poll_lock:
            active_item = self.agentic_tp_direct_admission_active.get(snapshot_id)
        if active_item is None:
            return False
        if not rolled_back:
            self.agentic_tp_direct_local_failed.add(snapshot_id)
            mailbox.publish_receipt(snapshot_id, -1)
            logger.warning(
                "AgenticKV tp_direct_abort_requested snapshot=%s reason=%s",
                snapshot_id,
                reason,
            )
            return True
        try:
            current = snapshot_store.load(request, require_ready=False)
        except Exception:
            logger.exception(
                "AgenticKV could not reload background Direct grant snapshot=%s",
                snapshot_id,
            )
            return False
        if current is not None and current.state is SnapshotState.CONSUMED:
            entry = self.agentic_early_direct_receives.get(snapshot_id)
            if entry is not None:
                entry.group_committed = True
            mailbox.publish_receipt(snapshot_id, -2)
            return False

        expected_claim_id = (
            "direct-early-tp:"
            f"{os.getenv('SGLANG_AGENTIC_KV_ENGINE_ID', 'prefill')}:"
            f"{snapshot_id}"
        )
        if (
            current is not None
            and current.state is SnapshotState.P_RECEIVED
            and current.claim_id == expected_claim_id
        ):
            try:
                snapshot_store.release_received_direct(
                    current, expected_claim_id
                )
            except Exception:
                logger.exception(
                    "AgenticKV failed to return received TP Direct snapshot=%s",
                    snapshot_id,
                )
                return False
        elif (
            current is not None
            and current.state is SnapshotState.DIRECT_LOADING
            and current.claim_id == expected_claim_id
        ):
            try:
                snapshot_store.release_direct_claim(current, expected_claim_id)
            except Exception:
                logger.exception(
                    "AgenticKV failed to release background Direct claim "
                    "snapshot=%s claim=%s",
                    snapshot_id,
                    expected_claim_id,
                )
        if active_item[4] is not None:
            entry = self.agentic_early_direct_receives.get(snapshot_id)
            self.agentic_p_workset_broker.request_release(
                snapshot_id,
                active_item[4],
                io_attempt=(
                    None if entry is None else getattr(entry, "io_attempt", None)
                ),
            )
        else:
            self.agentic_p_workset_broker.cancel_unstarted(
                snapshot_id,
                owner=AgenticPWorksetLeaseBroker.direct_owner(snapshot_id),
            )
        mailbox.publish_receipt(snapshot_id, -2)
        logger.warning(
            "AgenticKV tp_direct_background_abort_complete snapshot=%s reason=%s "
            "age_seconds=%.3f",
            snapshot_id,
            reason,
            max(0.0, time.time() - active_item[1]),
        )
        return True

    def _agentic_commit_tp_direct_groups(self, snapshot_store) -> None:
        """Publish completed TP Direct groups without waiting for P compute.

        Physical NIXL completion is rank-local.  Each background worker
        reports READY through the exact generation mailbox; rank zero commits
        the request-level lifecycle only after every shard is ready.  This is
        metadata-only and wakes Router/P even when the model scheduler is idle.
        Final Radix insertion remains scheduler-owned.
        """

        mailbox = getattr(self, "agentic_tp_direct_mailbox", None)
        if mailbox is None:
            return
        poll_lock = getattr(self, "agentic_early_direct_poll_lock", nullcontext())
        with poll_lock:
            active = tuple(self.agentic_tp_direct_admission_active.items())
            receives = dict(self.agentic_early_direct_receives)
        for snapshot_id, active_item in active:
            with poll_lock:
                if snapshot_id not in self.agentic_tp_direct_admission_active:
                    continue
                entry = receives.get(snapshot_id)
                if snapshot_id in getattr(
                    self, "agentic_tp_direct_local_rolled_back", ()
                ):
                    mailbox.publish_local_rollback_complete(snapshot_id)
                elif snapshot_id in getattr(
                    self, "agentic_tp_direct_local_failed", ()
                ):
                    mailbox.publish_local_progress(snapshot_id, -1)
                elif snapshot_id in getattr(
                    self, "agentic_tp_direct_local_admitted", ()
                ):
                    mailbox.publish_local_progress(snapshot_id, 5)
                elif entry is not None:
                    mailbox.publish_local_progress(
                        snapshot_id, 3 if entry.completed_at is not None else 2
                    )
        if self.tp_rank != 0:
            return
        for snapshot_id, active_item in active:
            with poll_lock:
                if snapshot_id not in self.agentic_tp_direct_admission_active:
                    continue
            entry = receives.get(snapshot_id)
            receipt = mailbox.receipt(snapshot_id)
            if receipt is not None and int(receipt) < 0:
                if int(receipt) <= -2:
                    continue
                if mailbox.rollback_group_complete(snapshot_id):
                    _native_scheduler_class()._agentic_abort_tp_direct_grant(
                        self,
                        active_item[0],
                        snapshot_store,
                        reason="all_ranks_rolled_back",
                        rolled_back=True,
                    )
                continue
            group_status = mailbox.group_status(snapshot_id)
            if group_status is None:
                continue
            group_status = int(group_status)
            if group_status < 0:
                _native_scheduler_class()._agentic_abort_tp_direct_grant(
                    self,
                    active_item[0],
                    snapshot_store,
                    reason="rank_failure",
                )
                continue
            if group_status >= 5:
                with poll_lock:
                    if snapshot_id in self.agentic_tp_direct_admission_active:
                        mailbox.publish_receipt(snapshot_id, 5)
                continue
            if group_status >= 4:
                current = snapshot_store.load(
                    active_item[0], require_ready=False
                )
                if (
                    current is not None
                    and current.state is SnapshotState.P_RECEIVED
                ):
                    try:
                        snapshot_store.commit_direct_bound(
                            current, current.claim_id
                        )
                    except Exception:
                        logger.exception(
                            "AgenticKV tp_direct_bind_commit_retry snapshot=%s",
                            snapshot_id,
                        )
                        continue
                with poll_lock:
                    if snapshot_id in self.agentic_tp_direct_admission_active:
                        mailbox.publish_receipt(snapshot_id, 4)
                continue
            if group_status < 3 or entry is None or entry.group_committed:
                continue
            completed = snapshot_store.complete_direct_group(
                entry.manifest, entry.claim_id
            )
            if completed.state not in {
                SnapshotState.P_RECEIVED,
                SnapshotState.CONSUMED,
            }:
                continue
            if entry.prefill_domain is not None and not entry.route_published:
                self.agentic_early_claim_store.publish_route(
                    entry.request,
                    route="direct_complete",
                    prefill_domain=entry.prefill_domain,
                    snapshot_tokens=entry.manifest.token_count,
                )
                entry.route_published = True
            entry.group_committed = True
            with poll_lock:
                if snapshot_id not in self.agentic_tp_direct_admission_active:
                    continue
                mailbox.publish_receipt(snapshot_id, 3)
            logger.info(
                "AgenticKV early_direct_group_complete snapshot=%s tokens=%d "
                "arrival_to_group_ms=%.3f",
                snapshot_id,
                entry.manifest.token_count,
                max(0.0, (time.time() - entry.arrived_at) * 1000.0),
            )

    def _agentic_fail_tp_direct_bind(
        self,
        entry: AgenticEarlyDirectReceive,
        req: Req,
        *,
        reason: str,
    ) -> bool:
        """Roll back a local prepare and force one TP-wide abort decision."""

        try:
            self._agentic_rollback_prepared_direct_bind(entry)
        except Exception:
            logger.exception(
                "Failed to roll back TP Direct bind snapshot=%s req=%s",
                entry.request.snapshot_id,
                req.rid,
            )
        self.agentic_tp_direct_local_failed.add(entry.request.snapshot_id)
        self.agentic_tp_direct_mailbox.publish_local_progress(
            entry.request.snapshot_id, -1
        )
        logger.error(
            "AgenticKV tp_direct_bind_failed snapshot=%s req=%s reason=%s",
            entry.request.snapshot_id,
            req.rid,
            reason,
        )
        return True

    def _agentic_v2_prepare_activation_control(self):
        """Drain endpoint action tickets on TP0 for the native TP broadcast.

        TCP completion order is not a scheduler order.  The group controller
        emits a ticket only after every target shard is staged; TP0 then puts
        that immutable order on SGLang's existing request broadcast.  No
        scheduler rank polls the network or independently selects ready work.

        A Decode activation also needs one native request-to-token row.  The
        transfer authority can stage more complete worksets than the native
        scheduler can run concurrently (notably c256 with two Decode groups),
        so keep excess activation tickets pending until a row is free.  Source
        release tickets must continue to pass the gate: they are what retire
        existing scheduler ownership and can make rows available on the next
        loop.
        """

        runtime = getattr(self, "agentic_multinode_runtime_v2", None)
        if runtime is None or self.tp_rank != 0:
            return None

        deferred = getattr(
            self, "_agentic_v2_deferred_activation_tickets", None
        )
        if deferred is None:
            deferred = deque()
            self._agentic_v2_deferred_activation_tickets = deferred

        role = getattr(self.disaggregation_mode, "value", self.disaggregation_mode)
        request_pool = getattr(self, "req_to_token_pool", None)
        gate_decode_rows = str(role) == "decode" and request_pool is not None
        activation_budget = (
            int(request_pool.available_size()) if gate_decode_rows else 64
        )

        def encode(ticket):
            return {
                "request_id": ticket.key.request_id,
                "generation": int(ticket.target_generation),
                "attempt": int(ticket.attempt),
                "lease_id": ticket.lease_id,
                "target_role": ticket.target_role,
                "source_generation": int(ticket.key.generation),
                "action": ticket.action,
            }

        tickets = []
        while deferred and activation_budget > 0 and len(tickets) < 64:
            tickets.append(encode(deferred.popleft()))
            activation_budget -= 1

        # Bound queue work per scheduler iteration just as before.  Even when
        # Decode has no request row, drain staged tickets so independent source
        # releases are never hidden behind target activations.
        drained = 0
        while drained < 64 and len(tickets) < 64:
            try:
                ticket = runtime.take_activation_ticket(timeout=0)
            except queue.Empty:
                break
            drained += 1
            if gate_decode_rows and ticket.action == "activate":
                # Preserve activation order.  New activations cannot pass an
                # older deferred activation even if a row appeared between
                # queue reads.
                if deferred or activation_budget <= 0:
                    deferred.append(ticket)
                    continue
                activation_budget -= 1
            tickets.append(encode(ticket))
        if not tickets:
            return None
        return {self._AGENTIC_V2_ACTIVATION_KEY: True, "tickets": tickets}

    def _agentic_v2_consume_activation_control(self, control) -> None:
        """Apply TP0's exact activation sequence on every local TP rank."""

        from sglang.srt.disaggregation.agentic_group_protocol import GenerationKey
        from sglang.srt.disaggregation.agentic_memory_authority import (
            RequestGenerationAttempt,
        )
        from sglang.srt.disaggregation.agentic_multinode_runtime import (
            EndpointActivationTicket,
        )

        runtime = getattr(self, "agentic_multinode_runtime_v2", None)
        if runtime is None:
            raise RuntimeError("received a V2 activation without a V2 runtime")
        role = self.disaggregation_mode.value
        attempts = []
        tickets = []
        for value in control.get("tickets", ()):
            if value["target_role"] != role:
                raise RuntimeError("V2 activation reached the wrong endpoint role")
            key = GenerationKey(
                runtime.config.run_id,
                str(value["request_id"]),
                int(value["source_generation"]),
            )
            ticket = EndpointActivationTicket(
                key=key,
                attempt=int(value["attempt"]),
                lease_id=str(value["lease_id"]),
                target_role=str(value["target_role"]),
                target_generation=int(value["generation"]),
                action=str(value.get("action", "activate")),
            )
            runtime.activate_staged(ticket)
            tickets.append(ticket)
            if ticket.action == "activate":
                attempts.append(
                    RequestGenerationAttempt(
                        str(value["request_id"]),
                        int(value["generation"]),
                        int(value["attempt"]),
                    )
                )

        if attempts:
            if role == "prefill":
                ready = runtime.p_memory_bridge.activate_prefill_attempts(attempts)
                self._merge_disagg_prefill_ready(list(ready))
            elif role == "decode":
                ready = runtime.d_memory_bridge.activate_decode_attempts(attempts)
                self.waiting_queue.extend(ready)
            else:  # pragma: no cover - validated at startup
                raise RuntimeError(f"unsupported V2 endpoint role {role!r}")
        # ACTIVATED means the request is now owned by the native scheduler,
        # not merely published in a rank-local ready queue.
        for ticket in tickets:
            runtime.confirm_scheduler_adopted(ticket)

    def _agentic_tp_reduce_direct_status(self) -> None:
        """Report local Direct progress; TP0 derives logical completion."""

        if not getattr(self, "agentic_tp_direct_command_visible", False):
            return
        active = getattr(self, "agentic_tp_direct_admission_active", {})
        visible_order = list(getattr(self, "agentic_tp_direct_visible_order", ()))
        if not visible_order:
            self.agentic_tp_direct_command_visible = False
            return
        mailbox = self.agentic_tp_direct_mailbox
        for snapshot_id in visible_order:
            item = active.get(snapshot_id)
            local_status = 0
            if item is None:
                mailbox.publish_local_progress(snapshot_id, -1)
                continue
            request = item[0]
            if request.snapshot_id in getattr(
                self, "agentic_tp_direct_local_rolled_back", ()
            ):
                local_status = 6
            elif request.snapshot_id in getattr(
                self, "agentic_tp_direct_local_failed", ()
            ):
                local_status = -1
            if (
                request.snapshot_id
                in getattr(self, "agentic_tp_direct_local_admitted", ())
                and 0 <= local_status < 6
            ):
                local_status = 5
            entry = getattr(self, "agentic_early_direct_receives", {}).get(
                request.snapshot_id
            )
            if entry is not None and 0 <= local_status < 5:
                local_status = 2
                if entry.completed_at is not None:
                    local_status = 3
                if entry.prepared_req is not None:
                    local_status = 4
            for req, _ in getattr(self, "agentic_kv_waiting_queue", ()):
                metadata = AgenticRequestMetadata.from_req(req)
                parent = metadata.parent if metadata is not None else None
                if parent == request and getattr(
                    req, "_agentic_kv_gate_complete", False
                ):
                    local_status = 5
                    break
            if local_status == 6:
                mailbox.publish_local_rollback_complete(snapshot_id)
            else:
                mailbox.publish_local_progress(snapshot_id, local_status)
        if self.tp_rank == 0:
            self.agentic_tp_direct_group_status = {
                snapshot_id: int(status)
                for snapshot_id in visible_order
                if (status := mailbox.group_status(snapshot_id)) is not None
            }

    def _agentic_tp_reduce_host_status(self) -> None:
        """Report every pipelined slow restore; TP0 reads each shard set."""

        if not getattr(self, "agentic_tp_host_command_visible", False):
            return
        active_requests = getattr(self, "agentic_tp_host_active_requests", {})
        visible = list(getattr(self, "_agentic_tp_host_actions", {}))
        if not visible:
            return
        mailbox = self.agentic_tp_host_mailbox
        statuses = {}
        host_staging = getattr(self, "agentic_host_staging_manager", None)
        waiting = tuple(getattr(self, "agentic_kv_waiting_queue", ()))
        for snapshot_id in visible:
            request = active_requests.get(snapshot_id)
            local_status = 0
            if request is None:
                mailbox.publish_local(snapshot_id, local_status)
                continue
            cancelled_rid = getattr(
                self, "agentic_tp_host_cancelled_requests", {}
            ).get(snapshot_id)
            if cancelled_rid is not None:
                # The HTTP waiter has left, but physical abort progress and
                # allocator retirement still own their original fences. Only
                # retire the obsolete control command after both have drained.
                if host_staging.tp_host_control_quiescent(
                    snapshot_id, cancelled_rid
                ):
                    local_status = 4
                mailbox.publish_local(snapshot_id, local_status)
                if self.tp_rank == 0:
                    status = mailbox.group_status(snapshot_id)
                    if status is not None:
                        statuses[snapshot_id] = int(status)
                continue
            if request.snapshot_id in getattr(
                self, "agentic_tp_host_local_admitted", ()
            ):
                local_status = 4
            for req, _ in waiting:
                metadata = AgenticRequestMetadata.from_req(req)
                parent = metadata.parent if metadata is not None else None
                if parent != request:
                    continue
                if getattr(req, "_agentic_tp_host_failed", False):
                    local_status = -1
                    break
                if getattr(req, "_agentic_host_rank_loaded", False):
                    local_status = max(local_status, 3)
                if host_staging is not None and req.rid in host_staging.loads:
                    load = host_staging.loads[req.rid]
                    local_status = max(
                        local_status,
                        2 if load.get("io_complete") else 1,
                    )
                if getattr(req, "_agentic_kv_gate_complete", False):
                    local_status = max(local_status, 3)
                break
            mailbox.publish_local(snapshot_id, local_status)
            if self.tp_rank == 0:
                status = mailbox.group_status(snapshot_id)
                if status is not None:
                    statuses[snapshot_id] = int(status)
        if self.tp_rank == 0:
            self.agentic_tp_host_group_statuses.update(statuses)
            self.agentic_tp_host_group_status = (
                0 if not visible else statuses.get(visible[0], 0)
            )

    def _agentic_tp_reduce_workset_retire_status(self) -> None:
        """Report local fence safety for TP0-authored workset tombstones."""

        visible = tuple(
            getattr(self, "agentic_tp_workset_retire_visible", ())
        )
        if not visible:
            return
        mailbox = self.agentic_tp_workset_retire_mailbox
        broker = self.agentic_p_workset_broker
        statuses = {}
        for snapshot_id in visible:
            mailbox.publish_local(
                snapshot_id,
                1 if broker.tp_retire_ready(snapshot_id) else 0,
            )
            if self.tp_rank == 0:
                status = mailbox.group_status(snapshot_id)
                if status is not None:
                    statuses[snapshot_id] = int(status)
        if self.tp_rank == 0:
            self.agentic_tp_workset_retire_group_statuses.update(statuses)

    @staticmethod
    def _agentic_tp_host_next_action(group_status: int) -> str:
        """Map the all-rank minimum status to one group-owned transition."""

        if int(group_status) < 0:
            return "abort"
        if int(group_status) >= 4:
            return "clear"
        if int(group_status) >= 3:
            return "commit"
        if int(group_status) >= 2:
            return "bind"
        if int(group_status) >= 1:
            return "start"
        return "prepare"

    def _agentic_tp_prepare_admission_control(self):
        """Build TP0's admission command for the native request broadcast."""

        if getattr(self, "tp_size", 1) <= 1 or self.tp_rank != 0:
            return None

        if self.disaggregation_mode is DisaggregationMode.DECODE:
            offload_manager = getattr(self, "decode_offload_manager", None)
            snapshot_id = (
                None
                if offload_manager is None
                else offload_manager.tp_pending_release_snapshot()
            )
            # P-ready is a rank-external filesystem event.  Two TP ranks can
            # observe its creation/deletion on adjacent scheduler ticks, so
            # it must not directly decide which rank allocates D pages.  TP0
            # snapshots the exact ready request ids here and carries that
            # decision on the native request broadcast.
            decode_admit_keys = []
            prealloc_queue = getattr(self, "disagg_decode_prealloc_queue", None)
            transfer_queue = getattr(self, "disagg_decode_transfer_queue", None)
            decode_transfer_keys = []
            if transfer_queue is not None:
                # A logical D engine owns one ordered transfer queue.  Broadcast
                # the whole bounded queue (max_transfer_inflight) so every TP
                # rank advances the same shards in the same order.  Selecting
                # only the local head can deadlock a multi-P/multi-D topology:
                # each P and D may choose a different request, leaving no
                # sender/receiver pair progressing.
                decode_transfer_keys = [
                    (str(entry.req.rid), int(entry.req.bootstrap_room))
                    for entry in transfer_queue.queue
                ]
            decode_transfer_statuses = []
            decode_transfer_cancel_keys = []
            receiver_mailbox = self.agentic_tp_p2d_receiver_mailbox
            for rid, room in decode_transfer_keys:
                key = request_generation_key(rid, room)
                status, cancel_requested = receiver_mailbox.transfer_group_status(key)
                if cancel_requested:
                    decode_transfer_cancel_keys.append((rid, room))
                logical_status = (
                    int(KVPoll.Transferring) if status is None else int(status)
                )
                decode_transfer_statuses.append(logical_status)
                if logical_status in (int(KVPoll.Success), int(KVPoll.Failed)):
                    receiver_mailbox.publish_receipt(key, logical_status)
            decode_transfer_rid = (
                None if not decode_transfer_keys else decode_transfer_keys[0][0]
            )
            decode_transfer_room = (
                None if not decode_transfer_keys else decode_transfer_keys[0][1]
            )
            previous_transfer_rid = getattr(
                self, "_agentic_tp_debug_decode_transfer_rid", None
            )
            if decode_transfer_rid != previous_transfer_rid:
                logger.info(
                    "AgenticKV tp_p2d_decode_select old=%s new=%s engine=%s transfer_queue=%d",
                    previous_transfer_rid,
                    decode_transfer_rid,
                    os.environ.get("SGLANG_AGENTIC_KV_ENGINE_ID", ""),
                    0 if transfer_queue is None else len(transfer_queue.queue),
                )
                self._agentic_tp_debug_decode_transfer_rid = decode_transfer_rid
            p_ready_dir = getattr(prealloc_queue, "p_ready_dir", "")
            if prealloc_queue is not None:
                limit = int(getattr(prealloc_queue, "max_transfer_inflight", 0))
                if limit <= 0:
                    limit = len(prealloc_queue.queue)
                available = max(
                    0,
                    limit
                    - len(self.disagg_decode_transfer_queue.queue)
                    - int(getattr(prealloc_queue, "_async_metadata_pending_count", 0)),
                )
                for decode_req in prealloc_queue.queue:
                    if len(decode_admit_keys) >= available:
                        break
                    if not decode_req.waiting_for_input:
                        continue
                    if (
                        prealloc_queue._requires_p_ready(decode_req)
                        and not os.path.exists(
                            os.path.join(
                                p_ready_dir,
                                f"{decode_req.req.bootstrap_room}.ready",
                            )
                        )
                    ):
                        continue
                    admission_mailbox = getattr(
                        self, "agentic_tp_p2d_admission_mailbox", None
                    )
                    if admission_mailbox is not None:
                        key = request_generation_key(
                            decode_req.req.rid,
                            decode_req.req.bootstrap_room,
                        )
                        if admission_mailbox.group_status(key) != int(KVPoll.Success):
                            continue
                    decode_admit_keys.append(
                        (
                            str(decode_req.req.rid),
                            int(decode_req.req.bootstrap_room),
                        )
                    )
            return {
                self._AGENTIC_TP_CONTROL_KEY: True,
                "decode_release_snapshot": (
                    None if snapshot_id is None else str(snapshot_id)
                ),
                "decode_admit_keys": decode_admit_keys,
                "decode_transfer_keys": decode_transfer_keys,
                "decode_transfer_statuses": decode_transfer_statuses,
                "decode_transfer_cancel_keys": decode_transfer_cancel_keys,
                "decode_transfer_rid": decode_transfer_rid,
                "decode_transfer_room": decode_transfer_room,
                "decode_agentic_commands": (
                    []
                    if offload_manager is None
                    else getattr(offload_manager, "tp_candidate_commands", lambda: [])()
                ),
            }

        if self.disaggregation_mode is not DisaggregationMode.PREFILL:
            return None
        active_direct = getattr(self, "agentic_tp_direct_admission_active", {})
        direct_commands = []
        direct_mailbox = getattr(self, "agentic_tp_direct_mailbox", None)
        for snapshot_id, active in tuple(active_direct.items()):
            direct_request, direct_arrived_at, direct_domain, _ = active[:4]
            receipt = (
                None if direct_mailbox is None else direct_mailbox.receipt(snapshot_id)
            )
            entry = self.agentic_early_direct_receives.get(snapshot_id)
            if receipt is not None and int(receipt) <= -2:
                direct_action = "clear"
            elif receipt is not None and int(receipt) < 0:
                direct_action = "abort"
            elif receipt is not None and int(receipt) >= 5:
                direct_action = "clear"
            elif (
                receipt is not None
                and int(receipt) >= 4
                and entry is not None
                and entry.prepared_req is not None
            ):
                direct_action = "commit_bind"
            elif (
                receipt is not None
                and int(receipt) >= 3
                and entry is not None
                and entry.group_committed
            ):
                direct_action = "prepare_bind"
            else:
                direct_action = "poll"
            direct_commands.append(
                {
                    "snapshot": snapshot_id,
                    "request_id": direct_request.request_id,
                    "generation": direct_request.generation,
                    "action": direct_action,
                    "arrived_at": direct_arrived_at,
                    "domain": direct_domain,
                    "required_tokens": int(active[3]),
                }
            )
        prefill_transfer_keys = []
        # Background workers own transport progress, but SGLang's native TP
        # scheduler broadcast remains the sole authority for a logical group
        # completion.  This keeps page release and request retirement on the
        # same scheduler iteration on every rank.
        tp_p2d_background = bool(
            getattr(self, "_prefill_transfer_tp_background_enabled", False)
        )
        prefill_inflight = getattr(self, "disagg_prefill_inflight_queue", None)
        if prefill_inflight:
            # TP0 owns the P-ready FIFO and broadcasts its complete ordered
            # transfer set.  Followers never select independently.  Advancing
            # all entries is necessary when different entries have been routed
            # to different D engines; a single local head can be waiting for D0
            # while D0 is polling an older entry produced by another P.
            prefill_transfer_keys = [
                (str(req.rid), int(req.bootstrap_room)) for req in prefill_inflight
            ]
        prefill_transfer_statuses = []
        prefill_submit_keys = []
        submit_limit = max(
            1, int(os.getenv("SGLANG_PREFILL_TRANSFER_TP_SUBMIT_BATCH", "24"))
        )
        sender_mailbox = self.agentic_tp_p2d_sender_mailbox
        receiver_mailbox = self.agentic_tp_p2d_receiver_mailbox
        p2d_host = getattr(self, "agentic_p2d_host_staging_manager", None)
        for index, (rid, room) in enumerate(prefill_transfer_keys):
            req = prefill_inflight[index]
            key = request_generation_key(rid, room)
            raw_sender_status = sender_mailbox.group_status(key)
            sender_status, _ = sender_mailbox.transfer_group_status(key)
            # Every TP rank has prepared an immutable local sender payload.
            # Only now may TP0 expose one logical P-ready marker to Router/D.
            # This ordering prevents a transient rank-local preparation delay
            # from permanently splitting the TP group.
            if (
                raw_sender_status is not None
                and raw_sender_status >= int(KVPoll.Bootstrapping)
                and not getattr(req, "disagg_p_ready_notified", False)
            ):
                self._publish_deferred_prefill_ready(req)
            host_path = bool(
                getattr(req, "_agentic_p2d_host_snapshot_id", None)
                or (p2d_host is not None and p2d_host.group_claimed(req))
            )
            receipt_status = receiver_mailbox.receipt(key)
            if index == 0:
                head_state = (
                    key,
                    sender_status,
                    receipt_status,
                    bool(getattr(req, "disagg_p_ready_transfer_started", False)),
                    host_path,
                )
                if head_state != getattr(
                    self, "_agentic_tp_debug_prefill_head_state", None
                ):
                    logger.info(
                        "AgenticKV tp_p2d_prefill_head key=%s sender=%s "
                        "receipt=%s started=%s host=%s",
                        *head_state,
                    )
                    self._agentic_tp_debug_prefill_head_state = head_state
            if tp_p2d_background:
                # A rank publishes terminal sender status only after its
                # background worker has stopped touching this generation.
                # Requiring the all-rank sender reduction therefore prevents
                # one scheduler from freeing pages while a peer worker still
                # polls or submits its shard.
                logical_status = (
                    int(KVPoll.Transferring)
                    if sender_status not in (int(KVPoll.Success), int(KVPoll.Failed))
                    else int(sender_status)
                )
            elif sender_status == int(KVPoll.Failed):
                logical_status = int(KVPoll.Failed)
            elif req.bootstrap_host == FAKE_BOOTSTRAP_HOST or host_path:
                # A complete Host snapshot is already an authoritative copy;
                # P may release before D finishes its later H2D restore.
                logical_status = (
                    int(KVPoll.Transferring)
                    if sender_status is None
                    else int(sender_status)
                )
            else:
                # Native Direct success is destination-authored.  A stale
                # sender handle can never pin P once all D shards have ACKed.
                logical_status = (
                    int(KVPoll.Transferring)
                    if receipt_status is None
                    else int(receipt_status)
                )
                if (
                    not tp_p2d_background
                    and len(prefill_submit_keys) < submit_limit
                    and raw_sender_status == int(KVPoll.WaitingForInput)
                    and not getattr(req, "disagg_p_ready_transfer_started", False)
                ):
                    prefill_submit_keys.append((rid, room))
            prefill_transfer_statuses.append(logical_status)
        prefill_transfer_rid = (
            None if not prefill_transfer_keys else prefill_transfer_keys[0][0]
        )
        prefill_transfer_room = (
            None if not prefill_transfer_keys else prefill_transfer_keys[0][1]
        )
        previous_transfer_rid = getattr(
            self, "_agentic_tp_debug_prefill_transfer_rid", None
        )
        if prefill_transfer_rid != previous_transfer_rid:
            logger.info(
                "AgenticKV tp_p2d_prefill_select old=%s new=%s domain=%s inflight=%d",
                previous_transfer_rid,
                prefill_transfer_rid,
                os.environ.get("SGLANG_AGENTIC_KV_PREFILL_DOMAIN", "0"),
                0 if prefill_inflight is None else len(prefill_inflight),
            )
            self._agentic_tp_debug_prefill_transfer_rid = prefill_transfer_rid
        previous_submit_keys = getattr(
            self, "_agentic_tp_debug_prefill_submit_keys", None
        )
        if prefill_submit_keys != previous_submit_keys:
            logger.info(
                "AgenticKV tp_p2d_prefill_submit_select old=%s new=%s statuses=%s",
                previous_submit_keys,
                prefill_submit_keys,
                prefill_transfer_statuses,
            )
            self._agentic_tp_debug_prefill_submit_keys = list(prefill_submit_keys)
        host_commands = []
        host_timeout_snapshot = None
        host_staging = getattr(self, "agentic_host_staging_manager", None)
        if host_staging is not None:
            active_host = self.agentic_tp_host_active_requests
            active_since = self.agentic_tp_host_active_since_by_snapshot
            try:
                requested_host_pipeline_depth = max(
                    1,
                    int(
                        os.getenv(
                            "SGLANG_AGENTIC_KV_TP_HOST_PIPELINE_DEPTH",
                            os.getenv("SGLANG_AGENTIC_KV_P_H2D_MAX_INFLIGHT", "1"),
                        )
                    ),
                )
            except ValueError:
                logger.exception("Invalid TP Host pipeline depth")
                raise
            # The Host manager currently owns one process-lifetime pinned
            # bounce buffer.  Never broadcast more independent snapshots than
            # the physical loader can make progress on; opposite rank order
            # could otherwise reserve A/B and deadlock the TP group.
            host_pipeline_depth = min(
                requested_host_pipeline_depth,
                int(getattr(host_staging, "max_h2d_inflight", 1)),
            )
            # Fill the bounded pipeline in request arrival order.  This only
            # chooses request-generations; every rank still allocates, copies,
            # and binds the exact same command through the native TP broadcast.
            if len(active_host) < host_pipeline_depth:
                for req, _ in self.agentic_kv_waiting_queue:
                    metadata = AgenticRequestMetadata.from_req(req)
                    parent = metadata.parent if metadata is not None else None
                    if (
                        parent is None
                        or parent.snapshot_id in active_host
                        or not host_staging.snapshot_ready(parent)
                    ):
                        continue
                    active_host[parent.snapshot_id] = parent
                    active_since[parent.snapshot_id] = time.monotonic()
                    if len(active_host) >= host_pipeline_depth:
                        break
            group_statuses = self.agentic_tp_host_group_statuses
            for snapshot_id, host_request in tuple(active_host.items()):
                host_status = int(group_statuses.get(snapshot_id, 0))
                if snapshot_id in getattr(
                    self, "agentic_tp_host_cancelled_requests", {}
                ):
                    # A stale peer status of 1/2/3 is not permission to start
                    # or commit an attempt that TP0 has already cancelled.
                    host_action = "clear" if host_status >= 4 else "abort"
                else:
                    host_action = self._agentic_tp_host_next_action(host_status)
                if host_action == "commit":
                    # Every rank has restored its physical shard. TP0 alone
                    # closes the logical slow-path manifest before the group
                    # admission command is broadcast.
                    try:
                        if not host_staging._complete_shared_host_manifest(
                            host_request
                        ):
                            continue
                    except Exception:
                        logger.exception(
                            "AgenticKV tp_shared_host_manifest_commit_retry "
                            "snapshot=%s",
                            snapshot_id,
                        )
                        continue
                host_commands.append(
                    {
                        "snapshot": snapshot_id,
                        "request_id": host_request.request_id,
                        "generation": host_request.generation,
                        "action": host_action,
                    }
                )
                if host_action == "clear":
                    active_host.pop(snapshot_id, None)
                    active_since.pop(snapshot_id, None)
                    group_statuses.pop(snapshot_id, None)
            # A timeout is diagnostic, not an eviction policy.  TP ranks keep
            # the metadata-only child queued until Host recovery succeeds or
            # an explicit request-generation eviction/cancel is published.
        workset_plan_epoch = int(
            getattr(self, "_agentic_tp_workset_plan_epoch", 0)
        ) + 1
        self._agentic_tp_workset_plan_epoch = workset_plan_epoch
        workset_broker = getattr(self, "agentic_p_workset_broker", None)
        retire_active = getattr(self, "agentic_tp_workset_retire_active", None)
        if retire_active is None:
            retire_active = set()
            self.agentic_tp_workset_retire_active = retire_active
        retire_group_statuses = getattr(
            self, "agentic_tp_workset_retire_group_statuses", None
        )
        if retire_group_statuses is None:
            retire_group_statuses = {}
            self.agentic_tp_workset_retire_group_statuses = retire_group_statuses
        workset_plan = ()
        if workset_broker is not None and hasattr(
            workset_broker, "prepare_tp_control"
        ):
            workset_plan, frozen_retirements, handoff_commits = (
                workset_broker.prepare_tp_control(
                    workset_plan_epoch,
                    retiring_ids=tuple(retire_active),
                )
            )
            retire_active.update(frozen_retirements)
        else:
            handoff_commits = ()
        retire_commands = []
        for snapshot_id in tuple(retire_active):
            ready = int(retire_group_statuses.get(snapshot_id, 0)) >= 1
            retire_commands.append(
                {
                    "snapshot": snapshot_id,
                    "action": "commit" if ready else "prepare",
                }
            )
            if ready:
                retire_active.discard(snapshot_id)
                retire_group_statuses.pop(snapshot_id, None)
        return {
            self._AGENTIC_TP_CONTROL_KEY: True,
            "workset_plan_epoch": workset_plan_epoch,
            "workset_allocation_plan": workset_plan,
            "workset_handoff_commits": handoff_commits,
            "workset_retire_commands": retire_commands,
            "direct_commands": direct_commands,
            "direct_snapshot": (
                None if not direct_commands else direct_commands[0]["snapshot"]
            ),
            "direct_request_id": (
                None if not direct_commands else direct_commands[0]["request_id"]
            ),
            "direct_generation": (
                None if not direct_commands else direct_commands[0]["generation"]
            ),
            "direct_action": (
                None if not direct_commands else direct_commands[0]["action"]
            ),
            "prefill_transfer_keys": prefill_transfer_keys,
            "prefill_transfer_statuses": prefill_transfer_statuses,
            "prefill_submit_keys": prefill_submit_keys,
            "prefill_transfer_rid": prefill_transfer_rid,
            "prefill_transfer_room": prefill_transfer_room,
            "host_commands": host_commands,
            # Scalar aliases keep older diagnostics/tests readable.
            "host_snapshot": (
                None if not host_commands else host_commands[0]["snapshot"]
            ),
            "host_request_id": (
                None if not host_commands else host_commands[0]["request_id"]
            ),
            "host_generation": (
                None if not host_commands else host_commands[0]["generation"]
            ),
            "host_action": (None if not host_commands else host_commands[0]["action"]),
            "host_timeout_snapshot": host_timeout_snapshot,
            "direct_arrived_at": (
                0.0 if not direct_commands else direct_commands[0]["arrived_at"]
            ),
            "direct_domain": (
                None if not direct_commands else direct_commands[0]["domain"]
            ),
        }

    def _agentic_tp_consume_admission_control(self, recv_reqs):
        """Apply and remove TP admission metadata from native recv traffic."""

        if not recv_reqs or getattr(self, "tp_size", 1) <= 1:
            return recv_reqs
        ordinary = []
        control = None
        v2_control = None
        v1_key = _native_scheduler_class()._AGENTIC_TP_CONTROL_KEY
        v2_key = _native_scheduler_class()._AGENTIC_V2_ACTIVATION_KEY
        for req in recv_reqs:
            if isinstance(req, dict):
                if req.get(v2_key):
                    v2_control = req
                    continue
                if req.get(v1_key):
                    control = req
                    continue
            ordinary.append(req)
        if v2_control is not None:
            if control is not None:
                raise RuntimeError("V1 and V2 TP controls were mixed")
            self._agentic_v2_consume_activation_control(v2_control)
            # The native recv broadcast gives every TP rank the same ticket
            # order, but applying a ticket performs rank-local allocator and
            # Radix mutations.  Those commits can take different amounts of
            # time (or briefly wait on a rank-local authority lock).  Without
            # a fence, a fast rank can schedule the newly activated request
            # and enter the next model collective while a slower rank is
            # still committing it, producing divergent forward batches.
            #
            # This barrier is ordered by the broadcast above and is entered
            # only when that broadcast carried V2 endpoint work, so every TP
            # rank takes it in the same scheduler-loop iteration.  DMA remains
            # asynchronous; only the short source-release/target-publication
            # metadata transaction is fenced before compute resumes.
            barrier(group=self.tp_cpu_group)
            return ordinary
        if control is None:
            return ordinary
        if self.disaggregation_mode is DisaggregationMode.PREFILL:
            handoff_commits = tuple(
                str(snapshot_id)
                for snapshot_id in control.get("workset_handoff_commits", ())
            )
            retire_commands = control.get("workset_retire_commands", ())
            self.agentic_p_workset_broker.install_tp_plan(
                int(control["workset_plan_epoch"]),
                control.get("workset_allocation_plan", ()),
                retiring_ids=tuple(
                    str(command["snapshot"]) for command in retire_commands
                ),
            )
            for snapshot_id in handoff_commits:
                if not self.agentic_p_workset_broker.commit_tp_handoff(
                    snapshot_id
                ):
                    raise RuntimeError(
                        "TP workset handoff commit reached a non-handed local "
                        f"lease for {snapshot_id}"
                    )
            retire_visible = getattr(
                self, "agentic_tp_workset_retire_visible", None
            )
            if retire_commands and retire_visible is None:
                retire_visible = set()
                self.agentic_tp_workset_retire_visible = retire_visible
            retire_mailbox = getattr(
                self, "agentic_tp_workset_retire_mailbox", None
            )
            for command in retire_commands:
                snapshot_id = str(command["snapshot"])
                action = str(command["action"])
                if action == "prepare":
                    self.agentic_p_workset_broker.prepare_tp_retire(snapshot_id)
                    retire_visible.add(snapshot_id)
                elif action == "commit":
                    if not self.agentic_p_workset_broker.commit_tp_retire(
                        snapshot_id
                    ):
                        raise RuntimeError(
                            "TP workset retire commit reached an unsafe local "
                            f"lease for {snapshot_id}"
                        )
                    retire_visible.discard(snapshot_id)
                    if retire_mailbox is not None:
                        retire_mailbox.clear_local(snapshot_id)
                    if self.tp_rank == 0 and retire_mailbox is not None:
                        retire_mailbox.clear_group(snapshot_id)
                else:
                    raise RuntimeError(
                        f"unknown TP workset retire action {action}"
                    )
        decode_release_snapshot = control.get("decode_release_snapshot")
        if decode_release_snapshot is not None:
            offload_manager = getattr(self, "decode_offload_manager", None)
            if offload_manager is None:
                raise RuntimeError("TP Decode release lost its offload manager")
            offload_manager.commit_tp_release(str(decode_release_snapshot))
        if self.disaggregation_mode is DisaggregationMode.DECODE:
            offload_manager = getattr(self, "decode_offload_manager", None)
            if offload_manager is not None:
                apply_commands = getattr(
                    offload_manager, "apply_tp_candidate_commands", None
                )
                if apply_commands is not None:
                    apply_commands(control.get("decode_agentic_commands", ()))
            # Admission is a rank-local allocator commit.  A follower can still
            # have the previous tick's terminal transfer entries in its local
            # queue when TP0 selects the next requests.  In that case
            # DecodePreallocQueue intentionally defers the selected request for
            # one tick.  Keep the broadcast key pending on that rank until the
            # local commit consumes it; replacing the list on the next native
            # broadcast would otherwise strand the request forever after TP0
            # has already moved its shard into the transfer queue.
            new_admit_keys = [
                (str(rid), int(room))
                for rid, room in control.get("decode_admit_keys", ())
            ]
            pending_admit_keys = []
            pending_admit_set = set()
            for key in (
                *getattr(self, "_agentic_tp_decode_admit_keys", ()),
                *new_admit_keys,
            ):
                key = (str(key[0]), int(key[1]))
                if key not in pending_admit_set:
                    pending_admit_keys.append(key)
                    pending_admit_set.add(key)
            self._agentic_tp_decode_admit_keys = pending_admit_keys
            transfer_keys = control.get("decode_transfer_keys")
            if transfer_keys is None:
                transfer_rid = control.get("decode_transfer_rid")
                transfer_room = control.get("decode_transfer_room")
                transfer_keys = (
                    []
                    if transfer_rid is None or transfer_room is None
                    else [(transfer_rid, transfer_room)]
                )
            self._agentic_tp_decode_transfer_keys = [
                (str(rid), int(room)) for rid, room in transfer_keys
            ]
            self._agentic_tp_decode_transfer_group_status = {
                (str(rid), int(room)): int(status)
                for (rid, room), status in zip(
                    self._agentic_tp_decode_transfer_keys,
                    control.get("decode_transfer_statuses", ()),
                )
            }
            transfer_queue = getattr(self, "disagg_decode_transfer_queue", None)
            cancel_keys = control.get("decode_transfer_cancel_keys", ())
            if transfer_queue is not None and cancel_keys:
                transfer_queue.abort_agentic_host_transfers(cancel_keys)
            return ordinary
        direct_commands = control.get("direct_commands")
        if direct_commands is None:
            snapshot_id = control.get("direct_snapshot")
            direct_commands = (
                []
                if snapshot_id is None
                else [
                    {
                        "snapshot": snapshot_id,
                        "request_id": control["direct_request_id"],
                        "generation": control["direct_generation"],
                        "action": control.get("direct_action"),
                        "arrived_at": control.get("direct_arrived_at", 0.0),
                        "domain": control.get("direct_domain"),
                    }
                ]
            )
        direct_actions = {}
        visible_order = []
        active_direct = self.agentic_tp_direct_admission_active
        group_status = self.agentic_tp_direct_group_status
        direct_poll_lock = getattr(
            self, "agentic_early_direct_poll_lock", nullcontext()
        )
        direct_terminal = getattr(self, "agentic_early_direct_terminal", None)
        if direct_terminal is None:
            direct_terminal = {}
            self.agentic_early_direct_terminal = direct_terminal
        for command in direct_commands:
            snapshot_id = str(command["snapshot"])
            direct_action = command.get("action")
            if direct_action in {"clear", "abort"}:
                with direct_poll_lock:
                    entry = self.agentic_early_direct_receives.get(snapshot_id)
                    active_item = active_direct.get(snapshot_id)
                    # This terminal marker closes the start-vs-abort race.  A
                    # shard that has not begun observes it before claiming;
                    # one already publishing DMA registers an aborting entry
                    # and remains polled until its physical fence arrives.
                    direct_terminal[snapshot_id] = time.monotonic()
                if direct_action == "abort":
                    if entry is not None:
                        self._agentic_rollback_prepared_direct_bind(entry)
                        self._agentic_drop_early_direct_receive(
                            entry,
                            self._agentic_snapshot_store(),
                            release_claim=False,
                            reason="tp_group_abort",
                        )
                        # A partially posted Direct DMA remains registered in
                        # the receive table until its physical fence becomes
                        # terminal.  Do not ACK rollback while those pages are
                        # still owned by transport.
                        with direct_poll_lock:
                            if snapshot_id in self.agentic_early_direct_receives:
                                continue
                    elif active_item is not None and active_item[4] is not None:
                        # No receiver owns the lease yet.  If a concurrent
                        # start already reserved it, request_release refuses
                        # the tokenless release and that start observes the
                        # terminal marker above before registering.
                        self.agentic_p_workset_broker.request_release(
                            snapshot_id, active_item[4]
                        )
                        if not self.agentic_p_workset_broker.tp_retire_ready(
                            snapshot_id, lease_id=active_item[4].lease_id
                        ):
                            # A start can own the transport lease before its
                            # receiver is visible.  Keep the group claim until
                            # that exact attempt reaches its physical fence.
                            continue
                    rolled_back = getattr(
                        self, "agentic_tp_direct_local_rolled_back", None
                    )
                    if rolled_back is None:
                        rolled_back = set()
                        self.agentic_tp_direct_local_rolled_back = rolled_back
                    rolled_back.add(snapshot_id)
                    mailbox = getattr(self, "agentic_tp_direct_mailbox", None)
                    if mailbox is not None:
                        mailbox.publish_local_rollback_complete(snapshot_id)
                    # TP0 returns P_RECEIVED ownership only after every rank
                    # has published this rollback ACK.  Keep the command
                    # active until the background worker publishes receipt -2.
                    continue
                with direct_poll_lock:
                    self.agentic_tp_direct_local_admitted.discard(snapshot_id)
                    self.agentic_tp_direct_local_failed.discard(snapshot_id)
                    getattr(
                        self, "agentic_tp_direct_local_rolled_back", set()
                    ).discard(snapshot_id)
                    if active_direct.get(snapshot_id) is active_item:
                        active_direct.pop(snapshot_id, None)
                    group_status.pop(snapshot_id, None)
                    mailbox = getattr(self, "agentic_tp_direct_mailbox", None)
                    if mailbox is not None:
                        mailbox.clear_local(snapshot_id)
                        mailbox.clear_local_rollback(snapshot_id)
                        if self.tp_rank == 0:
                            mailbox.clear_group(snapshot_id)
                            mailbox.clear_group_rollback(snapshot_id)
                continue
            request = RequestGeneration(
                str(command["request_id"]), int(command["generation"])
            )
            with direct_poll_lock:
                current = active_direct.get(snapshot_id)
                required_tokens = int(
                    command.get(
                        "required_tokens",
                        0 if current is None else int(current[3]),
                    )
                )
                # Logical TP commands never replace rank-local allocator
                # identity.  Background admission uses this same lock, so the
                # merge is a single compare/read/write operation and a None
                # observation cannot overwrite a concurrently granted lease.
                workset_lease = (
                    current[4]
                    if current is not None and current[4] is not None
                    else self.agentic_p_workset_broker.get(
                        snapshot_id,
                        owner=AgenticPWorksetLeaseBroker.direct_owner(snapshot_id),
                    )
                )
                active_direct[snapshot_id] = (
                    request,
                    float(command.get("arrived_at", 0.0)),
                    command.get("domain"),
                    required_tokens,
                    workset_lease,
                )
            visible_order.append(snapshot_id)
            direct_actions[snapshot_id] = direct_action
        self.agentic_tp_direct_visible_order = visible_order
        self.agentic_tp_direct_command_visible = bool(visible_order)
        self._agentic_tp_selected_snapshots = set(visible_order)
        self._agentic_tp_direct_actions = direct_actions
        # Compatibility aliases for focused tests and out-of-tree users that
        # still inspect the former single-command fields.
        self._agentic_tp_selected_snapshot = (
            None if not visible_order else visible_order[0]
        )
        self._agentic_tp_direct_action = (
            None if not visible_order else direct_actions[visible_order[0]]
        )
        prefill_transfer_keys = control.get("prefill_transfer_keys")
        if prefill_transfer_keys is None:
            prefill_transfer_rid = control.get("prefill_transfer_rid")
            prefill_transfer_room = control.get("prefill_transfer_room")
            prefill_transfer_keys = (
                []
                if prefill_transfer_rid is None or prefill_transfer_room is None
                else [(prefill_transfer_rid, prefill_transfer_room)]
            )
        self._agentic_tp_prefill_transfer_keys = [
            (str(rid), int(room)) for rid, room in prefill_transfer_keys
        ]
        self._agentic_tp_prefill_transfer_group_status = {
            (str(rid), int(room)): int(status)
            for (rid, room), status in zip(
                self._agentic_tp_prefill_transfer_keys,
                control.get("prefill_transfer_statuses", ()),
            )
        }
        submit_keys = control.get("prefill_submit_keys")
        if submit_keys is None:
            submit_key = control.get("prefill_submit_key")
            submit_keys = [] if submit_key is None else [submit_key]
        self._agentic_tp_prefill_submit_keys = [
            (str(rid), int(room)) for rid, room in submit_keys
        ]
        host_commands = control.get("host_commands")
        if host_commands is None:
            host_snapshot = control.get("host_snapshot")
            host_commands = (
                []
                if host_snapshot is None
                else [
                    {
                        "snapshot": host_snapshot,
                        "request_id": control["host_request_id"],
                        "generation": control["host_generation"],
                        "action": control.get("host_action"),
                    }
                ]
            )
        active_host = getattr(self, "agentic_tp_host_active_requests", None)
        if active_host is None:
            active_host = {}
            legacy_active = getattr(self, "agentic_tp_host_active", None)
            if legacy_active is not None:
                active_host[legacy_active.snapshot_id] = legacy_active
            self.agentic_tp_host_active_requests = active_host
        active_since = getattr(self, "agentic_tp_host_active_since_by_snapshot", None)
        if active_since is None:
            active_since = {}
            legacy_active = getattr(self, "agentic_tp_host_active", None)
            if legacy_active is not None:
                active_since[legacy_active.snapshot_id] = float(
                    getattr(self, "agentic_tp_host_active_since", 0.0)
                )
            self.agentic_tp_host_active_since_by_snapshot = active_since
        if not hasattr(self, "agentic_tp_host_group_statuses"):
            self.agentic_tp_host_group_statuses = {}
        host_actions = {}
        # Preserve TP0's broadcast order: a bounded admission batch must pick
        # the same generations on every rank, regardless of Python hash seed.
        commit_snapshots = []
        mailbox = getattr(self, "agentic_tp_host_mailbox", None)
        for command in host_commands:
            host_snapshot = str(command["snapshot"])
            host_action = command.get("action")
            if host_action == "clear":
                self.agentic_tp_host_local_admitted.discard(host_snapshot)
                cancelled_rid = getattr(self, "agentic_tp_host_cancelled_requests", {}).pop(
                    host_snapshot, None
                )
                if cancelled_rid is not None:
                    logger.info(
                        "AgenticKV tp_host_cancel_control_retired snapshot=%s rid=%s",
                        host_snapshot, cancelled_rid,
                    )
                active_host.pop(host_snapshot, None)
                active_since.pop(host_snapshot, None)
                self.agentic_tp_host_group_statuses.pop(host_snapshot, None)
                if mailbox is not None:
                    # Each rank removes its own report. TP0 additionally clears
                    # the logical receipt after the native broadcast made the
                    # CLEAR command visible to the complete group.
                    mailbox.clear_local(host_snapshot)
                    if self.tp_rank == 0:
                        mailbox.clear_group(host_snapshot)
                continue
            request = RequestGeneration(
                str(command["request_id"]), int(command["generation"])
            )
            active_host[host_snapshot] = request
            active_since.setdefault(host_snapshot, time.monotonic())
            host_actions[host_snapshot] = host_action
            if host_action == "commit":
                if host_snapshot not in commit_snapshots:
                    commit_snapshots.append(host_snapshot)
        self._agentic_tp_host_actions = host_actions
        self.agentic_tp_host_command_visible = bool(host_actions)
        visible_host = list(host_actions)
        first_host = None if not visible_host else visible_host[0]
        # Compatibility aliases for the former single-snapshot state machine.
        self.agentic_tp_host_active = (
            None if first_host is None else active_host[first_host]
        )
        self.agentic_tp_host_active_since = (
            0.0 if first_host is None else active_since[first_host]
        )
        self.agentic_tp_host_group_status = (
            0
            if first_host is None
            else self.agentic_tp_host_group_statuses.get(first_host, 0)
        )
        self._agentic_tp_host_selected_snapshot = first_host
        self._agentic_tp_host_action = (
            None if first_host is None else host_actions[first_host]
        )
        self._agentic_tp_host_commit_snapshots = tuple(commit_snapshots)
        self._agentic_tp_host_commit_snapshot = (
            None if not commit_snapshots else commit_snapshots[0]
        )
        host_staging = getattr(self, "agentic_host_staging_manager", None)
        if host_staging is not None:
            host_staging.tp_host_commit_snapshots = commit_snapshots
            host_staging.tp_host_commit_snapshot = self._agentic_tp_host_commit_snapshot
        host_timeout_snapshot = control.get("host_timeout_snapshot")
        self._agentic_tp_host_timeout_snapshot = (
            None if host_timeout_snapshot is None else str(host_timeout_snapshot)
        )
        return ordinary

    def _agentic_tp_start_direct_shard(
        self,
        request: RequestGeneration,
        *,
        arrived_at: float,
        prefill_domain,
    ) -> bool:
        """Execute TP0's Direct-start command for this rank's KV-head shard."""

        if request.snapshot_id in getattr(self, "agentic_tp_direct_local_admitted", ()):
            return True
        receives = getattr(self, "agentic_early_direct_receives", {})
        if request.snapshot_id in receives:
            return True
        snapshot_store = self._agentic_snapshot_store()
        if snapshot_store is None:
            self.agentic_tp_direct_local_failed.add(request.snapshot_id)
            return False
        manifest = snapshot_store.load(request, require_ready=False)
        if manifest is None:
            # Mooncake metadata publication can briefly lag the node-local
            # arrival marker.  Keep retrying; _agentic_admit_queued_direct_receives
            # already bounds the lifetime of such markers.
            return False
        if manifest.state not in {
            SnapshotState.DIRECT_READY,
            SnapshotState.DIRECT_LOADING,
        }:
            # D may fall back while this request waits in P's FIFO.  This is
            # a definitive lifecycle transition, not a receiver that can
            # become ready later.  Report it to rank 0 so the group command
            # is aborted and the existing Host/Mooncake path can proceed.
            self.agentic_tp_direct_local_failed.add(request.snapshot_id)
            active_item = getattr(self, "agentic_tp_direct_admission_active", {}).get(
                request.snapshot_id
            )
            entry = getattr(self, "agentic_early_direct_receives", {}).get(
                request.snapshot_id
            )
            self.agentic_p_workset_broker.request_release(
                request.snapshot_id,
                None if active_item is None else active_item[4],
                io_attempt=(
                    None if entry is None else getattr(entry, "io_attempt", None)
                ),
            )
            logger.info(
                "AgenticKV tp_direct_stale_abort snapshot=%s state=%s",
                request.snapshot_id,
                manifest.state.value,
            )
            return False
        workset_lease = self.agentic_p_workset_broker.get(
            request.snapshot_id,
            owner=AgenticPWorksetLeaseBroker.direct_owner(request.snapshot_id),
        )
        if workset_lease is None:
            # The rank-local scheduler has not granted the complete workset
            # yet.  Keep the TP command pending without claiming lifecycle or
            # allocating from the transport worker.
            return False
        started = self._agentic_start_early_direct_receive(
            request,
            manifest,
            snapshot_store,
            arrived_at=arrived_at,
            prefill_domain=(None if prefill_domain is None else int(prefill_domain)),
            workset_lease=workset_lease,
        )
        return bool(started)
