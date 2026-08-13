"""
Life cycle of a request in the prefill server

1. Bootstrap Queue
    a. Initialize a sender for each request
    b. Use the queue to store requests whose bootstrap (handshake and preallocation) has not finished
    c. Poll senders to check bootstrap state
    d. Once bootstrap is complete, move request to Waiting Queue

2. Waiting Queue
    a. Use PrefillAdder to pop requests
    b. Run forward
    c. Add the request to Inflight Queue

3. Inflight Queue
    a. Poll (non-blocking) the sender of the request
    b. Once the transfer has finished, return the request
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from http import HTTPStatus
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    KVClassType,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    get_kv_class,
    is_mla_backend,
    kv_to_page_indices,
    kv_to_page_num,
    poll_and_all_reduce_attn_cp_tp_group,
    prepare_abort,
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    Req,
    ScheduleBatch,
)
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.disaggregation.agentic_kv_lifecycle import AgenticRequestMetadata
from sglang.srt.disaggregation.agentic_direct_transfer import debug_kv_digest
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, NSATokenToKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.observability.req_time_stats import set_schedule_time_batch

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler
    from sglang.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)


def release_req_to_metadata_buffer(
    req: Req, allocator: ReqToMetadataIdxAllocator
) -> None:
    """
    Release the metadata buffer index allocated for a request in prefill disaggregation mode.

    This function safely releases the metadata buffer index if it was allocated.

    Args:
        req: The request object that may have a metadata_buffer_index allocated
        allocator: The ReqToMetadataIdxAllocator instance to free the index
    """
    if (
        hasattr(req, "metadata_buffer_index")
        and req.metadata_buffer_index is not None
        and req.metadata_buffer_index >= 0
    ):
        allocator.free(req.metadata_buffer_index)
        req.metadata_buffer_index = -1


class PrefillBootstrapQueue:
    """
    Store the requests in bootstrapping
    """

    def __init__(
        self,
        token_to_kv_pool: KVCache,
        draft_token_to_kv_pool: Optional[KVCache],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        tp_rank: int,
        tp_size: int,
        gpu_id: int,
        bootstrap_port: int,
        gloo_group: ProcessGroup,
        max_total_num_tokens: int,
        scheduler: Scheduler,
        pp_rank: int,
        pp_size: int,
        transfer_backend: TransferBackend,
    ):
        self.token_to_kv_pool = token_to_kv_pool
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self.is_mla_backend = is_mla_backend(token_to_kv_pool)
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.gpu_id = gpu_id
        self.bootstrap_port = bootstrap_port
        self.queue: List[Req] = []
        self.gloo_group = gloo_group
        self.max_total_num_tokens = max_total_num_tokens
        self.scheduler = scheduler
        self.transfer_backend = transfer_backend
        # Experimental same-host two-phase handshake. When configured, P can
        # compute before D publishes destination KV page indices. Computed
        # source KV stays locked until D observes the ready marker and sends
        # the normal NIXL destination metadata.
        self.p_ready_dir = os.environ.get("SGLANG_PD_P_READY_DIR", "")
        if envs.SGLANG_AGENTIC_KV_LIFECYCLE.get() and not self.p_ready_dir:
            raise ValueError(
                "SGLANG_AGENTIC_KV_LIFECYCLE requires SGLANG_PD_P_READY_DIR "
                "(use a node-local path such as /dev/shm/sglang-agentic-p-ready)"
            )
        if self.p_ready_dir:
            os.makedirs(self.p_ready_dir, exist_ok=True)
        self.kv_manager = self._init_kv_manager()

        if self.scheduler.tp_worker.is_hybrid_swa:
            # FIXME: current SWA allocation allocate full kv cache size in prefill
            self.max_total_num_tokens = min(
                self.max_total_num_tokens,
                self.scheduler.tp_worker.model_runner.swa_max_total_num_tokens,
            )

    def _init_kv_manager(self) -> CommonKVManager:
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()
        kv_args.engine_rank = self.tp_rank
        kv_args.pp_rank = self.pp_rank
        kv_args.system_dp_rank = self.scheduler.dp_rank
        kv_args.prefill_start_layer = self.token_to_kv_pool.start_layer
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            self.token_to_kv_pool.get_contiguous_buf_infos()
        )

        if self.draft_token_to_kv_pool is not None:
            # We should also transfer draft model kv cache. The indices are
            # always shared with a target model.
            draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = (
                self.draft_token_to_kv_pool.get_contiguous_buf_infos()
            )
            kv_data_ptrs += draft_kv_data_ptrs
            kv_data_lens += draft_kv_data_lens
            kv_item_lens += draft_kv_item_lens

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens
        if not self.is_mla_backend:
            kv_args.kv_head_num = self.token_to_kv_pool.head_num
            kv_args.total_kv_head_num = (
                self.scheduler.model_config.get_total_num_kv_heads()
            )
        kv_args.page_size = self.token_to_kv_pool.page_size

        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
            self.metadata_buffers.get_buf_infos()
        )
        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.gpu_id

        if hasattr(self.token_to_kv_pool, "get_state_buf_infos"):
            state_data_ptrs, state_data_lens, state_item_lens = (
                self.token_to_kv_pool.get_state_buf_infos()
            )
            kv_args.state_data_ptrs = state_data_ptrs
            kv_args.state_data_lens = state_data_lens
            kv_args.state_item_lens = state_item_lens

            if isinstance(self.token_to_kv_pool, SWAKVPool):
                kv_args.state_type = "swa"
            elif isinstance(self.token_to_kv_pool, HybridLinearKVPool):
                kv_args.state_type = "mamba"
                # Get state dimension info for cross-TP slice transfer
                if hasattr(self.token_to_kv_pool, "get_state_dim_per_tensor"):
                    kv_args.state_dim_per_tensor = (
                        self.token_to_kv_pool.get_state_dim_per_tensor()
                    )
            elif isinstance(self.token_to_kv_pool, NSATokenToKVPool):
                kv_args.state_type = "nsa"
            else:
                kv_args.state_type = "none"
        else:
            kv_args.state_data_ptrs = []
            kv_args.state_data_lens = []
            kv_args.state_item_lens = []
            kv_args.state_type = "none"

        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(
            kv_args,
            DisaggregationMode.PREFILL,
            self.scheduler.server_args,
            self.is_mla_backend,
        )
        # Pass KV pool tensor refs to the manager for GPU gather (staging mode)
        if (
            envs.SGLANG_DISAGG_STAGING_BUFFER.get()
            and hasattr(kv_manager, "set_kv_buffer_tensors")
            and not self.is_mla_backend
        ):
            kv_pool = self.token_to_kv_pool
            if hasattr(kv_pool, "full_kv_pool"):
                kv_pool = kv_pool.full_kv_pool
            if hasattr(kv_pool, "k_buffer") and hasattr(kv_pool, "v_buffer"):
                kv_manager.set_kv_buffer_tensors(
                    kv_pool.k_buffer,
                    kv_pool.v_buffer,
                    kv_pool.page_size,
                )
        return kv_manager

    def add(self, req: Req, num_kv_heads: int) -> None:
        if self._check_if_req_exceed_kv_capacity(req):
            return

        backend = (
            TransferBackend.FAKE
            if req.bootstrap_host == FAKE_BOOTSTRAP_HOST
            else self.transfer_backend
        )
        kv_sender_class = get_kv_class(backend, KVClassType.SENDER)

        dest_tp_ranks = [self.tp_rank]

        req.disagg_kv_sender = kv_sender_class(
            mgr=self.kv_manager,
            bootstrap_addr=f"{req.bootstrap_host}:{self.bootstrap_port}",
            bootstrap_room=req.bootstrap_room,
            dest_tp_ranks=dest_tp_ranks,
            pp_rank=self.pp_rank,
        )
        self._process_req(req)
        self.queue.append(req)

    def extend(self, reqs: List[Req], num_kv_heads: int) -> None:
        for req in reqs:
            self.add(req, num_kv_heads)

    def _check_if_req_exceed_kv_capacity(self, req: Req) -> bool:
        if len(req.origin_input_ids) > self.max_total_num_tokens:
            message = f"Request {req.rid} exceeds the maximum number of tokens: {len(req.origin_input_ids)} > {self.max_total_num_tokens}"
            logger.error(message)
            req.time_stats.trace_ctx.abort(abort_info={"reason": message})
            prepare_abort(req, message, status_code=HTTPStatus.BAD_REQUEST)
            self.scheduler.stream_output([req], req.return_logprob)
            return True
        return False

    def _process_req(self, req: Req) -> None:
        """
        Set max_new_tokens = 1, so PrefillAdder memory estimation is accurate
        """
        req.sampling_params.max_new_tokens = 1

    def pop_bootstrapped(
        self,
        return_failed_reqs: bool = False,
        rids_to_check: Optional[List[str]] = None,
    ) -> List[Req]:
        """
        pop the reqs which has finished bootstrapping

        return_failed_reqs: For PP, on rank 0, also return the failed reqs to notify the next rank
        rids_to_check: For PP, on rank > 0, check the rids from the previous rank has consensus with the current rank.
        """

        bootstrapped_reqs = []
        failed_reqs = []
        indices_to_remove = set()

        if len(self.queue) == 0:
            if return_failed_reqs is False:
                return []
            else:
                return [], []

        if self.p_ready_dir:
            # Compute-ahead mode: move requests to P's waiting queue without
            # waiting for D destination metadata. Transfer is initialized in
            # process_disagg_prefill_inflight_queue after D sees P-ready.
            bootstrapped_reqs = [
                req
                for req in self.queue
                if req.bootstrap_host != FAKE_BOOTSTRAP_HOST
            ]
            self.queue = [
                req
                for req in self.queue
                if req.bootstrap_host == FAKE_BOOTSTRAP_HOST
            ]
            for req in bootstrapped_reqs:
                req.disagg_p_ready_deferred = True
                req.disagg_p_ready_transfer_started = False
                req.time_stats.set_bootstrap_done_time()
                req.time_stats.set_wait_queue_entry_time()
            if not self.queue:
                if return_failed_reqs:
                    return bootstrapped_reqs, []
                return bootstrapped_reqs

        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in self.queue],
            self.scheduler.attn_cp_cpu_group,
            self.scheduler.attn_tp_cpu_group,
        )

        for i, (req, poll) in enumerate(zip(self.queue, polls)):
            if rids_to_check is not None:
                # if req not in reqs_info_to_check, skip
                if req.rid not in rids_to_check:
                    continue

            if poll == KVPoll.Bootstrapping:
                continue
            elif poll == KVPoll.Failed:
                error_message = f"Prefill bootstrap failed for request rank={self.tp_rank} {req.rid=} {req.bootstrap_room=}"
                try:
                    req.disagg_kv_sender.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.error(error_message)
                req.time_stats.trace_ctx.abort(abort_info={"reason": error_message})
                prepare_abort(
                    req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
                )
                self.scheduler.stream_output([req], req.return_logprob)
                indices_to_remove.add(i)
                failed_reqs.append(req)
                if self.scheduler.enable_metrics:
                    self.scheduler.metrics_collector.increment_bootstrap_failed_reqs()
                if self.scheduler.enable_hicache_storage:
                    # to release prefetch events associated with the request
                    self.scheduler.tree_cache.release_aborted_request(req.rid)
                continue

            # KV.WaitingForInput - init here
            req.time_stats.set_bootstrap_done_time()
            num_kv_indices = len(req.origin_input_ids)
            if self.req_to_metadata_buffer_idx_allocator.available_size() == 0:
                break

            req.metadata_buffer_index = (
                self.req_to_metadata_buffer_idx_allocator.alloc()
            )
            assert req.metadata_buffer_index is not None

            num_pages = kv_to_page_num(num_kv_indices, self.token_to_kv_pool.page_size)
            req.disagg_kv_sender.init(num_pages, req.metadata_buffer_index)

            bootstrapped_reqs.append(req)
            indices_to_remove.add(i)
            req.time_stats.set_wait_queue_entry_time()

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        if return_failed_reqs is False:
            return bootstrapped_reqs
        else:
            return bootstrapped_reqs, failed_reqs


class SchedulerDisaggregationPrefillMixin:
    """
    Mixin for Scheduler to handle disaggregation prefill
    """

    def maybe_prefetch_staging_for_batch(self: Scheduler, batch: ScheduleBatch) -> None:
        """Pre-send STAGING_REQ so decode allocates staging during GPU forward."""
        kv_mgr = self.disagg_prefill_bootstrap_queue.kv_manager
        prefetch = getattr(kv_mgr, "_prefetch_staging_reqs", None)
        if prefetch is None:
            return
        for req in batch.reqs:
            room = getattr(req, "bootstrap_room", None)
            if room is not None and room in kv_mgr.transfer_infos:
                prefetch(room)

    def start_prefill_transfer_progress_worker(self: Scheduler) -> None:
        """Start the P-ready FIFO and independent P->D transfer consumers.

        The scheduler is only the producer: after Prefill it snapshots an
        immutable transfer payload and appends the request to
        ``_prefill_ready_queue``.  Consumers publish P-ready in FIFO order and
        independently drive one sender through poll/init/send/terminal.  A
        slow transport operation therefore occupies one consumer instead of
        serializing every ready request or the Prefill scheduler.

        Request/KV cleanup remains scheduler-owned after a consumer publishes
        a terminal cached poll.
        """

        if self.tp_size != 1:
            return
        default_enabled = "1" if envs.SGLANG_AGENTIC_KV_LIFECYCLE.get() else "0"
        enabled = os.getenv(
            "SGLANG_PREFILL_TRANSFER_ASYNC_PROGRESS", default_enabled
        ).lower()
        if enabled not in {"1", "true", "yes", "on"}:
            return
        self._prefill_transfer_poll_lock = threading.Lock()
        self._prefill_transfer_stop = threading.Event()
        self._prefill_transfer_interval = max(
            0.0005,
            float(
                os.getenv(
                    "SGLANG_PREFILL_TRANSFER_PROGRESS_INTERVAL_SECONDS", "0.005"
                )
            ),
        )
        # ``max_transfer_inflight`` is enforced independently by every D.
        # Size P's sender-progress pool the same way: a fixed process-wide
        # default of eight underfeeds multi-D deployments even while all D
        # workers have destination capacity.  An explicit consumer count is
        # still honored for experiments and non-agentic deployments.
        consumers_per_d = max(
            1,
            int(os.getenv("SGLANG_PREFILL_TRANSFER_CONSUMERS_PER_D", "8")),
        )
        decode_workers = max(
            1,
            int(os.getenv("SGLANG_AGENTIC_KV_D_WRITERS", "1")),
        )
        self._prefill_transfer_consumer_count = max(
            1,
            int(
                os.getenv(
                    "SGLANG_PREFILL_TRANSFER_CONSUMERS",
                    str(consumers_per_d * decode_workers),
                )
            ),
        )
        self._prefill_ready_condition = threading.Condition()
        self._prefill_ready_queue = deque()
        self._prefill_ready_queued_rids = set()
        self._prefill_ready_publish_condition = threading.Condition()
        self._prefill_ready_next_publish_sequence = 0
        self._prefill_transfer_async_enabled = True
        self._prefill_transfer_threads = []
        for index in range(self._prefill_transfer_consumer_count):
            thread = threading.Thread(
                target=self._prefill_transfer_consumer_worker,
                args=(index,),
                name=f"sglang-prefill-transfer-{os.getpid()}-{index}",
                daemon=True,
            )
            thread.start()
            self._prefill_transfer_threads.append(thread)
        logger.info(
            "Prefill producer/ready-buffer/transfer-consumer pipeline enabled "
            "consumers=%d interval_ms=%.3f",
            self._prefill_transfer_consumer_count,
            self._prefill_transfer_interval * 1000.0,
        )

    def _enqueue_deferred_prefill_transfer(self: Scheduler, req: Req) -> bool:
        """Append one producer result to the transfer-consumer FIFO.

        Deferred requests require an immutable payload and are published to
        the Router by the consumer.  Legacy/bootstrap requests have already
        called ``send_kv_chunk`` on the scheduler thread; consumers only poll
        them to terminal so server warmup and compatibility paths remain
        asynchronous.
        """

        deferred = getattr(req, "disagg_p_ready_deferred", False)
        if deferred and getattr(req, "_async_prefill_transfer_payload", None) is None:
            return False
        with self._prefill_ready_condition:
            if (
                req.rid in self._prefill_ready_queued_rids
                or getattr(req, "_async_prefill_transfer_consumer_active", False)
                or getattr(req, "disagg_p_ready_notified", False)
            ):
                return True
            if deferred:
                req._p_ready_sequence = getattr(
                    self, "_p_ready_publish_sequence", 0
                )
                self._p_ready_publish_sequence = req._p_ready_sequence + 1
            self._prefill_ready_queue.append(req)
            self._prefill_ready_queued_rids.add(req.rid)
            self._prefill_ready_condition.notify()
        return True

    def _prefill_transfer_progress_req_once(self: Scheduler, req: Req) -> int:
        """Advance one consumer-owned sender by one transport state."""

        poll = int(req.disagg_kv_sender.poll())
        if (
            poll == int(KVPoll.WaitingForInput)
            and not getattr(req, "disagg_p_ready_transfer_started", False)
        ):
            num_pages, page_indices, state_indices = (
                req._async_prefill_transfer_payload
            )
            req.disagg_kv_sender.init(num_pages, req.metadata_buffer_index)
            req.disagg_kv_sender.send(page_indices, state_indices)
            req.disagg_p_ready_transfer_started = True
            req.time_stats.set_prefill_transfer_queue_entry_time()
            return int(KVPoll.Transferring)
        return poll

    def _prefill_transfer_consumer_worker(
        self: Scheduler, consumer_index: int
    ) -> None:
        cycles = 0
        total_seconds = 0.0
        max_seconds = 0.0
        last_stats_at = time.monotonic()
        while not self._prefill_transfer_stop.is_set():
            with self._prefill_ready_condition:
                while (
                    not self._prefill_ready_queue
                    and not self._prefill_transfer_stop.is_set()
                ):
                    self._prefill_ready_condition.wait(timeout=0.1)
                if self._prefill_transfer_stop.is_set():
                    return
                req = self._prefill_ready_queue.popleft()
                self._prefill_ready_queued_rids.discard(req.rid)
                req._async_prefill_transfer_consumer_active = True

            poll = int(KVPoll.Failed)
            try:
                if getattr(req, "disagg_p_ready_deferred", False):
                    self._publish_deferred_prefill_ready(req)
                while not self._prefill_transfer_stop.is_set():
                    started_at = time.perf_counter()
                    poll = self._prefill_transfer_progress_req_once(req)
                    elapsed = time.perf_counter() - started_at
                    cycles += 1
                    total_seconds += elapsed
                    max_seconds = max(max_seconds, elapsed)
                    if poll in (int(KVPoll.Success), int(KVPoll.Failed)):
                        break
                    self._prefill_transfer_stop.wait(
                        max(0.0, self._prefill_transfer_interval - elapsed)
                    )
            except Exception:
                logger.exception(
                    "P->D transfer consumer=%d failed rid=%s",
                    consumer_index,
                    req.rid,
                )
                poll = int(KVPoll.Failed)
                # A failed marker write must not leave every later FIFO
                # sequence waiting forever.  Advance exactly this failed head;
                # out-of-order consumers remain blocked until their turn.
                with self._prefill_ready_publish_condition:
                    if (
                        getattr(req, "_p_ready_sequence", -1)
                        == self._prefill_ready_next_publish_sequence
                    ):
                        self._prefill_ready_next_publish_sequence += 1
                        self._prefill_ready_publish_condition.notify_all()

            with self._prefill_transfer_poll_lock:
                req._async_prefill_transfer_poll = poll
                req._async_prefill_transfer_consumer_active = False

            now = time.monotonic()
            if now - last_stats_at >= 30.0:
                with self._prefill_ready_condition:
                    buffered = len(self._prefill_ready_queue)
                logger.info(
                    "Prefill P->D consumer stats worker=%d cycles=%d "
                    "avg_us=%.1f max_ms=%.3f ready_buffer=%d inflight=%d",
                    consumer_index,
                    cycles,
                    total_seconds * 1e6 / max(1, cycles),
                    max_seconds * 1000.0,
                    buffered,
                    len(self.disagg_prefill_inflight_queue),
                )
                cycles = 0
                total_seconds = 0.0
                max_seconds = 0.0
                last_stats_at = now

    def _prepare_deferred_prefill_transfer(self: Scheduler, req: Req) -> bool:
        """Prepare immutable NIXL submission data on the scheduler thread.

        The P->D progress worker may start the actual transfer as soon as D
        publishes its destination pages, including while the P GPU is running
        the next Prefill forward.  Allocator mutation and CUDA index reads stay
        on the scheduler thread; the worker only consumes the resulting NumPy
        page list and submits it to NIXL.
        """

        if getattr(req, "_async_prefill_transfer_payload", None) is not None:
            return True
        if req.return_logprob:
            # The client reconstructs generated token ids from the returned
            # logprob records, whereas Decode seeds ``req.output_ids`` from a
            # separate metadata field.  They must describe the same sampled
            # first token before either field is snapshotted for DMA.
            if (
                not req.output_ids
                or not req.output_token_logprobs_idx
                or int(req.output_ids[0])
                != int(req.output_token_logprobs_idx[0])
            ):
                return False
        if self.req_to_metadata_buffer_idx_allocator.available_size() == 0:
            return False

        req.metadata_buffer_index = (
            self.req_to_metadata_buffer_idx_allocator.alloc()
        )
        assert req.metadata_buffer_index is not None

        page_size = self.token_to_kv_pool_allocator.page_size
        end_idx = min(len(req.fill_ids), len(req.origin_input_ids))
        kv_indices = (
            self.req_to_token_pool.req_to_token[
                req.req_pool_idx, req.start_send_idx:end_idx
            ]
            .cpu()
            .numpy()
        )
        req.start_send_idx = end_idx
        self.disagg_metadata_buffers.set_buf(req)

        state_indices = None
        kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
        if isinstance(kv_pool, HybridLinearKVPool):
            state_indices = [
                self.req_to_token_pool.req_index_to_mamba_index_mapping[
                    req.req_pool_idx
                ]
                .cpu()
                .numpy()
            ]
        elif isinstance(kv_pool, SWAKVPool):
            window_start = max(0, end_idx - self.sliding_window_size)
            window_start = (window_start // page_size) * page_size
            window_kv_indices_full = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, window_start:end_idx
            ]
            window_kv_indices_swa = (
                self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                    window_kv_indices_full
                )
            )
            state_indices = kv_to_page_indices(
                window_kv_indices_swa.cpu().numpy(), page_size
            )
        elif isinstance(kv_pool, NSATokenToKVPool):
            state_indices = kv_to_page_indices(
                self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, :end_idx
                ]
                .cpu()
                .numpy(),
                page_size,
            )

        page_indices = kv_to_page_indices(kv_indices, page_size)
        req._async_prefill_transfer_payload = (
            kv_to_page_num(end_idx, page_size),
            page_indices,
            state_indices,
        )
        return True

    def _publish_deferred_prefill_ready(self: Scheduler, req: Req) -> None:
        """Publish P-ready only after the complete transfer payload exists.

        In particular, ``disagg_metadata_buffers.set_buf(req)`` must run after
        ``add_logprob_return_values``.  The Decode worker uses ``output_ids``
        to seed its request while the client reconstructs the same first token
        from ``output_token_logprobs_idx``.  Snapshotting before the latter is
        populated can make those token ids disagree and poison the next-turn
        reverse-KV digest even though the KV DMA itself succeeds.
        """

        if getattr(req, "disagg_p_ready_notified", False):
            return
        ready_sequence = getattr(req, "_p_ready_sequence", None)
        if ready_sequence is None:
            ready_sequence = getattr(self, "_p_ready_publish_sequence", 0)
            self._p_ready_publish_sequence = ready_sequence + 1
            req._p_ready_sequence = ready_sequence
        publish_condition = getattr(
            self, "_prefill_ready_publish_condition", None
        )
        if publish_condition is None:
            ready_path = os.path.join(
                self.disagg_prefill_bootstrap_queue.p_ready_dir,
                f"{req.bootstrap_room}.ready",
            )
            tmp_path = f"{ready_path}.{os.getpid()}.{ready_sequence}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "rid": req.rid,
                        "num_kv_tokens": len(req.origin_input_ids),
                        "ready_sequence": ready_sequence,
                    },
                    handle,
                    separators=(",", ":"),
                )
            os.replace(tmp_path, ready_path)
            req.disagg_p_ready_notified = True
            return
        # Multiple consumers may finish transport control calls out of order,
        # but the Router must observe the producer's FIFO completion order.
        with publish_condition:
            while (
                ready_sequence != self._prefill_ready_next_publish_sequence
                and not self._prefill_transfer_stop.is_set()
            ):
                publish_condition.wait(timeout=0.1)
            if self._prefill_transfer_stop.is_set():
                return
            ready_path = os.path.join(
                self.disagg_prefill_bootstrap_queue.p_ready_dir,
                f"{req.bootstrap_room}.ready",
            )
            tmp_path = f"{ready_path}.{os.getpid()}.{ready_sequence}.tmp"
            ready_metadata = {
                "rid": req.rid,
                "num_kv_tokens": len(req.origin_input_ids),
                "ready_sequence": ready_sequence,
            }
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(ready_metadata, handle, separators=(",", ":"))
            os.replace(tmp_path, ready_path)
            req.disagg_p_ready_notified = True
            self._prefill_ready_next_publish_sequence += 1
            publish_condition.notify_all()

    def _prefill_transfer_cached_polls(self: Scheduler) -> list[int]:
        with self._prefill_transfer_poll_lock:
            polls = []
            for req in self.disagg_prefill_inflight_queue:
                poll = getattr(
                    req, "_async_prefill_transfer_poll", int(KVPoll.Transferring)
                )
                if getattr(req, "_async_prefill_transfer_poll", None) is not None:
                    req._async_prefill_transfer_poll_claimed = True
                polls.append(poll)
            return polls

    def _release_prefill_transfer_poll_claims(
        self: Scheduler, undone_reqs: list[Req]
    ) -> None:
        """Release scheduler claims without discarding a newer terminal poll.

        The consumer can publish ``Success`` after the scheduler snapshots an
        older ``Transferring`` state but before this cleanup runs.  Deleting
        ``_async_prefill_transfer_poll`` here loses that terminal transition;
        because P-ready was already published, the request is never enqueued
        again and permanently pins its P-side KV.  Terminal results are
        therefore level-triggered and remain visible until the request leaves
        ``disagg_prefill_inflight_queue``.
        """

        with self._prefill_transfer_poll_lock:
            for req in undone_reqs:
                req._async_prefill_transfer_poll_claimed = False

    def _should_throttle_p_ready_compute_ahead(self: Scheduler) -> bool:
        """Bound only *new* P-ready compute-ahead by completed P-side KV.

        P-ready deliberately allows Prefill to finish before a Decode worker has
        allocated destination pages.  The completed request remains locked in
        ``disagg_prefill_inflight_queue`` meanwhile.  Under sustained load that
        queue can otherwise consume the complete P KV pool and turn normal
        downstream backpressure into a hard Prefill OOM.

        Parent turns recovered through Direct or the slow path are deliberately
        exempt: accepting one of them releases the producer's D-side KV and is
        part of draining the pipeline.  The returned boolean therefore means
        "hold ordinary new work", not "stop the P scheduler".

        A partially chunked request is always allowed to finish so throttling
        cannot strand an allocation halfway through a prompt.
        """
        bootstrap_queue = getattr(self, "disagg_prefill_bootstrap_queue", None)
        if not getattr(bootstrap_queue, "p_ready_dir", ""):
            return False
        if self.chunked_req is not None:
            return False

        try:
            mode = os.environ.get(
                "SGLANG_PD_P_READY_BACKPRESSURE_MODE", "hysteresis"
            ).strip().lower()
            high = float(os.environ.get("SGLANG_PD_P_READY_HBM_HIGH_WATERMARK", "0.70"))
            low = float(os.environ.get("SGLANG_PD_P_READY_HBM_LOW_WATERMARK", "0.55"))
            max_inflight = int(os.environ.get("SGLANG_PD_P_READY_MAX_INFLIGHT", "48"))
            request_cap = int(
                os.environ.get(
                    "SGLANG_PD_P_READY_REQUEST_CAP", str(max_inflight)
                )
            )
            token_cap_fraction = float(
                os.environ.get("SGLANG_PD_P_READY_TOKEN_CAP_FRACTION", "0.25")
            )
            resume_inflight = int(
                os.environ.get("SGLANG_PD_P_READY_RESUME_INFLIGHT", "40")
            )
        except ValueError:
            logger.exception("Invalid P-ready compute-ahead backpressure setting")
            raise
        if mode == "disabled":
            # Let the native SGLang scheduler admit Prefill work until its
            # allocator reports that no batch fits.  Agentic Direct/slow/new
            # priority remains in the waiting queue; only the synthetic
            # request/token compute-ahead credits are disabled.
            self._p_ready_compute_credit_tokens = None
            self._p_ready_compute_ahead_throttled = False
            return False
        if mode not in {"continuous", "hysteresis"}:
            raise ValueError(
                "SGLANG_PD_P_READY_BACKPRESSURE_MODE must be disabled, "
                "continuous, or "
                f"hysteresis (got {mode!r})"
            )
        if not (0.0 < high < 1.0):
            raise ValueError(f"P-ready high watermark must be in (0, 1), got {high}")
        if mode == "hysteresis" and not (0.0 < low < high):
            raise ValueError(
                "P-ready HBM watermarks must satisfy 0 < low < high < 1 "
                f"(got low={low}, high={high})"
            )
        if max_inflight < 0 or resume_inflight < 0 or request_cap < 0:
            raise ValueError("P-ready inflight limits must be non-negative")
        if not (0.0 < token_cap_fraction < 1.0):
            raise ValueError(
                "SGLANG_PD_P_READY_TOKEN_CAP_FRACTION must be in (0, 1), "
                f"got {token_cap_fraction}"
            )
        if mode == "hysteresis" and max_inflight and resume_inflight > max_inflight:
            raise ValueError(
                "SGLANG_PD_P_READY_RESUME_INFLIGHT must not exceed "
                "SGLANG_PD_P_READY_MAX_INFLIGHT"
            )

        num_used, token_usage, available_size, evictable_size = self._get_token_info()
        inflight = len(self.disagg_prefill_inflight_queue)
        ready_tokens = sum(
            len(req.origin_input_ids) for req in self.disagg_prefill_inflight_queue
        )
        ready_token_cap = max(
            1, int(self.max_total_num_tokens * token_cap_fraction)
        )
        # In continuous mode the scheduler consumes this as a token credit
        # while constructing the next Prefill batch.  Checking only after a
        # batch is built permits a group of long/cached prompts to overshoot
        # the watermark by tens of thousands of protected KV tokens.
        self._p_ready_compute_credit_tokens = (
            max(0, int(self.max_total_num_tokens * high) - num_used)
            if mode == "continuous"
            else None
        )
        was_throttled = getattr(self, "_p_ready_compute_ahead_throttled", False)
        over_count = bool(request_cap and inflight >= request_cap)
        over_ready_tokens = ready_tokens >= ready_token_cap
        if mode == "continuous":
            throttled = token_usage >= high or over_count or over_ready_tokens
        elif was_throttled:
            below_count = not max_inflight or inflight <= resume_inflight
            # HBM hysteresis must not override the independent request/token
            # credits.  The previous branch resumed New work below the low
            # watermark even when completed P-ready KV was already 2x over its
            # token cap, causing an avoidable P-HBM saturation burst.
            throttled = (
                over_count
                or over_ready_tokens
                or not (token_usage <= low and below_count)
            )
        else:
            throttled = token_usage >= high or over_count or over_ready_tokens

        if throttled != was_throttled:
            logger.info(
                "P-ready compute-ahead %s token_usage=%.3f inflight=%d "
                "ready_tokens=%d/%d available_tokens=%d evictable_tokens=%d "
                "high=%.2f low=%.2f request_cap=%d mode=%s scope=new_only",
                "throttled" if throttled else "resumed",
                token_usage,
                inflight,
                ready_tokens,
                ready_token_cap,
                available_size,
                evictable_size,
                high,
                low,
                request_cap,
                mode,
            )
        self._p_ready_compute_ahead_throttled = throttled
        return throttled

    def get_next_disagg_prefill_batch_to_run(
        self: Scheduler,
    ) -> Optional[ScheduleBatch]:
        # HACK (byronhsu): reset the batch_is_full flag because we never enter update_running_batch which resets it
        # Otherwise, it hangs under high concurrency
        self.running_batch.batch_is_full = False

        self.process_prefill_chunk()

        throttle_new = self._should_throttle_p_ready_compute_ahead()
        if not throttle_new:
            batch = self.get_new_batch_prefill()
        else:
            # P-ready is a soft cap.  Hold only initial/new requests while
            # allowing Direct and slow-path parent turns to enter Prefill and
            # release pressure from D.  Preserve the strict fast > slow > new
            # order when the held work is restored.
            held_new = [
                req
                for req in self.waiting_queue
                if getattr(req, "_agentic_kv_queue_class", "new") == "new"
            ]
            self.waiting_queue = [
                req
                for req in self.waiting_queue
                if getattr(req, "_agentic_kv_queue_class", "new") != "new"
            ]
            batch = self.get_new_batch_prefill()
            self.waiting_queue.extend(held_new)
            prioritize = getattr(self, "_prioritize_agentic_prefill_ready", None)
            if prioritize is not None:
                prioritize()
        batch = self.maybe_prepare_mlp_sync_batch(batch)

        if batch:
            set_schedule_time_batch(batch)

        return batch

    @torch.no_grad()
    def event_loop_normal_disagg_prefill(self: Scheduler) -> None:
        """A normal scheduler loop for prefill worker in disaggregation mode."""
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()

        while True:
            # Receive requests
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            self._merge_disagg_prefill_ready(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )

            # Get the next batch to run
            batch = self.get_next_disagg_prefill_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                if self.enable_staging:
                    self.maybe_prefetch_staging_for_batch(batch)
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                self.self_check_during_idle()

            self.process_disagg_prefill_inflight_queue()

            # Update last_batch
            self.last_batch = batch

    @torch.no_grad()
    def event_loop_overlap_disagg_prefill(self: Scheduler) -> None:
        self.result_queue = deque()
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()

        while True:
            # Receive requests
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            self._merge_disagg_prefill_ready(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )

            # Get the next batch to run
            batch = self.get_next_disagg_prefill_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                if self.enable_staging:
                    self.maybe_prefetch_staging_for_batch(batch)
                batch_result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None

            # Process the last batch
            if self.last_batch:
                tmp_batch, tmp_result = self.result_queue.popleft()
                self.process_batch_result(tmp_batch, tmp_result)
            elif batch is None:
                # When the server is idle, do self-check and re-init some states
                self.self_check_during_idle()

            self.process_disagg_prefill_inflight_queue()

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            self.launch_batch_sample_if_needed(batch_result)

            # Update last_batch
            self.last_batch = batch

    def process_batch_result_disagg_prefill(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        """
        Transfer kv for prefill completed requests and add it into disagg_prefill_inflight_queue
        Adapted from process_batch_result_prefill
        """
        (
            logits_output,
            next_token_ids,
            extend_input_len_per_req,
            extend_logprob_start_len_per_req,
            copy_done,
        ) = (
            result.logits_output,
            result.next_token_ids,
            result.extend_input_len_per_req,
            result.extend_logprob_start_len_per_req,
            result.copy_done,
        )

        if copy_done is not None:
            copy_done.synchronize()

        logprob_pt = 0
        # Transfer kv for prefill completed requests and add it into disagg_prefill_inflight_queue
        next_token_ids = result.next_token_ids.tolist()
        if batch.return_logprob:
            if logits_output.next_token_logprobs is not None:
                logits_output.next_token_logprobs = (
                    logits_output.next_token_logprobs.tolist()
                )
            if logits_output.input_token_logprobs is not None:
                logits_output.input_token_logprobs = tuple(
                    logits_output.input_token_logprobs.tolist()
                )

        for i, (req, next_token_id) in enumerate(
            zip(batch.reqs, next_token_ids, strict=True)
        ):
            if req.is_chunked <= 0:
                req.time_stats.set_prefill_finished_time()

                # There is no output_ids for prefill
                req.output_ids.append(next_token_id)
                self.tree_cache.cache_unfinished_req(req)  # update the tree and lock
                self.disagg_prefill_inflight_queue.append(req)
                if self.spec_algorithm.is_eagle() and batch.spec_info is not None:
                    req.output_topk_p = batch.spec_info.topk_p[i]
                    req.output_topk_index = batch.spec_info.topk_index[i]
                    req.hidden_states_tensor = (
                        batch.spec_info.hidden_states[i].cpu().clone()
                    )
                else:
                    req.hidden_states_tensor = None
                if req.return_logprob:
                    assert extend_logprob_start_len_per_req is not None
                    assert extend_input_len_per_req is not None
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    num_input_logprobs = extend_input_len - extend_logprob_start_len
                    self.add_logprob_return_values(
                        i,
                        req,
                        logprob_pt,
                        next_token_ids,
                        num_input_logprobs,
                        logits_output,
                    )
                    logprob_pt += num_input_logprobs
                if getattr(req, "disagg_p_ready_deferred", False):
                    if not getattr(self, "_prefill_transfer_async_enabled", False) or (
                        self._prepare_deferred_prefill_transfer(req)
                    ):
                        if getattr(self, "_prefill_transfer_async_enabled", False):
                            self._enqueue_deferred_prefill_transfer(req)
                        else:
                            self._publish_deferred_prefill_ready(req)
                else:
                    self.send_kv_chunk(req, last_chunk=True)
                    req.disagg_p_ready_transfer_started = True
                    req.time_stats.set_prefill_transfer_queue_entry_time()
                    if getattr(self, "_prefill_transfer_async_enabled", False):
                        self._enqueue_deferred_prefill_transfer(req)

                if req.grammar is not None:
                    # FIXME: this try-except block is for handling unexpected xgrammar issue.
                    try:
                        req.grammar.accept_token(next_token_id)
                    except ValueError as e:
                        # Grammar accept_token can raise ValueError if the token is not in the grammar.
                        # This can happen if the grammar is not set correctly or the token is invalid.
                        error_message = f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                        release_kv_cache(req, self.tree_cache)
                        prepare_abort(
                            req,
                            error_message,
                            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                        )
                    req.grammar.finished = req.finished()
            else:
                # being chunked reqs' prefill is not finished
                req.is_chunked -= 1

                if req.return_logprob:
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    if extend_logprob_start_len < extend_input_len:
                        # Update input logprobs.
                        num_input_logprobs = extend_input_len - extend_logprob_start_len
                        self.add_input_logprob_return_values(
                            i,
                            req,
                            logits_output,
                            logprob_pt,
                            num_input_logprobs,
                            last_prefill_chunk=False,
                        )
                        logprob_pt += num_input_logprobs

                if self.enable_overlap:
                    self.send_kv_chunk(req, last_chunk=False, end_idx=req.tmp_end_idx)
                req.time_stats.set_last_chunked_prefill_finish_time()

        can_run_cuda_graph = getattr(result, "can_run_cuda_graph", False)
        self.report_prefill_stats(
            prefill_stats=batch.prefill_stats,
            can_run_cuda_graph=can_run_cuda_graph,
            dp_cooperation_info=batch.dp_cooperation_info,
        )

    def process_disagg_prefill_inflight_queue(
        self: Scheduler, rids_to_check: Optional[List[str]] = None
    ) -> List[Req]:
        """
        Poll the requests in the middle of transfer. If done, return the request.
        rids_to_check: For PP, on rank > 0, check the rids from the previous rank has consensus with the current rank.
        """
        if len(self.disagg_prefill_inflight_queue) == 0:
            return []

        done_reqs = []

        if getattr(self, "_prefill_transfer_async_enabled", False):
            # Metadata-buffer slots are bounded.  If none was available when
            # the Prefill result was finalized, retry preparation here on the
            # scheduler thread.  Never publish P-ready before preparation:
            # Decode must not advertise destination pages to a worker that has
            # no immutable, fully-populated metadata payload to send.
            for req in self.disagg_prefill_inflight_queue:
                if (
                    getattr(req, "disagg_p_ready_deferred", False)
                    and not getattr(req, "disagg_p_ready_notified", False)
                    and self._prepare_deferred_prefill_transfer(req)
                ):
                    self._enqueue_deferred_prefill_transfer(req)
            polls = self._prefill_transfer_cached_polls()
        else:
            polls = poll_and_all_reduce_attn_cp_tp_group(
                [req.disagg_kv_sender for req in self.disagg_prefill_inflight_queue],
                self.attn_cp_cpu_group,
                self.attn_tp_cpu_group,
            )

        undone_reqs: List[Req] = []
        # Check .poll() for the reqs in disagg_prefill_inflight_queue. If Success, respond to the client and remove it from the queue
        for req, poll in zip(self.disagg_prefill_inflight_queue, polls):

            if rids_to_check is not None:
                if req.rid not in rids_to_check:
                    undone_reqs.append(req)
                    continue

                # In PP mode, the previous rank may have reached a terminal
                # state (Success/Failed) while this rank's local poll is still
                # in a transient state due to clock skew or propagation delay.
                # Treat non-terminal states as undone instead of crashing.
                if poll not in (
                    KVPoll.Success,
                    KVPoll.Failed,
                ):
                    logger.warning(
                        f"PP rank {self.pp_rank}: unexpected poll state {poll} for rid {req.rid} "
                        f"from consensus; treating as undone"
                    )
                    undone_reqs.append(req)
                    continue

            if (
                poll == KVPoll.WaitingForInput
                and getattr(req, "disagg_p_ready_deferred", False)
                and not getattr(req, "disagg_p_ready_transfer_started", False)
            ):
                if getattr(self, "_prefill_transfer_async_enabled", False):
                    if self._prepare_deferred_prefill_transfer(req):
                        self._enqueue_deferred_prefill_transfer(req)
                    undone_reqs.append(req)
                    continue
                if self.req_to_metadata_buffer_idx_allocator.available_size() == 0:
                    undone_reqs.append(req)
                    continue
                req.metadata_buffer_index = (
                    self.req_to_metadata_buffer_idx_allocator.alloc()
                )
                num_pages = kv_to_page_num(
                    len(req.origin_input_ids),
                    self.token_to_kv_pool_allocator.page_size,
                )
                req.disagg_kv_sender.init(num_pages, req.metadata_buffer_index)
                req.disagg_p_ready_transfer_started = True
                self.send_kv_chunk(req, last_chunk=True)
                req.time_stats.set_prefill_transfer_queue_entry_time()
                undone_reqs.append(req)
            elif poll in [
                KVPoll.Bootstrapping,
                KVPoll.WaitingForInput,
                KVPoll.Transferring,
            ]:
                undone_reqs.append(req)
            elif poll == KVPoll.Success:  # transfer done
                agentic_metadata = (
                    AgenticRequestMetadata.from_req(req)
                    if envs.SGLANG_AGENTIC_KV_LIFECYCLE.get()
                    else None
                )
                # release_kv_cache pops kv_committed_len, but the following
                # request-private branch cleanup still needs the exact prefix
                # depth to rematch the just-inserted radix node.
                agentic_committed_len = (
                    len(req.origin_input_ids)
                    if agentic_metadata is not None
                    else None
                )
                if agentic_metadata is not None:
                    digest_len = (
                        agentic_committed_len
                        // self.token_to_kv_pool_allocator.page_size
                        * self.token_to_kv_pool_allocator.page_size
                    )
                    source_indices = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, :digest_len
                    ]
                    source_digest = debug_kv_digest(
                        self.token_to_kv_pool_allocator.get_kvcache(),
                        source_indices,
                    )
                    if source_digest is not None:
                        logger.info(
                            "AgenticKV p_source_digest snapshot=%s digest=%s",
                            agentic_metadata.current.snapshot_id,
                            source_digest,
                        )
                # A transferred request-generation has left P.  Do not turn
                # its completed branch into an opportunistic prefix cache:
                # only live request-generations may own P-side KV.
                release_kv_cache(req, self.tree_cache, is_insert=False)
                if agentic_metadata is not None:
                    release_agentic = getattr(
                        self.tree_cache, "release_agentic_request_cache", None
                    )
                    if release_agentic is not None:
                        released = release_agentic(
                            req, committed_len=agentic_committed_len
                        )
                        logger.info(
                            "AgenticKV p_to_d_release tokens=%d req=%s extra_key=%s",
                            released,
                            req.rid,
                            req.extra_key,
                        )
                req.finished_reason = FINISH_LENGTH(length=0)
                # FIXME: clean up req's data in transfer engine
                if hasattr(req.disagg_kv_sender, "clear"):
                    req.disagg_kv_sender.clear()
                if hasattr(req, "_async_prefill_transfer_payload"):
                    delattr(req, "_async_prefill_transfer_payload")
                done_reqs.append(req)
                req.time_stats.set_prefill_kv_transfer_finish_time()
            elif poll == KVPoll.Failed:
                error_message = f"Prefill transfer failed for request rank={self.tp_rank} {req.rid=} {req.bootstrap_room=}"
                try:
                    req.disagg_kv_sender.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.warning(error_message)
                req.time_stats.trace_ctx.abort(abort_info={"reason": error_message})
                release_kv_cache(req, self.tree_cache)  # unlock the tree
                prepare_abort(
                    req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
                )
                done_reqs.append(req)
                if self.enable_metrics:
                    self.metrics_collector.increment_transfer_failed_reqs()
            else:
                logger.warning(
                    f"Unexpected polling state {poll} for rid {req.rid} in inflight queue; "
                    f"treating as undone"
                )
                undone_reqs.append(req)

        for req in done_reqs:
            req.time_stats.set_completion_time()

        page_size = self.token_to_kv_pool_allocator.page_size
        kv_item_lens = (
            self.disagg_prefill_bootstrap_queue.kv_manager.kv_args.kv_item_lens
        )
        bytes_per_page_all_layers = sum(kv_item_lens)

        for req in done_reqs:
            if isinstance(req.finished_reason, FINISH_ABORT):
                continue
            metrics = req.time_stats.compute_and_observe_kv_transfer_metrics(
                num_tokens=len(req.origin_input_ids),
                page_size=page_size,
                bytes_per_page_all_layers=bytes_per_page_all_layers,
            )
            if metrics:
                # Update last-value for REST API
                if "latency_ms" in metrics:
                    self.kv_transfer_latency_ms = metrics["latency_ms"]
                if "speed_gb_s" in metrics:
                    self.kv_transfer_speed_gb_s = metrics["speed_gb_s"]

        # Stream requests which have finished transfer
        self.stream_output(
            done_reqs,
            any(req.return_logprob for req in done_reqs),
            None,
        )
        for req in done_reqs:
            req: Req

            release_req_to_metadata_buffer(
                req, self.req_to_metadata_buffer_idx_allocator
            )

        if getattr(self, "_prefill_transfer_async_enabled", False):
            self._release_prefill_transfer_poll_claims(undone_reqs)
        self.disagg_prefill_inflight_queue = undone_reqs

        return done_reqs

    def get_transferred_rids(self: Scheduler) -> List[str]:
        """
        Used by PP, get the transferred rids but **do not pop**
        """
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in self.disagg_prefill_inflight_queue],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )

        transferred_rids: List[str] = []

        for req, poll in zip(self.disagg_prefill_inflight_queue, polls):
            if poll == KVPoll.Success or poll == KVPoll.Failed:
                transferred_rids.append(req.rid)

        return transferred_rids

    def process_prefill_chunk(self: Scheduler) -> None:
        chunked_req_to_exclude = set()
        if self.chunked_req:
            chunked_req_to_exclude.add(self.chunked_req)
            self.tree_cache.cache_unfinished_req(self.chunked_req, chunked=True)
            if self.enable_overlap:
                # Delay KV transfer to process_batch_result_disagg_prefill when overlap is enabled to ensure results are resolved
                self.chunked_req.tmp_end_idx = min(
                    len(self.chunked_req.fill_ids),
                    len(self.chunked_req.origin_input_ids),
                )
            else:
                self.send_kv_chunk(self.chunked_req)
            self.running_batch.batch_is_full = False

        if self.last_batch and self.last_batch.forward_mode.is_extend():
            if self.last_batch.chunked_req:
                # In the context pipeline parallelism, after the last chunk, the current microbatch still track outdated chunked_req.
                # We need to discard it.
                chunked_req_to_exclude.add(self.last_batch.chunked_req)

            last_bs = self.last_batch.batch_size()
            self.last_batch.filter_batch(
                chunked_req_to_exclude=list(chunked_req_to_exclude)
            )
            if self.last_batch.batch_size() < last_bs:
                self.running_batch.batch_is_full = False

    def send_kv_chunk(
        self: Scheduler,
        req: Req,
        last_chunk: bool = False,
        end_idx: Optional[int] = None,
    ) -> None:
        """
        Send a prefilled chunk to the decode server
        """
        if (
            self.disagg_prefill_bootstrap_queue.p_ready_dir
            and getattr(req, "disagg_p_ready_deferred", False)
            and not getattr(req, "disagg_p_ready_transfer_started", False)
        ):
            # Preserve start_send_idx. Once D allocates, send the complete
            # prompt KV from index zero instead of leaking partial chunks.
            return
        page_size = self.token_to_kv_pool_allocator.page_size
        start_idx = req.start_send_idx
        end_idx = (
            end_idx
            if end_idx is not None
            else min(len(req.fill_ids), len(req.origin_input_ids))
        )

        if not last_chunk:
            # if not the last chunk and the last page is partial, delay the last partial page to the next send
            end_idx = end_idx - end_idx % page_size

        kv_indices = (
            self.req_to_token_pool.req_to_token[req.req_pool_idx, start_idx:end_idx]
            .cpu()
            .numpy()
        )
        req.start_send_idx = end_idx
        state_indices = None
        if last_chunk:
            self.disagg_metadata_buffers.set_buf(req)

            # Prepare extra pool indices for hybrid models
            if isinstance(
                self.token_to_kv_pool_allocator.get_kvcache(), HybridLinearKVPool
            ):
                # Mamba hybrid model: send single mamba state index
                state_indices = [
                    self.req_to_token_pool.req_index_to_mamba_index_mapping[
                        req.req_pool_idx
                    ]
                    .cpu()
                    .numpy()
                ]
            elif isinstance(self.token_to_kv_pool_allocator.get_kvcache(), SWAKVPool):
                # SWA hybrid model: send last window KV indices
                seq_len = len(req.fill_ids)
                window_size = self.sliding_window_size
                window_start = max(0, seq_len - window_size)
                window_start = (window_start // page_size) * page_size

                window_kv_indices_full = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, window_start:seq_len
                ]

                # Translate to SWA pool indices
                window_kv_indices_swa = (
                    self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                        window_kv_indices_full
                    )
                )
                state_indices = window_kv_indices_swa.cpu().numpy()
                state_indices = kv_to_page_indices(state_indices, page_size)
            elif isinstance(
                self.token_to_kv_pool_allocator.get_kvcache(), NSATokenToKVPool
            ):
                seq_len = len(req.fill_ids)
                kv_indices_full = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, :seq_len
                ]
                state_indices = kv_indices_full.cpu().numpy()
                state_indices = kv_to_page_indices(state_indices, page_size)

        page_indices = kv_to_page_indices(kv_indices, page_size)
        if len(page_indices) == 0:
            logger.info(
                f"Skip sending kv chunk for request {req.rid=} {req.bootstrap_room=} because page_indices is empty"
            )
            return
        req.disagg_kv_sender.send(page_indices, state_indices)
