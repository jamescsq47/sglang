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

import hashlib
import json
import logging
import os
import threading
import time
from array import array
from collections import deque
from http import HTTPStatus
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.agentic_direct_transfer import debug_kv_digest
from sglang.srt.disaggregation.agentic_hybrid_transfer import (
    freeze_p2d_mamba_checkpoint_after_cache,
    p2d_mamba_source_indices,
    p2d_mamba_checkpoint_tokens,
)
from sglang.srt.disaggregation.agentic_kv_lifecycle import AgenticRequestMetadata
from sglang.srt.disaggregation.agentic_tp import request_generation_key
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    KVClassType,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    get_kv_class,
    is_aborted,
    is_mla_backend,
    poll_and_all_reduce_attn_cp_tp_group,
    prepare_abort,
    setup_state_kv_args,
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    Req,
    ScheduleBatch,
)
from sglang.srt.mem_cache.common import (
    kv_to_page_indices,
    kv_to_page_num,
    maybe_cache_unfinished_req,
    release_kv_cache,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.observability.req_time_stats import set_schedule_time_batch
from sglang.srt.utils.nvtx_utils import scheduler_nvtx_method

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler
    from sglang.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)


class _P2DHostOrNativePoller:
    """Expose one rank-local P->D status to the existing TP collective."""

    def __init__(self, host_manager, req: Req):
        self.host_manager = host_manager
        self.req = req

    def poll(self):
        host_poll = self.host_manager.poll(self.req)
        if host_poll is not None:
            return host_poll
        return self.req.disagg_kv_sender.poll()


def should_force_retry(req: Req) -> bool:
    """Test hook to force a request into optimistic prefill retry."""
    retry_prob = envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB.get()
    if retry_prob <= 0 or req.time_stats.prefill_retry_count > 0 or req.is_retracted:
        return False

    digest = hashlib.sha256(str(req.rid).encode()).digest()
    return int.from_bytes(digest[:8], "big") < retry_prob * 2**64


def maybe_release_metadata_buffer(
    req: Req, allocator: ReqToMetadataIdxAllocator
) -> None:
    """
    Release the metadata buffer index allocated for a request in prefill disaggregation mode.

    This function safely releases the metadata buffer index if it was allocated.

    Args:
        req: The request object that may have a metadata_buffer_index allocated
        allocator: The ReqToMetadataIdxAllocator instance to free the index
    """
    if req.metadata_buffer_index >= 0:
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
        # Compute-ahead/late-binding is enabled by the run-scoped ready
        # directory.  Keep the attribute present for every backend and TP
        # rank; an empty value preserves upstream bootstrap behavior.
        self.p_ready_dir = os.getenv("SGLANG_PD_P_READY_DIR", "")
        self.gloo_group = gloo_group
        self.scheduler = scheduler
        self.max_total_num_tokens = (
            self.scheduler.tp_worker.model_runner.max_token_pool_size
        )
        self.transfer_backend = transfer_backend
        if envs.SGLANG_DISAGG_STAGING_BUFFER.get() and self.is_mla_backend:
            raise RuntimeError(
                "SGLANG_DISAGG_STAGING_BUFFER is designed for non-MLA models "
                "(e.g. GQA, MHA). MLA models should not set this flag."
            )
        self.kv_manager = self._init_kv_manager()

    def _init_kv_manager(self) -> CommonKVManager:
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()
        kv_args.engine_rank = self.tp_rank
        kv_args.pp_rank = self.pp_rank
        kv_args.system_dp_rank = self.scheduler.ps.dp_rank
        kv_args.prefill_start_layer = self.token_to_kv_pool.start_layer
        kv_args.prefill_end_layer = getattr(self.token_to_kv_pool, "end_layer", None)
        kv_args.mla_compression_ratios = None
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
        kv_args.gpu_id = self.scheduler.ps.gpu_id

        req_to_token_pool = getattr(self.scheduler, "req_to_token_pool", None)
        setup_state_kv_args(
            kv_args,
            self.token_to_kv_pool,
            self.draft_token_to_kv_pool,
            self.scheduler.model_config.num_hidden_layers,
            req_to_token_pool=req_to_token_pool,
        )

        if isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool):
            # V4's KVCache is organized by compression-ratio
            # buckets rather than by layer.
            kv_args.mla_compression_ratios = list(
                self.token_to_kv_pool.compression_ratios
            )

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

    def create_sender(self, req: Req, num_kv_heads: int) -> bool:
        """Create a KV sender for the request without enqueuing it.
        Returns False if the request exceeds KV capacity."""
        if self._check_if_req_exceed_kv_capacity(req):
            return False

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
        req.pending_bootstrap = True
        return True

    def ensure_metadata_buffer(self, req: Req) -> bool:
        if req.metadata_buffer_index >= 0:
            return True

        if self.req_to_metadata_buffer_idx_allocator.available_size() == 0:
            return False
        req.metadata_buffer_index = self.req_to_metadata_buffer_idx_allocator.alloc()
        assert req.metadata_buffer_index is not None
        return True

    def finalize_bootstrap(self, req: Req) -> bool:
        """Initialize the sender after bootstrap completes.
        Returns False if no metadata buffer is available (non-terminal)."""
        assert req.pending_bootstrap, "finalize_bootstrap is not idempotent"
        if not self.ensure_metadata_buffer(req):
            return False

        req.time_stats.set_bootstrap_done_time()
        num_kv_indices = len(req.origin_input_ids)

        decode_prefix_len = req.disagg_kv_sender.pop_decode_prefix_len()
        req.start_send_idx = decode_prefix_len
        num_kv_indices_to_send = num_kv_indices - decode_prefix_len
        num_pages = kv_to_page_num(
            num_kv_indices_to_send, self.token_to_kv_pool.page_size
        )
        req.disagg_kv_sender.init(num_pages, req.metadata_buffer_index)
        req.pending_bootstrap = False
        return True

    def add(self, req: Req, num_kv_heads: int) -> None:
        if not self.create_sender(req, num_kv_heads):
            return
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
            self.scheduler.output_streamer.stream_output([req], req.return_logprob)
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
                req for req in self.queue if req.bootstrap_host != FAKE_BOOTSTRAP_HOST
            ]
            self.queue = [
                req for req in self.queue if req.bootstrap_host == FAKE_BOOTSTRAP_HOST
            ]
            for req in bootstrapped_reqs:
                # Late binding deliberately skips the native D bootstrap.
                # Mark that phase complete so the normal Prefill result path
                # prepares/publishes the immutable P-ready payload instead of
                # treating every forward as an optimistic-bootstrap miss.
                req.pending_bootstrap = False
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
            if (
                rids_to_check is not None
                and req.rid not in rids_to_check
                and poll != KVPoll.Failed
            ):
                # In PP mode, successful bootstrap still requires cross-rank
                # consensus. Local failures are terminal and must be drained
                # even if an earlier PP rank has already removed the request.
                continue

            if poll == KVPoll.Failed:
                self.scheduler.handle_bootstrap_failure(req)
                indices_to_remove.add(i)
                failed_reqs.append(req)
            elif poll == KVPoll.Bootstrapping:
                if (
                    req.time_stats.prefill_retry_count
                    < self.scheduler.server_args.optimistic_prefill_retries
                    and not req.is_retracted  # engine paused
                ):
                    if not self.ensure_metadata_buffer(req):
                        continue  # no more metadata buffer
                    bootstrapped_reqs.append(req)
                    indices_to_remove.add(i)
                    req.time_stats.set_wait_queue_entry_time()
            elif poll == KVPoll.WaitingForInput:
                if not self.finalize_bootstrap(req):
                    continue
                bootstrapped_reqs.append(req)
                indices_to_remove.add(i)
                req.time_stats.set_wait_queue_entry_time()
            else:
                raise RuntimeError(
                    f"Unexpected poll state {poll} for req {req.rid} in pop_bootstrapped"
                )

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        if return_failed_reqs is False:
            return bootstrapped_reqs
        else:
            return bootstrapped_reqs, failed_reqs

    def release_memory_occupation(self):
        self.queue.clear()
        if hasattr(self.kv_manager, "deregister_buffer_to_engine"):
            self.kv_manager.deregister_buffer_to_engine()

    def resume_memory_occupation(self):
        if hasattr(self.kv_manager, "register_buffer_to_engine"):
            self.kv_manager.register_buffer_to_engine()


class SchedulerDisaggregationPrefillMixin:
    def start_prefill_transfer_progress_worker(self: Scheduler) -> None:
        """Start the P-ready FIFO and independent P->D progress workers.

        The scheduler is only the producer: after Prefill it snapshots an
        immutable transfer payload and appends the request to
        ``_prefill_ready_queue``.  Workers publish P-ready in FIFO order, make
        one non-blocking progress step, and put non-terminal requests back at
        the tail.  No request may own a finite worker until completion:
        otherwise enough receivers waiting for metadata can exhaust the pool
        and permanently strand every later P result.

        Request/KV cleanup remains scheduler-owned after a consumer publishes
        a terminal cached poll.
        """

        default_enabled = "1" if envs.SGLANG_AGENTIC_KV_LIFECYCLE.get() else "0"
        enabled = os.getenv(
            "SGLANG_PREFILL_TRANSFER_ASYNC_PROGRESS", default_enabled
        ).lower()
        if enabled not in {"1", "true", "yes", "on"}:
            return
        self._prefill_transfer_poll_lock = threading.Lock()
        self._prefill_transfer_terminal_queue = deque()
        self._prefill_transfer_prepare_queue = deque()
        self._prefill_transfer_prepare_keys = set()
        self._prefill_transfer_stop = threading.Event()
        self._prefill_transfer_interval = max(
            0.0005,
            float(
                os.getenv("SGLANG_PREFILL_TRANSFER_PROGRESS_INTERVAL_SECONDS", "0.005")
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
        self._prefill_transfer_tp_background_enabled = self.tp_size > 1
        if self._prefill_transfer_tp_background_enabled:
            # Each process owns one physical TP shard.  One FIFO worker per
            # rank is sufficient: rank0 publishes the logical command through
            # the tmpfs mailbox and every rank advances the same generation in
            # the background.  No CUDA allocator mutation happens here.
            self._prefill_transfer_consumer_count = 1
        self._prefill_ready_condition = threading.Condition()
        self._prefill_ready_queue = deque()
        self._prefill_ready_queued_keys = set()
        self._prefill_transfer_active_reqs = {}
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
        self._prefill_transfer_cleanup_pending = set()
        self._prefill_transfer_cleanup_lock = threading.Lock()
        self._prefill_transfer_cleanup_thread = None
        if self._prefill_transfer_tp_background_enabled and self.tp_rank == 0:
            self._prefill_transfer_cleanup_thread = threading.Thread(
                target=self._prefill_transfer_cleanup_worker,
                name=f"sglang-prefill-transfer-cleanup-{os.getpid()}",
                daemon=True,
            )
            self._prefill_transfer_cleanup_thread.start()
        logger.info(
            "Prefill producer/ready-buffer/transfer pipeline enabled "
            "background_consumers=%d interval_ms=%.3f tp_owner=%s",
            self._prefill_transfer_consumer_count,
            self._prefill_transfer_interval * 1000.0,
            "tp-rank0" if self.tp_size > 1 else "worker",
        )

    @staticmethod
    def _prefill_transfer_key(req: Req) -> str:
        room = getattr(req, "bootstrap_room", None)
        # Legacy warmup/test requests predate bootstrap rooms.  Real serving
        # generations always carry one and therefore use the collision-free
        # request-generation identity below.
        return str(req.rid) if room is None else request_generation_key(req.rid, room)

    def _clear_tp_prefill_transfer_mailboxes(self: Scheduler, req: Req) -> None:
        """Acknowledge scheduler release; TP0 later retires group state."""

        if self.tp_size <= 1:
            return
        key = self._prefill_transfer_key(req)
        if not getattr(self, "_prefill_transfer_tp_background_enabled", False):
            for mailbox in (
                self.agentic_tp_p2d_sender_mailbox,
                self.agentic_tp_p2d_receiver_mailbox,
            ):
                mailbox.clear_local(key)
                if self.tp_rank == 0:
                    mailbox.clear_group(key)
            return
        cleanup_mailbox = self.agentic_tp_p2d_cleanup_mailbox
        cleanup_mailbox.publish_local(key, int(KVPoll.Success))
        if self.tp_rank == 0:
            with self._prefill_transfer_cleanup_lock:
                self._prefill_transfer_cleanup_pending.add(key)

    def _prefill_transfer_cleanup_worker(self: Scheduler) -> None:
        """Clear TP control files only after every scheduler released pages."""

        while not self._prefill_transfer_stop.wait(self._prefill_transfer_interval):
            self._prefill_transfer_cleanup_once()

    def _prefill_transfer_cleanup_once(self: Scheduler) -> int:
        """Run one non-blocking TP0 cleanup scan; return cleared groups."""

        with self._prefill_transfer_cleanup_lock:
            pending = tuple(self._prefill_transfer_cleanup_pending)
        cleared = 0
        for key in pending:
            if self.agentic_tp_p2d_cleanup_mailbox.group_status(key) != int(
                KVPoll.Success
            ):
                continue
            # Sender state/command and destination receipt remain visible
            # until every producer rank has cached the terminal result and
            # completed allocator cleanup on its scheduler thread.
            self.agentic_tp_p2d_sender_mailbox.clear_group(key)
            self.agentic_tp_p2d_receiver_mailbox.clear_group(key)
            self.agentic_tp_p2d_cleanup_mailbox.clear_group(key)
            with self._prefill_transfer_cleanup_lock:
                self._prefill_transfer_cleanup_pending.discard(key)
            cleared += 1
        return cleared

    def _cleanup_failed_prefill_transfer(
        self: Scheduler,
        req: Req,
        p2d_host,
        agentic_metadata: Optional[AgenticRequestMetadata],
    ) -> bool:
        """Release one failed P->D generation back to its owning pools."""

        if p2d_host is not None:
            # Invalidate even an unclaimed watcher candidate before returning
            # its source pages to the allocator.
            cancel_watch = getattr(
                p2d_host, "cancel_watch", p2d_host.mark_scheduler_consumed
            )
            if cancel_watch(req) is False:
                # A local or peer TP rank already committed this generation
                # to Host staging.  Keep the source pages alive and let the
                # group finish that path before cleanup.
                return False
        if agentic_metadata is None:
            release_kv_cache(req, self.tree_cache)
        else:
            committed_len = len(req.origin_input_ids)
            release_kv_cache(req, self.tree_cache, is_insert=False)
            release_agentic = getattr(
                self.tree_cache, "release_agentic_request_cache", None
            )
            if release_agentic is not None:
                release_agentic(req, committed_len=committed_len)
        if hasattr(req.disagg_kv_sender, "clear"):
            req.disagg_kv_sender.clear()
        if hasattr(req, "_async_prefill_transfer_payload"):
            delattr(req, "_async_prefill_transfer_payload")
        for name in (
            "_agentic_workset_backed",
            "_agentic_p_workset_lease",
            "_agentic_p_workset_broker",
            "_agentic_workset_suffix_allocated_tokens",
            "_agentic_workset_suffix_indices",
        ):
            if hasattr(req, name):
                delattr(req, name)
        self._clear_tp_prefill_transfer_mailboxes(req)
        return True

    def _prefill_queued_keys(self: Scheduler) -> set[str]:
        keys = getattr(self, "_prefill_ready_queued_keys", None)
        if keys is None:
            keys = getattr(self, "_prefill_ready_queued_rids", None)
        if keys is None:
            keys = set()
            self._prefill_ready_queued_keys = keys
        return keys

    def _report_tp_prefill_producer_ready(self: Scheduler, req: Req) -> None:
        """Report that this rank's immutable P->D payload is available."""

        if (
            getattr(self, "tp_size", 1) > 1
            and getattr(req, "_async_prefill_transfer_payload", None) is not None
            and not getattr(
                req, "_async_prefill_transfer_producer_reported_once", False
            )
        ):
            self.agentic_tp_p2d_sender_mailbox.publish_local(
                self._prefill_transfer_key(req), int(KVPoll.Bootstrapping)
            )
            # Producer preparation is immutable for one request-generation.
            # Re-reporting from TP followers would otherwise overwrite a
            # worker's later Waiting/terminal state or recreate its sender
            # file after group cleanup.
            req._async_prefill_transfer_producer_reported_once = True

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
        key = self._prefill_transfer_key(req)
        if deferred:
            self._report_tp_prefill_producer_ready(req)
        with self._prefill_ready_condition:
            queued_keys = self._prefill_queued_keys()
            tp_background = bool(
                getattr(
                    self,
                    "_prefill_transfer_tp_background_enabled",
                    getattr(self, "tp_size", 1) > 1,
                )
            )
            if (
                (
                    tp_background
                    and getattr(req, "_async_prefill_transfer_enqueued_once", False)
                )
                or key in queued_keys
                or getattr(req, "_async_prefill_transfer_consumer_active", False)
                or (
                    getattr(self, "tp_size", 1) == 1
                    and getattr(req, "disagg_p_ready_notified", False)
                )
            ):
                return True
            if deferred:
                # A generation receives exactly one producer sequence.  TP0
                # may publish its logical P-ready marker before the bounded
                # data-plane worker is activated; activating it must not
                # overwrite that sequence and create a permanent FIFO gap.
                if getattr(req, "_p_ready_sequence", None) is None:
                    req._p_ready_sequence = getattr(
                        self, "_p_ready_publish_sequence", 0
                    )
                    self._p_ready_publish_sequence = req._p_ready_sequence + 1
            self._prefill_ready_queue.append(req)
            if tp_background:
                # Every physical rank owns exactly one background state
                # machine for this request-generation.  Only TP0 publishes
                # the logical P-ready marker, so the rank-local notification
                # flag cannot be used as the enqueue guard on followers.
                req._async_prefill_transfer_enqueued_once = True
            queued_keys.add(key)
            self._prefill_ready_condition.notify()
        return True

    def _prefill_transfer_progress_req_once(self: Scheduler, req: Req) -> int:
        """Advance one consumer-owned sender by one transport state."""

        p2d_host = getattr(self, "agentic_p2d_host_staging_manager", None)
        if getattr(req, "_agentic_p2d_host_terminal", False):
            return int(KVPoll.Success)
        if p2d_host is not None:
            host_poll = p2d_host.poll(req)
            if host_poll is not None:
                # A local worker, or a peer rank's request-level Host claim,
                # exclusively owns this logical snapshot.  Do not let the
                # normal NIXL consumer race it.
                return int(host_poll)

        poll = int(req.disagg_kv_sender.poll())
        if poll == int(KVPoll.WaitingForInput) and not getattr(
            req, "disagg_p_ready_transfer_started", False
        ):
            if getattr(self, "tp_size", 1) > 1:
                self.agentic_tp_p2d_sender_mailbox.publish_local(
                    self._prefill_transfer_key(req), poll
                )
                return poll
            num_pages, page_indices, state_indices = req._async_prefill_transfer_payload
            req.disagg_kv_sender.init(num_pages, req.metadata_buffer_index)
            req.disagg_kv_sender.send(page_indices, state_indices)
            req.disagg_p_ready_transfer_started = True
            req.time_stats.set_prefill_transfer_queue_entry_time()
            return int(KVPoll.Transferring)
        return poll

    def _prefill_transfer_progress_tp1_req_once(self: Scheduler, req: Req) -> int:
        """Progress one TP=1 P->D request without coupling it to D->P polling.

        A submitted NIXL sender owns request-local transfer handles. Checking
        those handles does not drain the manager-wide notification queue used
        by reverse Direct receives, so it must not wait on the reverse-control
        lock. Only bootstrap discovery and the one-time send submission touch
        shared NIXL manager control state and remain serialized.

        The P->D Host fallback is also independent of NIXL. Poll it before
        taking the control lock so a Host-owned request never waits behind a
        reverse Direct control call.
        """

        p2d_host = getattr(self, "agentic_p2d_host_staging_manager", None)
        if p2d_host is not None:
            host_poll = p2d_host.poll(req)
            if host_poll is not None:
                return int(host_poll)

        sender = getattr(req, "disagg_kv_sender", None)
        native_rw = bool(
            getattr(getattr(sender, "kv_mgr", None), "thread_sync_rw_enabled", False)
        )
        if getattr(req, "disagg_p_ready_transfer_started", False) and native_rw:
            return int(req.disagg_kv_sender.poll())

        nixl_lock = getattr(self, "agentic_nixl_control_lock", None)
        if nixl_lock is None:
            return self._prefill_transfer_progress_req_once(req)
        with nixl_lock:
            # A reverse Direct request can arrive while this P->D sender is
            # waiting for the shared legacy NIXL control boundary.  Yield
            # before touching the sender so the scheduler-independent Direct
            # worker can claim and start that receive first.  Native RW uses
            # request-local handles and never takes this compatibility path.
            direct_requested = getattr(self, "agentic_direct_poll_requested", None)
            if direct_requested is not None and direct_requested.is_set():
                return int(KVPoll.Transferring)
            # Recheck Host ownership after waiting for the shared bootstrap /
            # submission boundary. A Host claim may have won meanwhile.
            return self._prefill_transfer_progress_req_once(req)

    def _prefill_transfer_progress_tp_req_once(self: Scheduler, req: Req) -> int:
        """Advance one TP P->D generation without a scheduler iteration.

        Every rank prepares and reports its immutable physical shard.  TP0
        publishes the P-ready marker only after all shards exist, then writes
        one submit receipt after all senders have observed their matching D
        receiver.  Direct completion is destination-authored; Host completion
        is reduced across the producer ranks through the same mailbox.
        """

        key = self._prefill_transfer_key(req)
        sender_mailbox = self.agentic_tp_p2d_sender_mailbox
        receiver_mailbox = self.agentic_tp_p2d_receiver_mailbox
        p2d_host = getattr(self, "agentic_p2d_host_staging_manager", None)

        # The scheduler consumes only a native TP-broadcast group terminal.
        # Once that boundary has been crossed on this rank, no rank-local
        # sender/receipt state may resurrect the request after mailbox cleanup.
        group_terminal = getattr(req, "_agentic_p2d_group_terminal", None)
        if group_terminal in (int(KVPoll.Success), int(KVPoll.Failed)):
            return int(group_terminal)

        if getattr(req, "_agentic_p2d_host_terminal", False):
            local_poll = int(KVPoll.Success)
        elif p2d_host is not None:
            host_poll = p2d_host.poll(req)
            local_poll = None if host_poll is None else int(host_poll)
        else:
            local_poll = None

        if local_poll is not None:
            sender_mailbox.publish_local(key, local_poll)
            if self.tp_rank == 0:
                group_poll, _ = sender_mailbox.transfer_group_status(key)
                if group_poll in (int(KVPoll.Success), int(KVPoll.Failed)):
                    sender_mailbox.publish_receipt(key, group_poll)
            receipt = sender_mailbox.receipt(key)
            return (
                int(KVPoll.Transferring)
                if receipt not in (int(KVPoll.Success), int(KVPoll.Failed))
                else int(receipt)
            )

        # A Direct transfer is complete only after D commits every shard and
        # TP0 on D publishes the destination-authored receipt.
        direct_receipt = receiver_mailbox.receipt(key)
        if direct_receipt in (int(KVPoll.Success), int(KVPoll.Failed)):
            return int(direct_receipt)

        local_terminal = getattr(req, "_agentic_p2d_sender_terminal", None)
        poll = (
            int(local_terminal)
            if local_terminal in (int(KVPoll.Success), int(KVPoll.Failed))
            else int(req.disagg_kv_sender.poll())
        )
        sender_mailbox.publish_local(key, poll)

        if self.tp_rank == 0:
            raw_group_poll = sender_mailbox.group_status(key)
            group_poll, _ = sender_mailbox.transfer_group_status(key)
            submit_authorized = sender_mailbox.receipt(key) == int(
                KVPoll.WaitingForInput
            )
            # Before TP0 authorizes submission, no rank may have started DMA,
            # so a preparation failure is immediately terminal.  Once the
            # command exists, a failed shard cannot make the group terminal
            # until every peer sender reaches a physical terminal state.
            if (
                raw_group_poll == int(KVPoll.Failed) and not submit_authorized
            ) or group_poll == int(KVPoll.Failed):
                sender_mailbox.publish_receipt(key, int(KVPoll.Failed))
            if (
                raw_group_poll is not None
                and raw_group_poll >= int(KVPoll.Bootstrapping)
                and not getattr(req, "disagg_p_ready_notified", False)
            ):
                self._publish_deferred_prefill_ready(req)
            if (
                raw_group_poll == int(KVPoll.WaitingForInput)
                and sender_mailbox.receipt(key) is None
            ):
                # This receipt is the rank0 command authorizing every P rank
                # to submit its already-prepared shard.
                sender_mailbox.publish_receipt(key, int(KVPoll.WaitingForInput))

        submit_command = sender_mailbox.receipt(key)
        if submit_command == int(KVPoll.Failed):
            return int(KVPoll.Failed)
        if submit_command == int(KVPoll.WaitingForInput) and not getattr(
            req, "disagg_p_ready_transfer_started", False
        ):
            if not self._submit_tp_prefill_transfer(req):
                # The submitter reports local failure.  Every rank remains in
                # the state machine until TP0 publishes one group failure.
                return int(KVPoll.Transferring)
        return int(KVPoll.Transferring)

    def _submit_tp_prefill_transfer(self: Scheduler, req: Req) -> bool:
        """Submit one physical P->D shard at a native TP scheduler boundary."""

        if self.tp_size <= 1 or getattr(req, "disagg_p_ready_transfer_started", False):
            return False
        payload = getattr(req, "_async_prefill_transfer_payload", None)
        if payload is None:
            return False
        num_pages, page_indices, state_indices = payload
        try:
            req.disagg_kv_sender.init(num_pages, req.metadata_buffer_index)
            req.disagg_kv_sender.send(page_indices, state_indices)
            req.disagg_p_ready_transfer_started = True
            req.time_stats.set_prefill_transfer_queue_entry_time()
            logger.info(
                "AgenticKV tp_p2d_submit key=%s rank=%d/%d pages=%d",
                self._prefill_transfer_key(req),
                self.tp_rank,
                self.tp_size,
                num_pages,
            )
            return True
        except Exception as error:
            logger.exception(
                "TP P->D shard submission failed key=%s rank=%d/%d",
                self._prefill_transfer_key(req),
                self.tp_rank,
                self.tp_size,
            )
            req.disagg_p_ready_transfer_started = True
            fence_failed_launch = getattr(
                req.disagg_kv_sender, "fence_failed_launch", None
            )
            local_poll = int(KVPoll.Transferring)
            if fence_failed_launch is not None:
                try:
                    local_poll = int(fence_failed_launch(error))
                except Exception:
                    # Losing the fence itself requires process-lifetime source
                    # quarantine.  Never manufacture a physical terminal from
                    # a control-plane exception.
                    logger.exception(
                        "Unable to fence failed TP P->D launch key=%s rank=%d/%d",
                        self._prefill_transfer_key(req),
                        self.tp_rank,
                        self.tp_size,
                    )
            self.agentic_tp_p2d_sender_mailbox.publish_local(
                self._prefill_transfer_key(req), local_poll
            )
            if local_poll in (int(KVPoll.Success), int(KVPoll.Failed)):
                req._agentic_p2d_sender_terminal = local_poll
            # The launch attempt is now owned by its physical fence, even if
            # that fence must quarantine the pages indefinitely.
            return True

    def _prefill_transfer_consumer_worker(self: Scheduler, consumer_index: int) -> None:
        """Round-robin P->D progress without per-request worker ownership."""

        cycles = 0
        total_seconds = 0.0
        max_seconds = 0.0
        last_stats_at = time.monotonic()
        # TP has one physical-shard worker per rank.  The configured progress
        # interval is a full-queue sweep cadence, not a per-request delay: at
        # c256 a 15 ms sleep after every request would make one control sweep
        # take about four seconds and starve otherwise-empty D workers.
        tp_background = bool(
            getattr(self, "_prefill_transfer_tp_background_enabled", False)
        )
        tp_sweep_remaining = 0
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
                key = self._prefill_transfer_key(req)
                if tp_background and tp_sweep_remaining <= 0:
                    # Include the item just removed.  Requests arriving during
                    # this sweep are picked up by the next sweep, preserving a
                    # bounded and deterministic polling cadence.
                    tp_sweep_remaining = len(self._prefill_ready_queue) + 1
                self._prefill_queued_keys().discard(key)
                if not getattr(req, "_async_prefill_transfer_consumer_active", False):
                    req._async_prefill_transfer_consumer_active = True
                    req._async_prefill_transfer_active_at = time.monotonic()
                    self._prefill_transfer_active_reqs[key] = req

            poll = int(KVPoll.Failed)
            elapsed = 0.0
            try:
                if (
                    getattr(req, "disagg_p_ready_deferred", False)
                    and getattr(self, "tp_size", 1) == 1
                ):
                    self._publish_deferred_prefill_ready(req)
                started_at = time.perf_counter()
                nixl_lock = getattr(self, "agentic_nixl_control_lock", None)
                if getattr(self, "tp_size", 1) > 1:
                    # TP P->D owns an independent state machine.  It shares a
                    # Python lock only on older NIXL versions. With native RW
                    # synchronization every rank-local state machine can
                    # progress independently of reverse D->P activity.
                    native_rw = bool(
                        getattr(
                            getattr(req.disagg_kv_sender, "kv_mgr", None),
                            "thread_sync_rw_enabled",
                            False,
                        )
                    )
                    if nixl_lock is None or native_rw:
                        poll = self._prefill_transfer_progress_tp_req_once(req)
                    else:
                        with nixl_lock:
                            poll = self._prefill_transfer_progress_tp_req_once(req)
                else:
                    poll = self._prefill_transfer_progress_tp1_req_once(req)
                elapsed = time.perf_counter() - started_at
                cycles += 1
                total_seconds += elapsed
                max_seconds = max(max_seconds, elapsed)
                now = time.monotonic()
                previous_poll = getattr(req, "_async_prefill_transfer_last_poll", None)
                req._async_prefill_transfer_last_poll = poll
                req._async_prefill_transfer_last_poll_at = now
                if previous_poll != poll:
                    req._async_prefill_transfer_last_progress_at = now
            except Exception as error:
                logger.exception(
                    "P->D transfer consumer=%d failed rid=%s",
                    consumer_index,
                    req.rid,
                )
                poll = int(KVPoll.Transferring)
                fence_failed_launch = getattr(
                    req.disagg_kv_sender, "fence_failed_launch", None
                )
                if fence_failed_launch is not None:
                    try:
                        poll = int(fence_failed_launch(error))
                    except Exception:
                        logger.exception(
                            "Unable to fence P->D worker exception rid=%s; "
                            "quarantining source KV",
                            req.rid,
                        )
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

            terminal = poll in (int(KVPoll.Success), int(KVPoll.Failed))
            if terminal:
                # For TP background progress this worker is the sole terminal
                # authority.  Quiesce every source of mailbox publication
                # before making the terminal result scheduler-visible;
                # otherwise scheduler cleanup can clear the group while this
                # worker is still able to recreate a sender file.
                if getattr(self, "tp_size", 1) > 1:
                    self.agentic_tp_p2d_sender_mailbox.publish_local(key, poll)
                with self._prefill_ready_condition:
                    self._prefill_transfer_active_reqs.pop(key, None)
                with self._prefill_transfer_poll_lock:
                    req._async_prefill_transfer_consumer_active = False
                    req._async_prefill_transfer_poll = poll
                    if self.tp_size == 1:
                        self._prefill_transfer_terminal_queue.append(key)
            elif not self._prefill_transfer_stop.is_set():
                # Keep the transfer active but relinquish this worker after
                # one step, so later P results cannot be starved by waiters.
                if not tp_background:
                    self._prefill_transfer_stop.wait(
                        max(0.0, self._prefill_transfer_interval - elapsed)
                    )
                if not self._prefill_transfer_stop.is_set():
                    with self._prefill_ready_condition:
                        self._prefill_ready_queue.append(req)
                        self._prefill_queued_keys().add(key)
                        self._prefill_ready_condition.notify()

            if tp_background:
                tp_sweep_remaining -= 1
                if tp_sweep_remaining <= 0 and not self._prefill_transfer_stop.is_set():
                    self._prefill_transfer_stop.wait(
                        max(0.0, self._prefill_transfer_interval - elapsed)
                    )

            now = time.monotonic()
            if now - last_stats_at >= 30.0:
                with self._prefill_ready_condition:
                    buffered = len(self._prefill_ready_queue)
                    active = len(self._prefill_transfer_active_reqs)
                    active_requests = tuple(self._prefill_transfer_active_reqs.values())
                    oldest_active_seconds = max(
                        (
                            now
                            - getattr(
                                active_req,
                                "_async_prefill_transfer_active_at",
                                now,
                            )
                            for active_req in active_requests
                        ),
                        default=0.0,
                    )
                    notified = sum(
                        bool(getattr(active_req, "disagg_p_ready_notified", False))
                        for active_req in active_requests
                    )
                    submitted = sum(
                        bool(
                            getattr(
                                active_req,
                                "disagg_p_ready_transfer_started",
                                False,
                            )
                        )
                        for active_req in active_requests
                    )
                    poll_states = {}
                    for active_req in active_requests:
                        state = int(
                            getattr(
                                active_req,
                                "_async_prefill_transfer_last_poll",
                                -1,
                            )
                        )
                        poll_states[state] = poll_states.get(state, 0) + 1
                logger.info(
                    "Prefill P->D consumer stats worker=%d cycles=%d "
                    "avg_us=%.1f max_ms=%.3f ready_buffer=%d active=%d "
                    "oldest_active_s=%.3f inflight=%d notified=%d "
                    "submitted=%d poll_states=%s",
                    consumer_index,
                    cycles,
                    total_seconds * 1e6 / max(1, cycles),
                    max_seconds * 1000.0,
                    buffered,
                    active,
                    oldest_active_seconds,
                    len(self.disagg_prefill_inflight_queue),
                    notified,
                    submitted,
                    poll_states,
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
            self._report_tp_prefill_producer_ready(req)
            return True
        if req.return_logprob:
            # The client reconstructs generated token ids from the returned
            # logprob records, whereas Decode seeds ``req.output_ids`` from a
            # separate metadata field.  They must describe the same sampled
            # first token before either field is snapshotted for DMA.
            output_token_logprobs_idx = getattr(
                getattr(req, "logprob", None),
                "output_token_logprobs_idx",
                getattr(req, "output_token_logprobs_idx", None),
            )
            if (
                not req.output_ids
                or not output_token_logprobs_idx
                or int(req.output_ids[0]) != int(output_token_logprobs_idx[0])
            ):
                return False
        if self.req_to_metadata_buffer_idx_allocator.available_size() == 0:
            return False

        req.metadata_buffer_index = self.req_to_metadata_buffer_idx_allocator.alloc()
        assert req.metadata_buffer_index is not None

        page_size = self.token_to_kv_pool_allocator.page_size
        # ``Req.fill_ids`` became ``get_fill_ids()`` plus the authoritative
        # ``fill_len`` scalar in the newer scheduler representation.  Only the
        # length is needed here, and keeping the legacy fallback makes the
        # transfer helper usable by older test doubles as well.
        fill_len = getattr(req, "fill_len", None)
        if fill_len is None:
            fill_len = len(getattr(req, "fill_ids", ()))
        end_idx = min(int(fill_len), len(req.origin_input_ids))
        kv_indices = (
            self.req_to_token_pool.req_to_token[
                req.req_pool_idx, req.start_send_idx : end_idx
            ]
            .cpu()
            .numpy()
        )
        req.start_send_idx = end_idx
        self.disagg_metadata_buffers.set_buf(req)

        def _mamba_payload():
            return p2d_mamba_source_indices(
                req,
                page_size,
                preserve_checkpoint=envs.SGLANG_AGENTIC_KV_LIFECYCLE.get(),
            )[0]

        def _swa_payload():
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
            return kv_to_page_indices(window_kv_indices_swa.cpu().numpy(), page_size)

        def _dsa_payload():
            return kv_to_page_indices(
                self.req_to_token_pool.req_to_token[req.req_pool_idx, :end_idx]
                .cpu()
                .numpy(),
                page_size,
            )

        def _swa_ring_payload():
            kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
            ring_stride = kv_pool.unified_swa_ring_size
            window_size = kv_pool.unified_swa_window
            window_start = max(0, end_idx - window_size)
            positions = np.arange(window_start, end_idx, dtype=np.int64)
            return (
                int(req.req_pool_idx) * ring_stride + (positions % ring_stride)
            ).astype(np.int32)

        state_indices = []
        state_types = self.disagg_prefill_bootstrap_queue.kv_manager.kv_args.state_types
        for state_type in state_types:
            if state_type == StateType.MAMBA:
                state_indices.append(_mamba_payload())
            elif state_type == StateType.SWA:
                state_indices.append(_swa_payload())
            elif state_type == StateType.DSA:
                state_indices.append(_dsa_payload())
            elif state_type == StateType.SWA_RING:
                state_indices.append(_swa_ring_payload())
            else:
                state_indices.append(None)

        page_indices = kv_to_page_indices(kv_indices, page_size)
        req._async_prefill_transfer_payload = (
            kv_to_page_num(end_idx, page_size),
            page_indices,
            state_indices,
        )
        self._report_tp_prefill_producer_ready(req)
        return True

    def _write_p_ready_marker(
        self: Scheduler,
        req: Req,
        ready_path: str,
        ready_sequence: int,
        ready_metadata: dict,
    ) -> bool:
        """Let TP0 publish the logical P-ready event for the whole TP group."""

        tp_size = int(getattr(self, "tp_size", 1))
        tp_rank = int(getattr(self, "tp_rank", 0))
        if tp_size > 1 and tp_rank != 0:
            return True

        tmp_path = f"{ready_path}.{os.getpid()}.{ready_sequence}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(ready_metadata, handle, separators=(",", ":"))
        os.replace(tmp_path, ready_path)
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
        publish_condition = getattr(self, "_prefill_ready_publish_condition", None)
        ready_path = os.path.join(
            self.disagg_prefill_bootstrap_queue.p_ready_dir,
            f"{req.bootstrap_room}.ready",
        )
        ready_metadata = {
            "rid": req.rid,
            "num_kv_tokens": len(req.origin_input_ids),
            "ready_sequence": ready_sequence,
            "prefill_domain": int(
                os.environ.get("SGLANG_AGENTIC_KV_PREFILL_DOMAIN", "0")
            ),
        }
        if publish_condition is None:
            req.disagg_p_ready_notified = self._write_p_ready_marker(
                req, ready_path, ready_sequence, ready_metadata
            )
            return
        # Multiple consumers may finish transport control calls out of order,
        # but the Router must observe the producer's FIFO completion order.
        with publish_condition:
            if (
                getattr(self, "tp_size", 1) > 1
                and ready_sequence != self._prefill_ready_next_publish_sequence
            ):
                # TP rank0 revisits the scheduler-owned inflight FIFO on the
                # next iteration.  Never block its native TP broadcast while
                # waiting for an earlier producer marker: followers and model
                # execution both depend on that broadcast making progress.
                return
            while (
                ready_sequence != self._prefill_ready_next_publish_sequence
                and not self._prefill_transfer_stop.is_set()
            ):
                publish_condition.wait(timeout=0.1)
            if self._prefill_transfer_stop.is_set():
                return
            req.disagg_p_ready_notified = self._write_p_ready_marker(
                req, ready_path, ready_sequence, ready_metadata
            )
            if not req.disagg_p_ready_notified:
                return
            self._prefill_ready_next_publish_sequence += 1
            publish_condition.notify_all()

    def _prefill_transfer_cached_polls(
        self: Scheduler, requests: Optional[List[Req]] = None
    ) -> list[int]:
        with self._prefill_transfer_poll_lock:
            polls = []
            for req in (
                self.disagg_prefill_inflight_queue if requests is None else requests
            ):
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
            mode = (
                os.environ.get("SGLANG_PD_P_READY_BACKPRESSURE_MODE", "hysteresis")
                .strip()
                .lower()
            )
            high = float(os.environ.get("SGLANG_PD_P_READY_HBM_HIGH_WATERMARK", "0.70"))
            low = float(os.environ.get("SGLANG_PD_P_READY_HBM_LOW_WATERMARK", "0.55"))
            max_inflight = int(os.environ.get("SGLANG_PD_P_READY_MAX_INFLIGHT", "48"))
            request_cap = int(
                os.environ.get("SGLANG_PD_P_READY_REQUEST_CAP", str(max_inflight))
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
            # Complete Direct/Slow workset leases are allocated from the same
            # ordinary P workspace as normal Prefill. There is no fixed Direct
            # receive region and therefore no percentage watermark here.
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
        ready_token_cap = max(1, int(self.max_total_num_tokens * token_cap_fraction))
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

    def resolve_waiting_queue_bootstrap(self: Scheduler) -> None:
        """Resolve bootstrap status for waiting prefill requests before admission.

        Covers the window between leaving the bootstrap queue and being admitted
        into a running batch: aborts requests whose decode peer died, and
        finalizes optimistic requests whose bootstrap completed so they skip
        the post-forward bootstrap check.
        """
        # Late-bound P-ready requests deliberately skip the native D bootstrap
        # and are advanced by the keyed TP sender/receiver mailboxes after
        # Prefill.  Agentic admission may leave those requests in temporarily
        # different rank-local waiting queues, so including them in this
        # queue-shaped collective can mismatch both the tensor shape and the
        # collective call count across TP ranks.
        candidates = [
            req
            for req in self.waiting_queue
            if not is_aborted(req)
            and not getattr(req, "disagg_p_ready_deferred", False)
        ]
        if not candidates:
            return
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in candidates],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )
        failed = set()
        for req, poll in zip(candidates, polls):
            if poll == KVPoll.Failed:
                self.handle_bootstrap_failure(req)
                failed.add(req)
            elif (
                poll == KVPoll.WaitingForInput
                and req.pending_bootstrap
                and not should_force_retry(req)
            ):
                # Optimistic requests reserved a metadata buffer when popped, so
                # finalize cannot fail here; if it ever does, the request stays
                # pending and the post-forward check resolves it.
                self.disagg_prefill_bootstrap_queue.finalize_bootstrap(req)
        if failed:
            self.waiting_queue = [
                req for req in self.waiting_queue if req not in failed
            ]

    @scheduler_nvtx_method("scheduler.get_next_batch_to_run")
    def get_next_disagg_prefill_batch_to_run(
        self: Scheduler,
    ) -> Optional[ScheduleBatch]:
        service_worksets = getattr(self, "_agentic_service_p_workset_leases", None)
        if service_worksets is not None:
            service_worksets()
        self.process_pending_chunked_abort()

        # HACK (byronhsu): reset the batch_is_full flag because we never enter update_running_batch which resets it
        # Otherwise, it hangs under high concurrency
        self.running_batch.batch_is_full = False

        self.process_prefill_chunk()

        self.resolve_waiting_queue_bootstrap()

        batch = self.get_new_batch_prefill()
        batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(batch)

        if batch:
            set_schedule_time_batch(batch)

        return batch

    @torch.no_grad()
    def event_loop_normal_disagg_prefill(self: Scheduler) -> None:
        """A normal scheduler loop for prefill worker in disaggregation mode."""
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()

        while True:
            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            self._merge_disagg_prefill_ready(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )
            if self._engine_paused:
                continue

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
                self.on_idle()

            self.process_disagg_prefill_inflight_queue()

            # Update last_batch
            self.last_batch = batch

    @torch.no_grad()
    def event_loop_overlap_disagg_prefill(self: Scheduler) -> None:
        self.result_queue = deque()
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()

        while True:
            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            self._merge_disagg_prefill_ready(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )
            if self._engine_paused:
                continue

            self._apply_war_barrier()

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
                self.on_idle()

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
        if result.routed_experts_output is not None:
            result.routed_experts_output.finalize()
            result.routed_experts_output = None
        if result.indexer_topk_output is not None:
            result.indexer_topk_output.finalize()
            result.indexer_topk_output = None

        logprob_pt = 0
        # Transfer kv for prefill completed requests and add it into disagg_prefill_inflight_queue
        next_token_ids = result.next_token_ids.tolist()
        self.batch_result_processor.move_logprobs_to_cpu(
            batch=batch,
            logits_output=logits_output,
        )

        def advance_logprob_pt(i: int, req: Req) -> None:
            nonlocal logprob_pt
            if not req.return_logprob or extend_input_len_per_req is None:
                return
            extend_logprob_start_len = extend_logprob_start_len_per_req[i]
            extend_input_len = extend_input_len_per_req[i]
            if extend_logprob_start_len < extend_input_len:
                logprob_pt += extend_input_len - extend_logprob_start_len

        # Poll optimistic prefill requests in this batch.
        # Note: In overlap scheduling, a chunked request that was still pending
        # during process_prefill_chunk is not checked again here.
        # If it becomes ready in the gap, we still retry the request to keep
        # chunked-prefill state management simple.
        optimistic_polls = {}
        optimistic_reqs = [
            (i, req)
            for i, req in enumerate(batch.reqs)
            if req.pending_bootstrap and req.inflight_middle_chunks <= 0
        ]
        if optimistic_reqs:
            polls = poll_and_all_reduce_attn_cp_tp_group(
                [req.disagg_kv_sender for _, req in optimistic_reqs],
                self.attn_cp_cpu_group,
                self.attn_tp_cpu_group,
            )
            optimistic_polls = {
                idx: poll for (idx, _), poll in zip(optimistic_reqs, polls)
            }

        for i, (req, next_token_id) in enumerate(
            zip(batch.reqs, next_token_ids, strict=True)
        ):
            if req.inflight_middle_chunks <= 0:
                req.time_stats.set_prefill_finished_time()

                # For optimistic requests, check bootstrap before side effects
                if i in optimistic_polls:
                    if not self.handle_pending_bootstrap(
                        req, optimistic_polls[i], defer_release=False
                    ):
                        advance_logprob_pt(i, req)
                        continue

                req.output_ids.append(next_token_id)
                token_to_kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
                p2d_mamba_tracked_tokens = (
                    getattr(req, "mamba_last_track_seqlen", None)
                    if envs.SGLANG_AGENTIC_KV_LIFECYCLE.get()
                    and hasattr(token_to_kv_pool, "mamba_pool")
                    else None
                )
                if (
                    p2d_mamba_tracked_tokens is None
                    and envs.SGLANG_AGENTIC_KV_LIFECYCLE.get()
                    and hasattr(token_to_kv_pool, "mamba_pool")
                ):
                    page_size = self.token_to_kv_pool_allocator.page_size
                    expected_checkpoint = p2d_mamba_checkpoint_tokens(req, page_size)
                    protected = int(getattr(req, "cache_protected_len", 0) or 0)
                    # A final chunk shorter than one page creates no new track
                    # event.  The checkpoint donated by the preceding chunk is
                    # already the exact floor(prompt/page) parent.
                    if protected == expected_checkpoint:
                        p2d_mamba_tracked_tokens = protected
                maybe_cache_unfinished_req(req, self.tree_cache)
                if envs.SGLANG_AGENTIC_KV_LIFECYCLE.get() and hasattr(
                    token_to_kv_pool, "mamba_pool"
                ):
                    freeze_p2d_mamba_checkpoint_after_cache(
                        req,
                        p2d_mamba_tracked_tokens,
                        self.token_to_kv_pool_allocator.page_size,
                    )
                self.disagg_prefill_inflight_queue.append(req)
                p2d_host = getattr(self, "agentic_p2d_host_staging_manager", None)
                if p2d_host is not None:
                    # Register once at the producer boundary.  The manager's
                    # offer watcher starts D2H as soon as D advertises that
                    # direct admission failed, even while the scheduler is in
                    # the next Prefill forward.
                    p2d_source_indices = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, : len(req.origin_input_ids)
                    ].clone()
                    p2d_host.watch(req, p2d_source_indices)
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
                    self.batch_result_processor.logprob_result_processor.add_logprob_return_values(
                        i,
                        req,
                        logprob_pt,
                        next_token_ids,
                        num_input_logprobs,
                        logits_output,
                    )
                    logprob_pt += num_input_logprobs
                if getattr(req, "disagg_p_ready_deferred", False):
                    async_progress = getattr(
                        self, "_prefill_transfer_async_enabled", False
                    )
                    if not async_progress or self._prepare_deferred_prefill_transfer(
                        req
                    ):
                        if getattr(self, "_prefill_transfer_async_enabled", False):
                            self._enqueue_deferred_prefill_transfer(req)
                        else:
                            self._publish_deferred_prefill_ready(req)
                    elif self.tp_size == 1:
                        key = self._prefill_transfer_key(req)
                        if key not in self._prefill_transfer_prepare_keys:
                            self._prefill_transfer_prepare_keys.add(key)
                            self._prefill_transfer_prepare_queue.append(req)
                else:
                    self.send_kv_chunk(req, last_chunk=True)
                    req.disagg_p_ready_transfer_started = True
                    req.time_stats.set_prefill_transfer_queue_entry_time()
                    if getattr(self, "_prefill_transfer_async_enabled", False):
                        self._enqueue_deferred_prefill_transfer(req)

                if req.grammar is not None:
                    try:
                        req.grammar.accept_token(next_token_id)
                    except ValueError as e:
                        error_message = f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                        release_safe = p2d_host is None or p2d_host.cancel_watch(req)
                        if release_safe:
                            release_kv_cache(req, self.tree_cache)
                            prepare_abort(
                                req,
                                error_message,
                                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                            )
                        else:
                            # Extremely rare: Host D2H won concurrently with
                            # grammar validation.  Preserve pages until that
                            # copy reaches a terminal poll, then abort through
                            # the normal inflight cleanup path.
                            req._agentic_p2d_abort_after_copy = error_message
                    req.grammar.finished = req.finished()
            else:
                # being chunked reqs' prefill is not finished
                req.inflight_middle_chunks -= 1

                # Overlap deferred release for optimistic requests stopped in process_prefill_chunk
                if req.pending_bootstrap:
                    advance_logprob_pt(i, req)
                    self.optimistic_release_and_requeue(req)
                    req.time_stats.set_last_chunked_prefill_finish_time()
                    continue

                # Optimistic bootstrap can fail while this overlapped chunk is
                # already running. Drop aborted chunks instead of sending KV.
                if is_aborted(req):
                    advance_logprob_pt(i, req)
                    req.time_stats.set_last_chunked_prefill_finish_time()
                    continue

                if req.return_logprob:
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    if extend_logprob_start_len < extend_input_len:
                        num_input_logprobs = extend_input_len - extend_logprob_start_len
                        self.batch_result_processor.logprob_result_processor.add_input_logprob_return_values(
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

        can_run_cuda_graph = result.can_run_cuda_graph
        self.metrics_reporter.report_prefill_stats(
            batch=batch,
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

        full_inflight_queue = self.disagg_prefill_inflight_queue
        inflight_queue = full_inflight_queue
        unselected_reqs = []
        if self.tp_size == 1 and getattr(
            self, "_prefill_transfer_async_enabled", False
        ):
            # Retry only requests that explicitly failed immutable payload
            # preparation; do not rescan every P-ready request each tick.
            prepare_budget = min(8, len(self._prefill_transfer_prepare_queue))
            if prepare_budget:
                live_ids = {id(req) for req in full_inflight_queue}
                for _ in range(prepare_budget):
                    req = self._prefill_transfer_prepare_queue.popleft()
                    key = self._prefill_transfer_key(req)
                    self._prefill_transfer_prepare_keys.discard(key)
                    if id(req) not in live_ids:
                        continue
                    if self._prepare_deferred_prefill_transfer(req):
                        self._enqueue_deferred_prefill_transfer(req)
                    elif key not in self._prefill_transfer_prepare_keys:
                        self._prefill_transfer_prepare_keys.add(key)
                        self._prefill_transfer_prepare_queue.append(req)

            with self._prefill_transfer_poll_lock:
                terminal_keys = set(self._prefill_transfer_terminal_queue)
                self._prefill_transfer_terminal_queue.clear()
            if not terminal_keys:
                return []
            selected = []
            for req in full_inflight_queue:
                if self._prefill_transfer_key(req) in terminal_keys:
                    selected.append(req)
                else:
                    unselected_reqs.append(req)
            inflight_queue = selected
            if not inflight_queue:
                return []
        if self.tp_size > 1:
            selected_keys = getattr(self, "_agentic_tp_prefill_transfer_keys", ())
            if not selected_keys:
                return []
            by_key = {
                (str(req.rid), int(req.bootstrap_room)): req
                for req in full_inflight_queue
            }
            inflight_queue = [by_key[key] for key in selected_keys if key in by_key]
            if not inflight_queue:
                return []

        done_reqs = []

        p2d_host = getattr(self, "agentic_p2d_host_staging_manager", None)

        if getattr(self, "_prefill_transfer_async_enabled", False):
            # Metadata-buffer slots are bounded.  If none was available when
            # the Prefill result was finalized, retry preparation here on the
            # scheduler thread.  Never publish P-ready before preparation:
            # Decode must not advertise destination pages to a worker that has
            # no immutable, fully-populated metadata payload to send.
            for req in inflight_queue:
                if getattr(
                    req, "disagg_p_ready_deferred", False
                ) and self._prepare_deferred_prefill_transfer(req):
                    if not getattr(req, "disagg_p_ready_notified", False):
                        self._enqueue_deferred_prefill_transfer(req)
            if self.tp_size > 1:
                tp_background = bool(
                    getattr(self, "_prefill_transfer_tp_background_enabled", False)
                )
                submit_keys = set(getattr(self, "_agentic_tp_prefill_submit_keys", ()))
                if not tp_background:
                    for req in inflight_queue:
                        request_key = (str(req.rid), int(req.bootstrap_room))
                        if request_key in submit_keys and not getattr(
                            req, "disagg_p_ready_transfer_started", False
                        ):
                            self._submit_tp_prefill_transfer(req)
                        local_poll = self._prefill_transfer_progress_req_once(req)
                        self.agentic_tp_p2d_sender_mailbox.publish_local(
                            self._prefill_transfer_key(req), int(local_poll)
                        )
                group_status = getattr(
                    self, "_agentic_tp_prefill_transfer_group_status", {}
                )
                polls = [
                    group_status.get(
                        (str(req.rid), int(req.bootstrap_room)),
                        int(KVPoll.Transferring),
                    )
                    for req in inflight_queue
                ]
            else:
                polls = self._prefill_transfer_cached_polls(inflight_queue)
        else:
            pollers = [
                (
                    _P2DHostOrNativePoller(p2d_host, req)
                    if p2d_host is not None
                    else req.disagg_kv_sender
                )
                for req in inflight_queue
            ]
            polls = poll_and_all_reduce_attn_cp_tp_group(
                pollers,
                self.attn_cp_cpu_group,
                self.attn_tp_cpu_group,
            )
        if (
            self.tp_size == 1
            and p2d_host is not None
            and getattr(self, "_prefill_transfer_async_enabled", False)
        ):
            polls = [
                host_poll if host_poll is not None else poll
                for req, poll in zip(inflight_queue, polls)
                for host_poll in [p2d_host.poll(req)]
            ]

        undone_reqs: List[Req] = list(unselected_reqs)
        # Check .poll() for the reqs in disagg_prefill_inflight_queue. If Success, respond to the client and remove it from the queue
        for req, poll in zip(inflight_queue, polls):

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
                    logger.warning_once(
                        f"PP rank {self.ps.pp_rank}: unexpected poll state {poll} for rid {req.rid} "
                        f"from consensus; treating as undone",
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
                if self.tp_size > 1:
                    # The native TP broadcast has made this terminal result
                    # scheduler-visible on every rank.  Cache it on the Req
                    # before group mailbox cleanup, so a background worker
                    # that is between its receipt read and active-map cleanup
                    # observes the same level-triggered terminal instead of
                    # re-entering Transferring after the receipt disappears.
                    req._agentic_p2d_group_terminal = int(KVPoll.Success)
                # Resolve native-vs-Host ownership before freeing any GPU
                # page.  A Host claim arriving after the last native poll
                # keeps the request inflight until its D2H completes.
                if p2d_host is not None and not p2d_host.prepare_scheduler_release(req):
                    undone_reqs.append(req)
                    continue
                staged_p2d = bool(
                    p2d_host is not None
                    and getattr(req, "_agentic_p2d_host_snapshot_id", None)
                )
                agentic_metadata = (
                    AgenticRequestMetadata.from_req(req)
                    if envs.SGLANG_AGENTIC_KV_LIFECYCLE.get()
                    else None
                )
                # release_kv_cache pops kv_committed_len, but the following
                # request-private branch cleanup still needs the exact prefix
                # depth to rematch the just-inserted radix node.
                agentic_committed_len = (
                    len(req.origin_input_ids) if agentic_metadata is not None else None
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
                # The workset reserves the complete page containing the new
                # suffix so a restored parent can always enter Prefill.  The
                # native paged extend path may use its own output page; in
                # that case the broker's handed suffix is still live after
                # release_kv_cache(req).  Retire it explicitly at the same
                # P->D ownership boundary.  If the suffix was consumed by a
                # workset-aware allocator, release_handed is an idempotent
                # no-op because the lease has already left the broker.
                workset_lease = getattr(req, "_agentic_p_workset_lease", None)
                workset_broker = getattr(req, "_agentic_p_workset_broker", None)
                if workset_lease is not None and workset_broker is not None:
                    workset_broker.release_handed(
                        workset_lease.snapshot_id,
                        workset_lease,
                        req=req,
                    )
                for name in (
                    "_agentic_workset_backed",
                    "_agentic_p_workset_lease",
                    "_agentic_p_workset_broker",
                    "_agentic_workset_suffix_allocated_tokens",
                    "_agentic_workset_suffix_indices",
                ):
                    if hasattr(req, name):
                        delattr(req, name)
                abort_after_copy = getattr(req, "_agentic_p2d_abort_after_copy", None)
                if abort_after_copy is None:
                    req.finished_reason = FINISH_LENGTH(length=0)
                else:
                    delattr(req, "_agentic_p2d_abort_after_copy")
                    prepare_abort(
                        req,
                        abort_after_copy,
                        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                # FIXME: clean up req's data in transfer engine
                if hasattr(req.disagg_kv_sender, "clear"):
                    req.disagg_kv_sender.clear()
                if staged_p2d:
                    logger.info(
                        "AgenticKV p2d_host_prefill_release snapshot=%s req=%s",
                        getattr(req, "_agentic_p2d_host_snapshot_id", ""),
                        req.rid,
                    )
                if hasattr(req, "_async_prefill_transfer_payload"):
                    delattr(req, "_async_prefill_transfer_payload")
                self._clear_tp_prefill_transfer_mailboxes(req)
                done_reqs.append(req)
                req.time_stats.set_prefill_kv_transfer_finish_time()
            elif poll == KVPoll.Failed:
                tp_rank = getattr(
                    getattr(self, "ps", None), "tp_rank", getattr(self, "tp_rank", 0)
                )
                error_message = f"Prefill transfer failed for request rank={tp_rank} {req.rid=} {req.bootstrap_room=}"
                is_propagated = False
                try:
                    req.disagg_kv_sender.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                    is_propagated = getattr(e, "is_from_another_rank", False)
                # Mute error message for propagated exceptions to avoid duplicate logging
                if is_propagated:
                    logger.debug(error_message)
                else:
                    logger.warning(error_message)
                req.time_stats.trace_ctx.abort(abort_info={"reason": error_message})
                agentic_metadata = (
                    AgenticRequestMetadata.from_req(req)
                    if envs.SGLANG_AGENTIC_KV_LIFECYCLE.get()
                    else None
                )
                if not self._cleanup_failed_prefill_transfer(
                    req, p2d_host, agentic_metadata
                ):
                    undone_reqs.append(req)
                    continue
                if self.tp_size > 1:
                    # Failed becomes a level-triggered group terminal only
                    # after cleanup has proved that Host staging did not win
                    # the ownership race.  If Host already claimed the
                    # generation, cleanup returns False and progress must keep
                    # polling that path until its durable terminal is visible.
                    req._agentic_p2d_group_terminal = int(KVPoll.Failed)
                prepare_abort(
                    req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
                )
                done_reqs.append(req)
                if self.metrics_reporter.enable_metrics:
                    self.metrics_collector.increment_transfer_failed_reqs()
            else:
                logger.warning_once(
                    f"Unexpected polling state {poll} for rid {req.rid} in inflight queue; "
                    f"treating as undone",
                )
                undone_reqs.append(req)

        for req in done_reqs:
            req.time_stats.set_completion_time()

        for req in done_reqs:
            if isinstance(req.finished_reason, FINISH_ABORT):
                continue
            if req.bootstrap_host == FAKE_BOOTSTRAP_HOST:
                continue
            kv_mgr = getattr(req.disagg_kv_sender, "kv_mgr", None)
            if kv_mgr and getattr(kv_mgr, "is_dummy_cp_rank", False):
                continue
            metrics = req.time_stats.compute_and_observe_kv_transfer_metrics(
                req.disagg_kv_sender.get_transfer_metric()
            )
            if metrics:
                # Update last-value for REST API
                if "latency_ms" in metrics:
                    self.metrics_reporter.kv_transfer_latency_ms = metrics["latency_ms"]
                if "speed_gb_s" in metrics:
                    self.metrics_reporter.kv_transfer_speed_gb_s = metrics["speed_gb_s"]

        # Stream requests which have finished transfer
        if hasattr(self, "output_streamer"):
            self.output_streamer.stream_output(
                done_reqs,
                any(req.return_logprob for req in done_reqs),
                None,
            )
        else:
            # Compatibility for isolated state-machine tests and older
            # scheduler wrappers; production v0.5.14 uses output_streamer.
            self.stream_output(
                done_reqs,
                any(req.return_logprob for req in done_reqs),
                None,
            )
        for req in done_reqs:
            req: Req

            maybe_release_metadata_buffer(
                req, self.req_to_metadata_buffer_idx_allocator
            )

        if self.tp_size == 1 and getattr(
            self, "_prefill_transfer_async_enabled", False
        ):
            selected_ids = {id(req) for req in inflight_queue}
            with self._prefill_transfer_poll_lock:
                for req in undone_reqs:
                    if id(req) not in selected_ids:
                        continue
                    poll = getattr(req, "_async_prefill_transfer_poll", None)
                    if poll in (int(KVPoll.Success), int(KVPoll.Failed)):
                        self._prefill_transfer_terminal_queue.append(
                            self._prefill_transfer_key(req)
                        )
            self._release_prefill_transfer_poll_claims(undone_reqs)
        if self.tp_size > 1:
            done_ids = {id(req) for req in done_reqs}
            self.disagg_prefill_inflight_queue = [
                req for req in full_inflight_queue if id(req) not in done_ids
            ]
        else:
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

    def handle_bootstrap_failure(self: Scheduler, req: Req) -> None:
        error_message = (
            f"Prefill bootstrap failed for request rank={self.ps.tp_rank} "
            f"{req.rid=} {req.bootstrap_room=}"
        )
        is_propagated = False
        try:
            req.disagg_kv_sender.failure_exception()
        except Exception as e:
            error_message += f" with exception {e}"
            is_propagated = getattr(e, "is_from_another_rank", False)
        # Mute error message for propagated exceptions to avoid duplicate logging
        if is_propagated:
            logger.debug(error_message)
        else:
            logger.warning(error_message)
        req.time_stats.trace_ctx.abort(abort_info={"reason": error_message})
        if req.req_pool_idx is not None or self.tree_cache.supports_mamba():
            release_kv_cache(req, self.tree_cache)
        maybe_release_metadata_buffer(req, self.req_to_metadata_buffer_idx_allocator)
        req.pending_bootstrap = False
        prepare_abort(req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
        self.output_streamer.stream_output([req], req.return_logprob)
        if self.metrics_reporter.enable_metrics:
            self.metrics_collector.increment_bootstrap_failed_reqs()
        if self.enable_hicache_storage:
            self.tree_cache.release_aborted_request(req.rid)

    def handle_pending_bootstrap(
        self: Scheduler, req: Req, poll: KVPoll, defer_release: bool
    ) -> bool:
        """Return True when bootstrap is finalized and KV transfer can proceed."""
        if poll == KVPoll.Failed:
            self.handle_bootstrap_failure(req)
            return False
        elif poll == KVPoll.Bootstrapping:
            if not defer_release:
                self.optimistic_release_and_requeue(req)
            return False
        elif poll == KVPoll.WaitingForInput:
            force_retry = should_force_retry(req)  # test hook
            if force_retry:
                if not defer_release:
                    self.optimistic_release_and_requeue(req)
                return False
            # Metadata buffer was allocated in pop_bootstrapped before
            # the request entered the waiting queue, so finalize should not fail.
            assert self.disagg_prefill_bootstrap_queue.finalize_bootstrap(req)
            return True
        else:
            raise RuntimeError(
                f"Unexpected poll state {poll} for req {req.rid} in handle_pending_bootstrap"
            )

    def check_bootstrap(self: Scheduler, req: Req) -> bool:
        """Check bootstrap status for an optimistic prefilled request.
        Returns True if bootstrap is finished."""
        if not req.pending_bootstrap:
            return True
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )
        return self.handle_pending_bootstrap(
            req, polls[0], defer_release=self.enable_overlap
        )

    def process_prefill_chunk(self: Scheduler) -> None:
        chunked_req_to_exclude = set()
        if self.chunked_req:
            chunked_req_to_exclude.add(self.chunked_req)
            maybe_cache_unfinished_req(self.chunked_req, self.tree_cache, chunked=True)

            if not self.check_bootstrap(self.chunked_req):
                self.chunked_req = None  # stop the current chunked prefill
            elif self.enable_overlap:
                # Delay KV transfer to process_batch_result_disagg_prefill when overlap is enabled to ensure results are resolved
                self.chunked_req.tmp_end_idx = min(
                    self.chunked_req.fill_len,
                    len(self.chunked_req.origin_input_ids),
                )
            else:
                self.send_kv_chunk(self.chunked_req)

            if self.chunked_req is not None:
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
        assert (
            req.metadata_buffer_index >= 0
        ), f"Req {req.rid} does not have metadata buffer allocated"
        page_size = self.token_to_kv_pool_allocator.page_size
        start_idx = req.start_send_idx
        end_idx = (
            end_idx
            if end_idx is not None
            else min(req.fill_len, len(req.origin_input_ids))
        )

        if not last_chunk:
            # if not the last chunk and the last page is partial, delay the last partial page to the next send
            end_idx = end_idx - end_idx % page_size

        if end_idx < start_idx:
            logger.debug(
                "send_kv_chunk skip: rid=%s start_send_idx=%s end_idx=%s",
                req.rid,
                start_idx,
                end_idx,
            )
            return

        kv_indices = (
            self.req_to_token_pool.req_to_token[req.req_pool_idx, start_idx:end_idx]
            .cpu()
            .numpy()
        )
        state_indices: Optional[List] = None
        if last_chunk:
            self.disagg_metadata_buffers.set_buf(req)

            # fill_ids includes the token sampled during prefill, but decode
            # registers state pages over origin_input_ids (DecodePreallocQueue)
            # and the main pool send is clamped to end_idx above. Matching that
            # length here avoids emitting an extra state page when the sampled
            # token crosses a page boundary, which mismatched src/dst lengths in
            # group_concurrent_contiguous.
            seq_len = min(req.fill_len, len(req.origin_input_ids))

            def _mamba_payload():
                return p2d_mamba_source_indices(
                    req,
                    page_size,
                    preserve_checkpoint=envs.SGLANG_AGENTIC_KV_LIFECYCLE.get(),
                )[0]

            def _swa_payload():
                window_size = self.sliding_window_size
                window_start = max(0, seq_len - window_size)
                window_start = (window_start // page_size) * page_size
                window_kv_indices_full = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, window_start:seq_len
                ]
                window_kv_indices_swa = (
                    self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                        window_kv_indices_full
                    )
                )
                return kv_to_page_indices(
                    window_kv_indices_swa.cpu().numpy(), page_size
                )

            def _dsa_payload():
                kv_indices_full = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, :seq_len
                ]
                return kv_to_page_indices(kv_indices_full.cpu().numpy(), page_size)

            def _swa_ring_payload():
                # Unified_kv SWA ring rows (req_pool_idx*ring_stride + pos%ring_stride)
                # for the last `window` positions, in ascending position order so
                # decode (its own req_pool_idx) matches positionally.
                _pool = self.token_to_kv_pool_allocator.get_kvcache()
                ring_stride = _pool.unified_swa_ring_size
                window_size = _pool.unified_swa_window
                window_start = max(0, seq_len - window_size)
                positions = np.arange(window_start, seq_len, dtype=np.int64)
                state_slot = int(req.req_pool_idx)
                ring_rows = state_slot * ring_stride + (positions % ring_stride)
                return ring_rows.astype(np.int32)

            state_types = (
                self.disagg_prefill_bootstrap_queue.kv_manager.kv_args.state_types
            )
            state_indices = []
            for st in state_types:
                if st == StateType.MAMBA:
                    state_indices.append(_mamba_payload())
                elif st == StateType.SWA:
                    state_indices.append(_swa_payload())
                elif st == StateType.DSA:
                    state_indices.append(_dsa_payload())
                elif st == StateType.SWA_RING:
                    state_indices.append(_swa_ring_payload())
                else:
                    state_indices.append(None)

        page_indices = kv_to_page_indices(kv_indices, page_size)
        if not req.disagg_kv_sender.should_send_kv_chunk(len(page_indices), last_chunk):
            return
        req.disagg_kv_sender.send(page_indices, state_indices)
        req.start_send_idx = end_idx

    def optimistic_release_and_requeue(self: Scheduler, req: Req) -> None:
        """Release KV cache and requeue an optimistic prefill request."""
        max_retries = self.server_args.optimistic_prefill_retries
        maybe_cache_unfinished_req(req, self.tree_cache)
        release_kv_cache(req, self.tree_cache)
        req.reset_for_retract()
        req.output_ids = array("q")
        req.start_send_idx = 0
        req.tmp_end_idx = -1
        req.hidden_states_tensor = None
        req.pending_bootstrap = True
        req.time_stats.reset_prefill_retry_time()
        if req.time_stats.prefill_retry_count >= max_retries:
            logger.info(
                f"Req {req.rid} exhausted optimistic prefill retries "
                "falling back to bootstrap queue"
            )
            # Reset it so the next real bootstrap done can be recorded.
            req.time_stats.bootstrap_done_time = 0.0
            self.disagg_prefill_bootstrap_queue.queue.append(req)
        else:
            req.time_stats.prefill_retry_count += 1
            logger.info(
                f"Req {req.rid} optimistic prefill retry "
                f"{req.time_stats.prefill_retry_count}/{max_retries}"
            )
            if self.metrics_reporter.enable_metrics:
                self.metrics_collector.increment_prefill_retries(1)
            req.time_stats.set_wait_queue_entry_time()
            self.waiting_queue.insert(0, req)
