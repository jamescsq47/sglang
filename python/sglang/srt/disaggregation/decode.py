"""
Life cycle of a request in the decode server

1. PreallocQueue:
    a. Initialize a receiver for each request
    b. The request handshakes first, and pre-allocate kv once there is available kv.
    c. Move the request to TransferQueue.

2. TransferQueue:
    a. Poll the receiver to check the transfer state
    b. If the transfer has finished, move the request to waiting queue

3. WaitingQueue:
    a. Use the requests in the queue to construct a PrebuiltExtendBatch
    b. Skip the prefill forward but only populate metadata

4. RunningBatch:
    a. Merge the resolved PrebuiltExtendBatch into running batch to run decoding
"""

from __future__ import annotations

import logging
import os
import queue as thread_queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.distributed import ProcessGroup

from sglang.srt.configs.mamba_utils import Mamba2CacheParams
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager, CommonKVReceiver
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
    poll_and_all_reduce,
    poll_and_all_reduce_with_staging,
    prepare_abort,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.managers.schedule_batch import FINISH_ABORT, ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, EvictParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    KVCache,
    NSATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.observability.req_time_stats import (
    set_schedule_time_batch,
    set_time_batch,
)
from sglang.srt.utils.network import NetworkAddress
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = logging.getLogger(__name__)


def prepare_pd_decode_radix_key(scheduler, req) -> None:
    """Use one D-local Radix namespace for agentic requests.

    The wire ``extra_key`` is generation-scoped so P-side snapshot ownership
    cannot alias across requests.  That isolation is transport metadata, not a
    model-KV property.  Keeping it as D's Radix namespace would prevent even an
    identical system prompt from sharing pages across requests.
    """

    if scheduler.tree_cache.disable:
        return
    custom_params = getattr(req.sampling_params, "custom_params", None) or {}
    if "agentic_request_id" not in custom_params:
        return
    req._pd_transport_extra_key = req.extra_key
    lora_id = getattr(req, "lora_id", None) or ""
    req.extra_key = f"agentic-pd-decode-v1:{lora_id}"


def reconcile_pd_decode_imported_prefix(scheduler, req) -> int:
    """Deduplicate a fully imported P->D snapshot against D's Radix cache."""

    if scheduler.tree_cache.disable:
        return 0
    prefix_len = len(req.prefix_indices)
    scheduler.tree_cache.inc_lock_ref(req.last_node)
    if prefix_len:
        imported_prefix = scheduler.req_to_token_pool.req_to_token[
            req.req_pool_idx, :prefix_len
        ].clone()
        scheduler.req_to_token_pool.write(
            (req.req_pool_idx, slice(0, prefix_len)), req.prefix_indices
        )
        # Slot 0 is SGLang's permanent dummy KV location.  A complete PD
        # import may contain it for padding at a page boundary; returning it
        # to the paged allocator creates one phantom free page and eventually
        # aliases two live requests to the same physical KV storage.
        imported_prefix = imported_prefix[imported_prefix != 0]
        if imported_prefix.numel():
            scheduler.token_to_kv_pool_allocator.free(imported_prefix)
    req._pd_decode_radix_reused_tokens = prefix_len
    if prefix_len:
        logger.info(
            "AgenticKV decode_radix_reuse req=%s shared_tokens=%d full_tokens=%d",
            req.rid,
            prefix_len,
            len(req.fill_ids),
        )
    return prefix_len

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.server_args import ServerArgs

CLIP_MAX_NEW_TOKEN = envs.SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION.get()


def _is_fake_transfer(req: Req, server_args: ServerArgs) -> bool:
    return req.bootstrap_host == FAKE_BOOTSTRAP_HOST or (
        req.bootstrap_host is None
        and server_args.disaggregation_transfer_backend == "fake"
    )


def _bootstrap_addr(req: Req) -> str:
    # FIXME: make a property of a req
    return NetworkAddress(req.bootstrap_host, req.bootstrap_port).to_host_port_str()


class DecodeReqToTokenPool:
    """
    The difference of DecodeReqToTokenPool and ReqToTokenPool is that
    DecodeReqToTokenPool subscribes memory for pre-allocated requests.

    In ReqToTokenPool, if `--max-running-requests` is 8,
    #pre-allocated + #transfer + #running <= 8, but there are in fact more memory can carry pre-allocated requests.

    In DecodeReqToTokenPool, if `--max-running-requests` is 8,
    #running <= 8, #pre-allocated + #transfer <= pre_alloc_size, so we can use the free memory to pre-allocate requests to unblock prefill.
    """

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        pre_alloc_size: int,
    ):
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size
        self.max_context_len = max_context_len
        self.device = device
        self.pre_alloc_size = pre_alloc_size
        with memory_saver_adapter.region(tag=GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_token = torch.zeros(
                (size + pre_alloc_size, max_context_len),
                dtype=torch.int32,
                device=device,
            )

        self.free_slots = list(range(size + pre_alloc_size))

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, reqs: List["Req"]) -> Optional[List[int]]:
        # Indices of reqs that already have a req_pool_idx and will reuse
        # their existing slot (e.g. chunked prefill continuing across chunks).
        reusing = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]
        assert (
            len(reusing) <= 1
        ), "only one chunked request may reuse req_pool_idx in a batch"
        assert all(
            reqs[i].is_chunked > 0 or reqs[i].kv_committed_len > 0 for i in reusing
        ), "reusing request must be chunked or have committed KV"

        need_size = len(reqs) - len(reusing)
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        offset = 0
        for r in reqs:
            if r.req_pool_idx is None:
                r.req_pool_idx = select_index[offset]
                offset += 1
        return [r.req_pool_idx for r in reqs]

    def free(self, req: "Req"):
        assert req.req_pool_idx is not None, "request must have req_pool_idx"
        self.free_slots.append(req.req_pool_idx)
        req.req_pool_idx = None

    def clear(self):
        self.free_slots = list(range(self.size + self.pre_alloc_size))


class HybridMambaDecodeReqToTokenPool(HybridReqToTokenPool):

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        cache_params: "Mamba2CacheParams",
        mamba_layer_ids: List[int],
        speculative_num_draft_tokens: int,
        enable_mamba_extra_buffer: bool,
        pre_alloc_size: int,
        enable_overlap_schedule: bool,
        mamba_size: int = None,
        start_layer: int = None,
    ):
        DecodeReqToTokenPool.__init__(
            self,
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
            pre_alloc_size=pre_alloc_size,
        )

        self.mamba_ping_pong_track_buffer_size = 2 if enable_overlap_schedule else 1
        self.enable_mamba_extra_buffer = enable_mamba_extra_buffer
        self.enable_memory_saver = enable_memory_saver
        if mamba_size is not None:
            effective_mamba_size = min(mamba_size, size + pre_alloc_size)
            if mamba_size > size + pre_alloc_size:
                logger.warning(
                    "mamba_size (%d) exceeds size + pre_alloc_size (%d), "
                    "capping effective_mamba_size to %d",
                    mamba_size,
                    size + pre_alloc_size,
                    effective_mamba_size,
                )
        else:
            effective_mamba_size = size + pre_alloc_size
        self.start_layer = start_layer if start_layer is not None else 0
        self.layer_transfer_counter = None
        self._init_mamba_pool(
            size=effective_mamba_size,
            mamba_spec_state_size=size + pre_alloc_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
            enable_mamba_extra_buffer=self.enable_mamba_extra_buffer,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )

    def clear(self):
        self.free_slots = list(range(self.size + self.pre_alloc_size))
        self.mamba_pool.clear()


@dataclass
class DecodeRequest:
    req: Req
    kv_receiver: CommonKVReceiver
    waiting_for_input: bool = False
    metadata_buffer_index: int = -1

    @property
    def seqlen(self) -> int:
        return self.req.seqlen


class DecodePreallocQueue:
    """
    Store the requests that are preallocating.
    """

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        draft_token_to_kv_pool: Optional[KVCache],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        scheduler: Scheduler,
        transfer_queue: DecodeTransferQueue,
        tree_cache: BasePrefixCache,
        gloo_group: ProcessGroup,
        tp_rank: int,
        tp_size: int,
        dp_size: int,
        gpu_id: int,
        bootstrap_port: int,
        max_total_num_tokens: int,
        pp_rank: int,
        num_reserved_decode_tokens: int,
        transfer_backend: TransferBackend,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.token_to_kv_pool = token_to_kv_pool_allocator.get_kvcache()
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self.is_mla_backend = is_mla_backend(self.token_to_kv_pool)
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.scheduler = scheduler
        self.transfer_queue = transfer_queue
        self.tree_cache = tree_cache  # this is always a chunk cache
        self.gloo_group = gloo_group
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.gpu_id = gpu_id
        self.bootstrap_port = bootstrap_port
        self.max_total_num_tokens = max_total_num_tokens
        self.pp_rank = pp_rank
        self.num_reserved_decode_tokens = num_reserved_decode_tokens
        self.transfer_backend = transfer_backend
        # Keep D-side KV reservation closely coupled to work that P can
        # finish soon. Zero preserves the upstream unbounded behavior.
        default_transfer_inflight = (
            "8" if envs.SGLANG_AGENTIC_KV_LIFECYCLE.get() else "0"
        )
        self.max_transfer_inflight = int(
            os.environ.get(
                "SGLANG_PD_MAX_TRANSFER_INFLIGHT", default_transfer_inflight
            )
        )
        self.p_ready_dir = os.environ.get("SGLANG_PD_P_READY_DIR", "")
        if envs.SGLANG_AGENTIC_KV_LIFECYCLE.get() and not self.p_ready_dir:
            raise ValueError(
                "SGLANG_AGENTIC_KV_LIFECYCLE requires SGLANG_PD_P_READY_DIR "
                "(use a node-local path such as /dev/shm/sglang-agentic-p-ready)"
            )
        if self.p_ready_dir:
            os.makedirs(self.p_ready_dir, exist_ok=True)
        # Queue for requests pending pre-allocation
        self.queue: List[DecodeRequest] = []
        self.retracted_queue: List[Req] = []
        self.pending_reqs: List[DecodeRequest] = []
        self._async_progress_enabled = False
        self._async_pending_lock = threading.Lock()
        self._async_metadata_work: thread_queue.SimpleQueue = (
            thread_queue.SimpleQueue()
        )
        self._async_metadata_done: thread_queue.SimpleQueue = (
            thread_queue.SimpleQueue()
        )
        self._async_metadata_pending_count = 0
        self._async_metadata_count_lock = threading.Lock()
        self._async_control_next_at = 0.0
        self._async_control_interval = max(
            0.001,
            float(os.getenv("SGLANG_AGENTIC_KV_D_CONTROL_POLL_SECONDS", "0.02")),
        )
        self._prealloc_blocked_next_log_at = 0.0
        self._ensure_retry_count: Dict[str, int] = {}
        self._max_ensure_retries: int = 15  # scheduling cycles
        self._ensure_last_attempt_time: Dict[str, float] = {}
        self._ensure_retry_interval: float = 1.0  # seconds
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()
        self.kv_manager = self._init_kv_manager()
        if self.enable_staging:
            self.transfer_queue._init_staging_handler(self.kv_manager)

        if self.scheduler.tp_worker.is_hybrid_swa:
            # FIXME: current SWA allocation allocate full kv cache size in prefill
            self.max_total_num_tokens = min(
                self.max_total_num_tokens,
                self.scheduler.tp_worker.model_runner.swa_max_total_num_tokens,
            )

    def enable_async_progress(self) -> None:
        if self.tp_size != 1:
            raise ValueError("Decode I/O async progress V1 requires TP=1")
        self._async_progress_enabled = True

    def background_progress(self) -> None:
        """Run network discovery and receiver polling outside Decode scheduling."""

        if not self._async_progress_enabled:
            return
        # Receiver discovery, handshake checks, and P-ready filesystem scans
        # are control-plane work.  Running all three at the 2 ms NIXL polling
        # cadence makes a long prealloc queue monopolize the Python GIL even
        # when no state changes.  Keep transport/metadata progress responsive,
        # but pace the O(queue) control scan independently.
        now = time.monotonic()
        if now >= self._async_control_next_at:
            self._async_control_next_at = now + self._async_control_interval
            self._resolve_pending_reqs()
            self._update_handshake_waiters()
            self._background_update_p_ready()
        self._background_prepare_metadata()

    def _background_update_p_ready(self) -> None:
        if not self.p_ready_dir:
            return
        now = time.monotonic()
        for decode_req in list(self.queue):
            if not decode_req.waiting_for_input:
                continue
            if decode_req.req.bootstrap_host == FAKE_BOOTSTRAP_HOST:
                decode_req._async_p_ready = True
                continue
            if getattr(decode_req, "_async_p_ready", False):
                continue
            if now < getattr(decode_req, "_async_p_ready_next_poll_at", 0.0):
                continue
            decode_req._async_p_ready_next_poll_at = now + 0.01
            ready_path = os.path.join(
                self.p_ready_dir, f"{decode_req.req.bootstrap_room}.ready"
            )
            if os.path.exists(ready_path):
                decode_req._async_p_ready = True
                decode_req._async_p_ready_path = ready_path

    def _background_prepare_metadata(self) -> None:
        # Metadata setup includes CPU index conversion and NIXL publication.
        # One item per 2 ms data-plane tick is already far above the admission
        # rate and avoids a burst of eight Python-heavy setups delaying Decode.
        max_batch = max(
            1, int(os.getenv("SGLANG_DECODE_IO_METADATA_BATCH", "1"))
        )
        for _ in range(max_batch):
            try:
                decode_req, page_size = self._async_metadata_work.get_nowait()
            except thread_queue.Empty:
                return
            error = None
            try:
                self._send_preallocated_metadata(decode_req, page_size)
            except Exception as exc:
                error = exc
                logger.exception(
                    "Background P->D metadata setup failed for %s",
                    decode_req.req.rid,
                )
            self._async_metadata_done.put((decode_req, error))

    def _drain_background_metadata(self):
        ready = []
        failed = []
        while True:
            try:
                decode_req, error = self._async_metadata_done.get_nowait()
            except thread_queue.Empty:
                break
            with self._async_metadata_count_lock:
                self._async_metadata_pending_count = max(
                    0, self._async_metadata_pending_count - 1
                )
            if error is None:
                ready.append(decode_req)
                continue
            prepare_abort(
                decode_req.req,
                f"P->D metadata setup failed: {error}",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            try:
                decode_req.kv_receiver.abort()
            except Exception:
                logger.debug("Receiver abort after metadata failure failed", exc_info=True)
            if decode_req.metadata_buffer_index != -1:
                self.req_to_metadata_buffer_idx_allocator.free(
                    decode_req.metadata_buffer_index
                )
                decode_req.metadata_buffer_index = -1
            release_kv_cache(decode_req.req, self.tree_cache, is_insert=False)
            self.scheduler.stream_output(
                [decode_req.req], decode_req.req.return_logprob
            )
            failed.append(decode_req)
        return ready, failed

    def _send_preallocated_metadata(self, decode_req, page_size: int) -> None:
        """CPU index preparation and NIXL metadata publication (no allocation)."""

        origin_input_len = len(decode_req.req.origin_input_ids)
        if self.scheduler.enable_hisparse:
            dst_kv_indices = self.req_to_token_pool.req_to_token[
                decode_req.req.req_pool_idx, :origin_input_len
            ]
            kv_indices = dst_kv_indices.cpu().numpy().astype(np.int32)
        else:
            kv_indices_full = self.req_to_token_pool.req_to_token[
                decode_req.req.req_pool_idx
            ][:origin_input_len]
            kv_indices = kv_indices_full.cpu().numpy()

        if isinstance(self.token_to_kv_pool, HybridLinearKVPool):
            state_indices = [
                self.req_to_token_pool.req_index_to_mamba_index_mapping[
                    decode_req.req.req_pool_idx
                ]
                .cpu()
                .numpy()
            ]
        elif isinstance(self.token_to_kv_pool, SWAKVPool):
            seq_len = len(decode_req.req.origin_input_ids)
            window_size = self.scheduler.sliding_window_size
            window_start = max(0, seq_len - window_size)
            window_start = (window_start // page_size) * page_size
            window_kv_indices_full = self.req_to_token_pool.req_to_token[
                decode_req.req.req_pool_idx, window_start:seq_len
            ]
            window_kv_indices_swa = (
                self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                    window_kv_indices_full
                )
            )
            state_indices = kv_to_page_indices(
                window_kv_indices_swa.cpu().numpy(), page_size
            )
        elif isinstance(self.token_to_kv_pool, NSATokenToKVPool):
            seq_len = len(decode_req.req.origin_input_ids)
            kv_indices_full = self.req_to_token_pool.req_to_token[
                decode_req.req.req_pool_idx, :seq_len
            ]
            state_indices = kv_to_page_indices(
                kv_indices_full.cpu().numpy(), self.token_to_kv_pool.page_size
            )
        else:
            state_indices = None

        page_indices = kv_to_page_indices(kv_indices, page_size)
        decode_req.kv_receiver.send_metadata(
            page_indices, decode_req.metadata_buffer_index, state_indices
        )
        ready_path = getattr(decode_req, "_async_p_ready_path", None)
        if ready_path:
            try:
                os.unlink(ready_path)
            except FileNotFoundError:
                pass
        if self.transfer_queue.enable_staging and decode_req.kv_receiver.require_staging:
            self.transfer_queue.staging_handler.register_decode_req(
                decode_req.req.bootstrap_room, decode_req
            )
        decode_req.req.time_stats.set_decode_transfer_queue_entry_time()

    def _init_kv_manager(self) -> CommonKVManager:
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()

        attn_tp_size = get_attention_tp_size()
        kv_args.engine_rank = self.tp_rank % (attn_tp_size)

        kv_args.pp_rank = self.pp_rank
        kv_args.system_dp_rank = self.scheduler.dp_rank
        if self.scheduler.enable_hisparse:
            # Direct-to-host: register host pool pointers so P writes to D's host memory
            host_pool = self.scheduler.hisparse_coordinator.mem_pool_host
            kv_data_ptrs, kv_data_lens, kv_item_lens = (
                host_pool.get_contiguous_buf_infos()
            )
        else:
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
        # HiSparse Host pool has page_size=1; use it when hisparse is enabled
        kv_args.page_size = (
            1 if self.scheduler.enable_hisparse else self.token_to_kv_pool.page_size
        )

        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
            self.metadata_buffers.get_buf_infos()
        )

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

        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.gpu_id
        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(
            kv_args,
            DisaggregationMode.DECODE,
            self.scheduler.server_args,
            self.is_mla_backend,
        )
        # Staging buffer setup (only when heterogeneous TP staging is enabled)
        if self.enable_staging and not self.is_mla_backend:
            kv_pool_for_heads = self.token_to_kv_pool
            if hasattr(kv_pool_for_heads, "full_kv_pool"):
                kv_pool_for_heads = kv_pool_for_heads.full_kv_pool
            per_rank_kv_heads = getattr(kv_pool_for_heads, "head_num", 0)
            if per_rank_kv_heads > 0:
                kv_args.kv_head_num = per_rank_kv_heads
                kv_args.total_kv_head_num = per_rank_kv_heads * attn_tp_size
            if hasattr(kv_manager, "set_kv_buffer_tensors"):
                kv_pool = kv_pool_for_heads
                if hasattr(kv_pool, "k_buffer") and hasattr(kv_pool, "v_buffer"):
                    kv_manager.set_kv_buffer_tensors(
                        kv_pool.k_buffer, kv_pool.v_buffer, kv_pool.page_size
                    )
        return kv_manager

    def add(self, req: Req, is_retracted: bool = False) -> None:
        """Add a request to the pending queue."""
        if self._check_if_req_exceed_kv_capacity(req):
            return

        if is_retracted:
            req.retraction_mb_id = None
            self.retracted_queue.append(req)
        else:
            decode_req = self._create_receiver_and_enqueue(req)

            # NOTE: fake transfer does not need to resolve prefill dp rank in the pending queue
            if _is_fake_transfer(req, self.scheduler.server_args):
                decode_req.kv_receiver.init(0)
                return

            # Fast path: cache-only lookup, no network calls
            prefill_dp_rank = self._resolve_prefill_dp_rank(req)
            if prefill_dp_rank is not None:
                decode_req.kv_receiver.init(prefill_dp_rank)
                return

            with self._async_pending_lock:
                self.pending_reqs.append(decode_req)

    def _resolve_prefill_dp_rank(self, req: Req) -> Optional[int]:
        if req.disagg_prefill_dp_rank is not None:
            return req.disagg_prefill_dp_rank

        prefill_info = self.kv_manager.prefill_info_table.get(_bootstrap_addr(req))
        if prefill_info is None:
            return None

        if prefill_info.dp_size == 1:
            return 0

        if prefill_info.follow_bootstrap_room:
            return req.bootstrap_room % prefill_info.dp_size

        return None

    def _create_receiver_and_enqueue(self, req: Req) -> DecodeRequest:
        backend = (
            TransferBackend.FAKE
            if _is_fake_transfer(req, self.scheduler.server_args)
            else self.transfer_backend
        )
        kv_receiver_class = get_kv_class(backend, KVClassType.RECEIVER)

        kv_receiver = kv_receiver_class(
            mgr=self.kv_manager,
            bootstrap_addr=_bootstrap_addr(req),
            bootstrap_room=req.bootstrap_room,
        )

        decode_req = DecodeRequest(req=req, kv_receiver=kv_receiver)
        self.queue.append(decode_req)
        return decode_req

    def _check_if_req_exceed_kv_capacity(self, req: Req) -> bool:
        if len(req.origin_input_ids) > self.max_total_num_tokens:
            message = f"Request {req.rid} exceeds the maximum number of tokens: {len(req.origin_input_ids)} > {self.max_total_num_tokens}"
            logger.error(message)
            prepare_abort(req, message, status_code=HTTPStatus.BAD_REQUEST)
            self.scheduler.stream_output([req], req.return_logprob)
            return True
        return False

    def extend(self, reqs: List[Req], is_retracted: bool = False) -> None:
        """Add a request to the pending queue."""
        for req in reqs:
            self.add(req, is_retracted=is_retracted)

    def resume_retracted_reqs(
        self, rids_to_check: Optional[List[str]] = None
    ) -> List[Req]:
        # TODO refactor the scheduling part, reuse with the unified engine logic as much as possible

        # allocate memory
        resumed_reqs = []
        indices_to_remove = set()
        allocatable_tokens = self._allocatable_tokens(count_retracted=False)

        for i, req in enumerate(self.retracted_queue):
            if rids_to_check is not None and req.rid not in rids_to_check:
                continue

            if self.req_to_token_pool.available_size() <= 0:
                break

            required_tokens_for_request = (
                len(req.origin_input_ids)
                + len(req.output_ids)
                + self.num_reserved_decode_tokens
            )
            if required_tokens_for_request > allocatable_tokens:
                break

            resumed_reqs.append(req)
            indices_to_remove.add(i)
            req.is_retracted = False
            self._pre_alloc(req)
            allocatable_tokens -= required_tokens_for_request

            # load from cpu, release the cpu copy
            req.load_kv_cache(self.req_to_token_pool, self.token_to_kv_pool_allocator)

        self.retracted_queue = [
            entry
            for i, entry in enumerate(self.retracted_queue)
            if i not in indices_to_remove
        ]

        return resumed_reqs

    def _update_handshake_waiters(
        self, rids_to_check: Optional[List[str]] = None
    ) -> None:
        if not self.queue:
            return

        if all(decode_req.waiting_for_input for decode_req in self.queue):
            return

        queue_snapshot = list(self.queue)
        if self._async_progress_enabled:
            # Agentic V1 is TP=1.  Avoid a Gloo collective in a background
            # thread and keep the model/collective submission thread isolated.
            polls = [int(req.kv_receiver.poll()) for req in queue_snapshot]
        else:
            polls = poll_and_all_reduce(
                [decode_req.kv_receiver for decode_req in queue_snapshot],
                self.gloo_group,
            )

        for i, (decode_req, poll) in enumerate(zip(queue_snapshot, polls)):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            if poll == KVPoll.Bootstrapping:
                pass
            elif poll == KVPoll.WaitingForInput:
                decode_req.waiting_for_input = True
                decode_req.req.time_stats.set_bootstrap_done_time()
            elif poll == KVPoll.Failed:
                error_message = f"Decode handshake failed for request rank={self.tp_rank} {decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                try:
                    decode_req.kv_receiver.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.error(error_message)
                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                if self.scheduler.enable_metrics:
                    self.scheduler.metrics_collector.increment_bootstrap_failed_reqs()
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

    def _ensure_prefill_info(
        self, addr_to_reqs: Dict[str, List[DecodeRequest]]
    ) -> Tuple[Dict[str, List[DecodeRequest]], List[DecodeRequest]]:
        """Non-blocking ensure parallel info for each addr.
        Returns (ready_addrs, remaining_reqs)."""
        ready: Dict[str, List[DecodeRequest]] = {}
        remaining: List[DecodeRequest] = []

        now = time.monotonic()
        for bootstrap_addr, reqs in addr_to_reqs.items():
            last_attempt = self._ensure_last_attempt_time.get(bootstrap_addr)
            if last_attempt is not None and (
                now - last_attempt < self._ensure_retry_interval
            ):
                remaining.extend(reqs)
                continue

            self._ensure_last_attempt_time[bootstrap_addr] = now

            if self.kv_manager.try_ensure_parallel_info(bootstrap_addr):
                if bootstrap_addr in self._ensure_retry_count:
                    del self._ensure_retry_count[bootstrap_addr]
                if bootstrap_addr in self._ensure_last_attempt_time:
                    del self._ensure_last_attempt_time[bootstrap_addr]
                ready[bootstrap_addr] = reqs
                continue

            count = self._ensure_retry_count.get(bootstrap_addr, 0) + 1
            self._ensure_retry_count[bootstrap_addr] = count

            if count >= self._max_ensure_retries:
                error_msg = f"Could not fetch prefill parallel info from {bootstrap_addr} after {count} attempts"
                logger.error(error_msg)
                for decode_req in reqs:
                    decode_req.kv_receiver.abort()
                del self._ensure_retry_count[bootstrap_addr]
                del self._ensure_last_attempt_time[bootstrap_addr]
            else:
                remaining.extend(reqs)

        return ready, remaining

    def _resolve_pending_reqs(self) -> None:
        """Batch-resolve prefill_dp_ranks for pending requests and initialize receivers."""
        with self._async_pending_lock:
            if not self.pending_reqs:
                return
            # New arrivals can append while DNS/RPC/NIXL initialization runs;
            # detach this batch so no scheduler request is lost on reassignment.
            pending_reqs = self.pending_reqs
            self.pending_reqs = []

        # Group pending requests by bootstrap_addr
        addr_to_reqs: Dict[str, List[DecodeRequest]] = {}
        for decode_req in pending_reqs:
            addr = _bootstrap_addr(decode_req.req)
            addr_to_reqs.setdefault(addr, []).append(decode_req)

        # Pass 1: ensure parallel info for each addr
        ready_addrs, remaining = self._ensure_prefill_info(addr_to_reqs)

        resolved: List[Tuple[DecodeRequest, int]] = []
        for bootstrap_addr, decode_reqs in ready_addrs.items():
            need_query: List[DecodeRequest] = []
            for decode_req in decode_reqs:
                prefill_dp_rank = self._resolve_prefill_dp_rank(decode_req.req)
                if prefill_dp_rank is not None:
                    resolved.append((decode_req, prefill_dp_rank))
                else:
                    need_query.append(decode_req)

            # Pass 2: resolve dp rank for addrs whose info is available
            if need_query:
                rooms = [decode_req.req.bootstrap_room for decode_req in need_query]
                room_to_rank = CommonKVReceiver.query_prefill_dp_ranks(
                    bootstrap_addr, rooms
                )
                for decode_req in need_query:
                    prefill_dp_rank = room_to_rank.get(
                        str(decode_req.req.bootstrap_room)
                    )
                    if prefill_dp_rank is not None:
                        resolved.append((decode_req, int(prefill_dp_rank)))
                    else:
                        remaining.append(decode_req)

        with self._async_pending_lock:
            self.pending_reqs = remaining + self.pending_reqs

        for decode_req, prefill_dp_rank in resolved:
            decode_req.kv_receiver.init(prefill_dp_rank)

    def pop_preallocated(
        self, rids_to_check: Optional[List[str]] = None
    ) -> Tuple[List[DecodeRequest], List[DecodeRequest]]:
        """Pop the preallocated requests from the pending queue (FIFO)."""
        if not self._async_progress_enabled:
            self._resolve_pending_reqs()
            self._update_handshake_waiters(rids_to_check)

        if self._async_progress_enabled:
            preallocated_reqs, failed_reqs = self._drain_background_metadata()
        else:
            failed_reqs = []
            preallocated_reqs = []
        indices_to_remove = set()

        # We need to make sure that the sum of inflight tokens and allocatable tokens is greater than maximum input+output length of each inflight request
        # Otherwise it is possible for one request running decode out of memory, while all other requests are in the transfer queue that cannot be retracted.
        retractable_tokens = sum(
            len(r.origin_input_ids) + len(r.output_ids)
            for r in self.scheduler.running_batch.reqs
        )
        allocatable_tokens = self._allocatable_tokens(
            retractable_tokens=retractable_tokens, count_retracted=True
        )
        blocked_reason = None
        blocked_req = None
        blocked_required_tokens = 0
        # First, remove all failed requests from the queue
        for i, decode_req in enumerate(self.queue):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue
            if isinstance(decode_req.req.finished_reason, FINISH_ABORT):
                self.scheduler.stream_output(
                    [decode_req.req], decode_req.req.return_logprob
                )
                failed_reqs.append(decode_req)
                indices_to_remove.add(i)

        # Then, preallocate the remaining requests if possible
        for i, decode_req in enumerate(self.queue):
            if (
                self.max_transfer_inflight > 0
                and len(self.transfer_queue.queue)
                + len(preallocated_reqs)
                + self._async_metadata_pending_count
                >= self.max_transfer_inflight
            ):
                break
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            if i in indices_to_remove:
                continue

            if not decode_req.waiting_for_input:
                blocked_reason = blocked_reason or "handshake"
                blocked_req = blocked_req or decode_req
                continue

            require_p_ready = bool(
                self.p_ready_dir
                and decode_req.req.bootstrap_host != FAKE_BOOTSTRAP_HOST
            )
            if require_p_ready:
                if self._async_progress_enabled:
                    if not getattr(decode_req, "_async_p_ready", False):
                        blocked_reason = blocked_reason or "p_ready"
                        blocked_req = blocked_req or decode_req
                        continue
                    ready_path = getattr(decode_req, "_async_p_ready_path", None)
                else:
                    ready_path = os.path.join(
                        self.p_ready_dir,
                        f"{decode_req.req.bootstrap_room}.ready",
                    )
                    if not os.path.exists(ready_path):
                        continue

            if self.req_to_token_pool.available_size() <= 0:
                blocked_reason = "request_slots"
                blocked_req = decode_req
                break

            if self.req_to_metadata_buffer_idx_allocator.available_size() <= 0:
                blocked_reason = "metadata_slots"
                blocked_req = decode_req
                break

            # Memory estimation: don't add if the projected memory cannot be met
            # TODO: add new_token ratio
            origin_input_len = len(decode_req.req.origin_input_ids)
            required_tokens_for_request = (
                origin_input_len + self.num_reserved_decode_tokens
            )

            if (
                max(
                    required_tokens_for_request,
                    origin_input_len
                    + min(
                        decode_req.req.sampling_params.max_new_tokens,
                        CLIP_MAX_NEW_TOKEN,
                    )
                    - retractable_tokens,
                )
                > allocatable_tokens
            ):
                blocked_reason = "decode_safety"
                blocked_req = decode_req
                blocked_required_tokens = max(
                    required_tokens_for_request,
                    origin_input_len
                    + min(
                        decode_req.req.sampling_params.max_new_tokens,
                        CLIP_MAX_NEW_TOKEN,
                    )
                    - retractable_tokens,
                )
                break
            if required_tokens_for_request > allocatable_tokens:
                blocked_reason = "prompt_kv"
                blocked_req = decode_req
                blocked_required_tokens = required_tokens_for_request
                break

            allocatable_tokens -= required_tokens_for_request
            dst_kv_indices = self._pre_alloc(decode_req.req)

            if self._async_progress_enabled:
                decode_req.metadata_buffer_index = (
                    self.req_to_metadata_buffer_idx_allocator.alloc()
                )
                assert decode_req.metadata_buffer_index is not None
                page_size = (
                    1
                    if self.scheduler.enable_hisparse
                    else self.token_to_kv_pool_allocator.page_size
                )
                with self._async_metadata_count_lock:
                    self._async_metadata_pending_count += 1
                self._async_metadata_work.put((decode_req, page_size))
                indices_to_remove.add(i)
                continue

            origin_input_len = len(decode_req.req.origin_input_ids)
            if self.scheduler.enable_hisparse:
                # Must cast to int32 for ZMQ serialization — from_zmq reads np.int32.
                kv_indices = (
                    dst_kv_indices[:origin_input_len].cpu().numpy().astype(np.int32)
                )
                page_size = 1  # host pool page_size
            else:
                kv_indices_full = self.req_to_token_pool.req_to_token[
                    decode_req.req.req_pool_idx
                ][:origin_input_len]
                kv_indices = kv_indices_full.cpu().numpy()
                page_size = self.token_to_kv_pool_allocator.page_size

            # Prepare extra pool indices for hybrid models
            if isinstance(self.token_to_kv_pool, HybridLinearKVPool):
                # Mamba hybrid model: single mamba state index
                state_indices = [
                    self.req_to_token_pool.req_index_to_mamba_index_mapping[
                        decode_req.req.req_pool_idx
                    ]
                    .cpu()
                    .numpy()
                ]
            elif isinstance(self.token_to_kv_pool, SWAKVPool):
                # SWA hybrid model: send decode-side SWA window indices
                seq_len = len(decode_req.req.origin_input_ids)
                window_size = self.scheduler.sliding_window_size

                window_start = max(0, seq_len - window_size)
                window_start = (window_start // page_size) * page_size
                window_kv_indices_full = self.req_to_token_pool.req_to_token[
                    decode_req.req.req_pool_idx, window_start:seq_len
                ]

                # Translate to SWA pool indices
                window_kv_indices_swa = (
                    self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                        window_kv_indices_full
                    )
                )
                state_indices = window_kv_indices_swa.cpu().numpy()
                state_indices = kv_to_page_indices(state_indices, page_size)
            elif isinstance(self.token_to_kv_pool, NSATokenToKVPool):
                seq_len = len(decode_req.req.origin_input_ids)
                kv_indices_full = self.req_to_token_pool.req_to_token[
                    decode_req.req.req_pool_idx, :seq_len
                ]
                state_indices = kv_indices_full.cpu().numpy()
                # Indexer lives on device pool; always use device page_size
                device_page_size = self.token_to_kv_pool.page_size
                state_indices = kv_to_page_indices(state_indices, device_page_size)
            else:
                state_indices = None

            decode_req.metadata_buffer_index = (
                self.req_to_metadata_buffer_idx_allocator.alloc()
            )
            assert decode_req.metadata_buffer_index is not None
            page_indices = kv_to_page_indices(kv_indices, page_size)
            decode_req.kv_receiver.send_metadata(
                page_indices, decode_req.metadata_buffer_index, state_indices
            )
            if require_p_ready:
                try:
                    os.unlink(ready_path)
                except FileNotFoundError:
                    pass
            if (
                self.transfer_queue.enable_staging
                and decode_req.kv_receiver.require_staging
            ):
                self.transfer_queue.staging_handler.register_decode_req(
                    decode_req.req.bootstrap_room, decode_req
                )
            preallocated_reqs.append(decode_req)
            indices_to_remove.add(i)
            decode_req.req.time_stats.set_decode_transfer_queue_entry_time()

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        if self.queue and not preallocated_reqs:
            now = time.monotonic()
            if now >= self._prealloc_blocked_next_log_at:
                self._prealloc_blocked_next_log_at = now + 5.0
                req = blocked_req or self.queue[0]
                logger.info(
                    "PD_PREALLOC_BLOCKED reason=%s queue=%d rid=%s room=%s "
                    "prompt_tokens=%d required_tokens=%d allocatable_tokens=%d "
                    "free_kv_tokens=%d request_slots=%d metadata_slots=%d "
                    "running=%d waiting=%d transfer=%d async_metadata=%d "
                    "handshake=%s p_ready=%s ready_file=%s",
                    blocked_reason or "unknown",
                    len(self.queue),
                    req.req.rid,
                    req.req.bootstrap_room,
                    len(req.req.origin_input_ids),
                    blocked_required_tokens,
                    allocatable_tokens,
                    self.token_to_kv_pool_allocator.available_size(),
                    self.req_to_token_pool.available_size(),
                    self.req_to_metadata_buffer_idx_allocator.available_size(),
                    len(self.scheduler.running_batch.reqs),
                    len(self.scheduler.waiting_queue),
                    len(self.transfer_queue.queue),
                    self._async_metadata_pending_count,
                    req.waiting_for_input,
                    getattr(req, "_async_p_ready", False),
                    os.path.exists(
                        os.path.join(
                            self.p_ready_dir,
                            f"{req.req.bootstrap_room}.ready",
                        )
                    )
                    if self.p_ready_dir
                    else False,
                )

        return preallocated_reqs, failed_reqs

    @property
    def num_tokens_pre_allocated(self):
        return sum(
            len(decode_req.req.fill_ids) for decode_req in self.transfer_queue.queue
        )

    def _allocatable_tokens(
        self, retractable_tokens: Optional[int] = None, count_retracted: bool = True
    ) -> int:
        need_space_for_single_req = (
            max(
                [
                    min(x.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKEN)
                    + len(x.origin_input_ids)
                    - retractable_tokens
                    for x in self.scheduler.running_batch.reqs
                ]
            )
            if retractable_tokens is not None
            and len(self.scheduler.running_batch.reqs) > 0
            else 0
        )
        available_size = self.token_to_kv_pool_allocator.available_size()
        # Upstream PD Decode normally uses ChunkCache, so only allocator-free
        # pages matter.  Agentic PD can enable Radix on D; completed request
        # suffixes then remain as *evictable* LRU pages.  Treat those pages as
        # admission capacity, otherwise preallocation stalls while most of the
        # pool is immediately reclaimable.
        evictable_size = self.tree_cache.evictable_size()
        if isinstance(evictable_size, int):
            available_size += evictable_size
        allocatable_tokens = available_size - max(
            # preserve some space for future decode
            self.num_reserved_decode_tokens
            * (
                len(self.scheduler.running_batch.reqs)
                + len(self.transfer_queue.queue)
                + len(self.scheduler.waiting_queue)
            ),
            # make sure each request can finish if reach max_tokens with all other requests retracted
            need_space_for_single_req,
        )

        # Note: if the last prebuilt extend just finishes, and we enter `pop_preallocated` immediately in the next iteration
        #       the extend batch is not in any queue, so we need to explicitly add the tokens slots here
        if (
            self.scheduler.last_batch
            and self.scheduler.last_batch.forward_mode.is_prebuilt()
        ):
            allocatable_tokens -= self.num_reserved_decode_tokens * len(
                self.scheduler.last_batch.reqs
            )

        if count_retracted:
            allocatable_tokens -= sum(
                [
                    len(req.origin_input_ids)
                    + len(req.output_ids)
                    + self.num_reserved_decode_tokens
                    for req in self.retracted_queue
                ]
            )
        return allocatable_tokens

    def _ensure_prealloc_kv_available(self, num_tokens: int) -> int:
        """Reclaim only the evictable D Radix pages needed by one import."""

        available = self.token_to_kv_pool_allocator.available_size()
        deficit = max(0, int(num_tokens) - available)
        if deficit == 0 or self.tree_cache.is_chunk_cache():
            return 0
        result = self.tree_cache.evict(EvictParams(num_tokens=deficit))
        return int(result.num_tokens_evicted)

    def _pre_alloc(self, req: Req) -> torch.Tensor:
        """Pre-allocate the memory for req_to_token and token_kv_pool"""
        req_pool_indices = self.req_to_token_pool.alloc([req])

        assert (
            req_pool_indices is not None
        ), "req_pool_indices is full! There is a bug in memory estimation."

        # Alloc all tokens for the prebuilt req (except for the reserved input token for decoding)
        fill_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
        req.kv_allocated_len = fill_len
        req.kv_committed_len = fill_len

        # A paged allocation can require one extra partially filled page.  D's
        # shared/protected prefixes stay locked; native Radix LRU evicts only
        # unreferenced leaves, preserving prefix sharing for active requests.
        self._ensure_prealloc_kv_available(
            fill_len + max(1, self.token_to_kv_pool_allocator.page_size)
        )

        if self.scheduler.enable_hisparse:
            # Direct-to-host path: only allocate logical indices (no hisparse
            # device indices) and allocate host indices for RDMA destination.
            coordinator = self.scheduler.hisparse_coordinator
            device = self.token_to_kv_pool_allocator.device
            kv_loc = self.token_to_kv_pool_allocator.alloc_logical_only(
                prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
                prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
                seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
                seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
                last_loc=torch.tensor([-1], dtype=torch.int64, device=device),
                extend_num_tokens=fill_len,
            )
            # Allocate host indices for the RDMA transfer target
            host_indices = coordinator.mem_pool_host.alloc(fill_len)
            if host_indices is None:
                raise RuntimeError(
                    f"HiSparse host mem pool alloc failed for {fill_len} tokens "
                    f"in _pre_alloc (req {req.rid})"
                )
            host_indices = host_indices.to(device=coordinator.device)
            coordinator.req_to_host_pool[req.req_pool_idx, :fill_len] = host_indices
        elif self.token_to_kv_pool_allocator.page_size == 1:
            kv_loc = self.token_to_kv_pool_allocator.alloc(fill_len)
        else:
            device = self.token_to_kv_pool_allocator.device
            kv_loc = self.token_to_kv_pool_allocator.alloc_extend(
                prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
                prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
                seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
                seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
                last_loc=torch.tensor([-1], dtype=torch.int64, device=device),
                extend_num_tokens=fill_len,
            )

        assert (
            kv_loc is not None
        ), "KV cache is full! There is a bug in memory estimation."

        self.req_to_token_pool.write((req.req_pool_idx, slice(0, len(kv_loc))), kv_loc)

        # populate metadata
        req.fill_ids = req.origin_input_ids + req.output_ids
        req.set_extend_input_len(len(req.fill_ids))

        # Return the transfer destination indices:
        if self.scheduler.enable_hisparse:
            return host_indices
        return kv_loc


class DecodeTransferQueue:
    """
    Store the requests that is polling kv
    """

    def __init__(
        self,
        gloo_group: ProcessGroup,
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        tp_rank: int,
        metadata_buffers: MetadataBuffers,
        scheduler: Scheduler,
        tree_cache: BasePrefixCache,
    ):
        self.queue: List[DecodeRequest] = []
        self.gloo_group = gloo_group
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.tp_rank = tp_rank
        self.metadata_buffers = metadata_buffers
        self.scheduler = scheduler
        self.tree_cache = tree_cache
        self.spec_algorithm = scheduler.spec_algorithm
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()
        self.staging_handler = None
        self._async_progress_enabled = False
        self._async_poll_lock = threading.Lock()

    def enable_async_progress(self) -> None:
        self._async_progress_enabled = True

    def background_progress(self) -> None:
        """Poll P->D DMA off the scheduler; commits remain scheduler-owned."""

        if not self._async_progress_enabled:
            return
        with self._async_poll_lock:
            queue_snapshot = []
            for decode_req in list(self.queue):
                if (
                    getattr(decode_req, "_async_transfer_poll", None) is None
                    and not getattr(decode_req, "_async_transfer_poll_inflight", False)
                    and not getattr(decode_req, "_async_transfer_poll_claimed", False)
                ):
                    decode_req._async_transfer_poll_inflight = True
                    queue_snapshot.append(decode_req)
        if not queue_snapshot:
            return
        try:
            if self.enable_staging:
                # Async agentic V1 is TP=1, so perform the staging advancement and
                # local receiver polls without entering a background collective.
                for decode_req in queue_snapshot:
                    if decode_req.kv_receiver.require_staging and not self.staging_handler.is_done(
                        decode_req
                    ):
                        self.staging_handler.advance_scatter(decode_req)
                polls = [int(req.kv_receiver.poll()) for req in queue_snapshot]
                for i, decode_req in enumerate(queue_snapshot):
                    if polls[i] == int(KVPoll.Success) and (
                        decode_req.kv_receiver.require_staging
                        and not self.staging_handler.is_done(decode_req)
                    ):
                        polls[i] = int(KVPoll.Transferring)
            else:
                receivers = [req.kv_receiver for req in queue_snapshot]
                poll_many = getattr(type(receivers[0]), "poll_many", None)
                if poll_many is None:
                    polls = [int(receiver.poll()) for receiver in receivers]
                else:
                    polls = [int(poll) for poll in poll_many(receivers)]
        except Exception:
            with self._async_poll_lock:
                for decode_req in queue_snapshot:
                    decode_req._async_transfer_poll_inflight = False
            raise
        with self._async_poll_lock:
            for decode_req, poll in zip(queue_snapshot, polls):
                decode_req._async_transfer_poll = int(poll)
                decode_req._async_transfer_poll_inflight = False

    def add(self, decode_req: DecodeRequest) -> None:
        self.queue.append(decode_req)

    def extend(self, decode_reqs: List[DecodeRequest]) -> None:
        self.queue.extend(decode_reqs)
        if self.enable_staging:
            for dr in decode_reqs:
                if dr.kv_receiver.require_staging:
                    self.staging_handler.register_decode_req(dr.req.bootstrap_room, dr)

    def _commit_transfer_to_req(self, decode_req: DecodeRequest) -> bool:
        """
        Returns:
            True if the request should be removed from the queue (success or corruption)
            False if metadata not ready yet (keep in queue for next poll)
        """
        idx = decode_req.metadata_buffer_index
        (
            output_id,
            cached_tokens,
            output_token_logprobs_val,
            output_token_logprobs_idx,
            output_top_logprobs_val,
            output_top_logprobs_idx,
            output_topk_p,
            output_topk_index,
            output_hidden_states,
            output_bootstrap_room,
        ) = self.metadata_buffers.get_buf(idx)

        # Validate bootstrap_room to detect context corruption
        actual_room = output_bootstrap_room[0].item()
        expected_room = (
            decode_req.req.bootstrap_room
            if decode_req.req.bootstrap_room is not None
            else 0
        )

        if _is_fake_transfer(decode_req.req, self.scheduler.server_args):
            pass
        elif actual_room == 0:
            # Case 1: Metadata not ready yet (actual_room == 0)
            # Keep request in queue and wait for next poll
            return False
        elif actual_room != expected_room:
            # Case 2: Real corruption detected (mismatch)
            # Abort the request and remove from the queue
            error_msg = (
                f"Context corruption detected: Request {decode_req.req.rid} "
                f"(bootstrap_room={expected_room}) received metadata from "
                f"bootstrap_room={actual_room}. "
                f"Metadata buffer index: {idx}. "
                f"This indicates metadata buffer index collision."
            )
            logger.error(error_msg)
            prepare_abort(
                decode_req.req,
                "Metadata corruption detected - bootstrap_room mismatch",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            decode_req.kv_receiver.clear()
            decode_req.kv_receiver = None
            return True

        # Case 3: Success - commit the transfer
        decode_req.req.output_ids.append(output_id[0].item())
        decode_req.req.cached_tokens = cached_tokens[0].item()
        decode_req.req.cached_tokens_device = cached_tokens[1].item()
        decode_req.req.cached_tokens_host = cached_tokens[2].item()
        decode_req.req.cached_tokens_storage = cached_tokens[3].item()
        if not self.spec_algorithm.is_none():
            decode_req.req.output_topk_p = output_topk_p
            decode_req.req.output_topk_index = output_topk_index
            decode_req.req.hidden_states_tensor = output_hidden_states

        if decode_req.req.return_logprob:
            decode_req.req.output_token_logprobs_val.append(
                output_token_logprobs_val[0].item()
            )
            decode_req.req.output_token_logprobs_idx.append(
                output_token_logprobs_idx[0].item()
            )
            decode_req.req.output_top_logprobs_val.append(
                output_top_logprobs_val[: decode_req.req.top_logprobs_num].tolist()
            )
            decode_req.req.output_top_logprobs_idx.append(
                output_top_logprobs_idx[: decode_req.req.top_logprobs_num].tolist()
            )

        decode_req.kv_receiver.clear()
        decode_req.kv_receiver = None
        decode_req.req.time_stats.set_wait_queue_entry_time()
        return True

    def _poll_with_staging(self) -> list:
        return poll_and_all_reduce_with_staging(
            self.queue, self.staging_handler, self.gloo_group
        )

    def _init_staging_handler(self, kv_manager):
        """Create staging handler from kv_manager. Must be called exactly once."""
        from sglang.srt.disaggregation.common.staging_handler import (
            DecodeStagingHandler,
        )

        self.staging_handler = DecodeStagingHandler.create(
            kv_manager, self.scheduler, self.tp_rank
        )
        kv_manager._staging_handler = self.staging_handler

    def pop_transferred(self, rids_to_check: Optional[List[str]] = None) -> List[Req]:
        if not self.queue:
            return []

        if self._async_progress_enabled:
            # No transport call is allowed here.  Missing cached status simply
            # means the background worker has not completed its next poll.
            with self._async_poll_lock:
                polls = []
                for dr in self.queue:
                    poll = getattr(
                        dr, "_async_transfer_poll", int(KVPoll.Transferring)
                    )
                    if getattr(dr, "_async_transfer_poll", None) is not None:
                        dr._async_transfer_poll_claimed = True
                    polls.append(poll)
        elif self.enable_staging:
            polls = self._poll_with_staging()
        else:
            polls = poll_and_all_reduce(
                [dr.kv_receiver for dr in self.queue], self.gloo_group
            )

        transferred_reqs = []
        indices_to_remove = set()
        for i, (decode_req, poll) in enumerate(zip(self.queue, polls)):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            if poll == KVPoll.Failed:
                error_message = f"Decode transfer failed for request rank={self.tp_rank} {decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                try:
                    decode_req.kv_receiver.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.error(error_message)
                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                self.scheduler.stream_output(
                    [decode_req.req], decode_req.req.return_logprob
                )
                if self.scheduler.enable_hisparse:
                    self.scheduler.hisparse_coordinator.request_finished(decode_req.req)
                # release pre-allocated kv cache, but don't insert into the tree since it's failed
                release_kv_cache(decode_req.req, self.tree_cache, is_insert=False)
                indices_to_remove.add(i)
                if self.scheduler.enable_metrics:
                    self.scheduler.metrics_collector.increment_transfer_failed_reqs()
                continue
            elif poll == KVPoll.Success:
                should_remove = self._commit_transfer_to_req(decode_req)
                if should_remove:
                    indices_to_remove.add(i)
                    # Check if request was aborted due to corruption
                    if isinstance(decode_req.req.finished_reason, FINISH_ABORT):
                        self.scheduler.stream_output(
                            [decode_req.req], decode_req.req.return_logprob
                        )
                        if self.scheduler.enable_hisparse:
                            self.scheduler.hisparse_coordinator.request_finished(
                                decode_req.req
                            )
                        release_kv_cache(
                            decode_req.req, self.tree_cache, is_insert=False
                        )
                        if self.scheduler.enable_metrics:
                            self.scheduler.metrics_collector.increment_transfer_failed_reqs()
                    else:
                        transferred_reqs.append(decode_req.req)
            elif poll in [
                KVPoll.Bootstrapping,
                KVPoll.WaitingForInput,
                KVPoll.Transferring,
            ]:
                pass
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

        for i in indices_to_remove:
            if self.enable_staging and self.staging_handler.is_staging_room(
                self.queue[i].req.bootstrap_room
            ):
                self.staging_handler.unregister_decode_req(
                    self.queue[i].req.bootstrap_room
                )
            idx = self.queue[i].metadata_buffer_index
            assert idx != -1
            self.req_to_metadata_buffer_idx_allocator.free(idx)

        if self._async_progress_enabled:
            with self._async_poll_lock:
                for i, decode_req in enumerate(self.queue):
                    if i in indices_to_remove:
                        continue
                    if hasattr(decode_req, "_async_transfer_poll"):
                        delattr(decode_req, "_async_transfer_poll")
                    decode_req._async_transfer_poll_claimed = False

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return transferred_reqs


class SchedulerDisaggregationDecodeMixin:

    @torch.no_grad()
    def event_loop_normal_disagg_decode(self: Scheduler):
        """A normal scheduler loop for decode worker in disaggregation mode."""

        while True:
            # Receive requests
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            # polling and allocating kv cache
            self.process_decode_queue()

            # Get the next batch to run
            batch = self.get_next_disagg_decode_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # When the server is idle, do self-check and re-init some states
                self.self_check_during_idle()

            # Update last_batch
            self.last_batch = batch

    @torch.no_grad()
    def event_loop_overlap_disagg_decode(self: Scheduler):
        self.result_queue = deque()
        self.last_batch: Optional[ScheduleBatch] = None

        while True:
            # Receive requests
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            # polling and allocating kv cache
            self.process_decode_queue()

            # Get the next batch to run
            batch = self.get_next_disagg_decode_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                batch_result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None

            # Process the last batch
            if self.last_batch:
                tmp_batch, tmp_result = self.result_queue.popleft()
                self.process_batch_result(tmp_batch, tmp_result)
            elif batch is None:
                self.self_check_during_idle()

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            self.launch_batch_sample_if_needed(batch_result)

            # Update last_batch
            self.last_batch = batch

    def _run_batch_prebuilt(
        self: Scheduler, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        if batch.inner_idle_batch is not None:
            idle_batch = batch.inner_idle_batch
            # Reset the inner idle batch to avoid reusing it.
            batch.inner_idle_batch = None
            return self.run_batch(idle_batch)

        return GenerationBatchResult()

    def get_next_disagg_decode_batch_to_run(
        self: Scheduler,
    ) -> Optional[ScheduleBatch]:
        """Process prebuilt batch and schedule the next decode batch."""
        # Process pending prebuilt batch: output processing + filter + merge
        new_prebuilt_batch = self.get_new_prebuilt_batch()
        if new_prebuilt_batch:
            assert self.chunked_req is None
            self.process_batch_result_prebuilt(new_prebuilt_batch)
            new_prebuilt_batch.filter_batch()
            if not new_prebuilt_batch.is_empty():
                if self.running_batch.is_empty():
                    self.running_batch = new_prebuilt_batch
                    if self.enable_hisparse:
                        self.running_batch.hisparse_coordinator = (
                            self.hisparse_coordinator
                        )
                else:
                    self.running_batch.merge_batch(new_prebuilt_batch)

        # Schedule decode batch
        if self.running_batch.is_empty():
            ret = None
        else:
            self.running_batch = self.update_running_batch(self.running_batch)
            ret = self.running_batch if not self.running_batch.is_empty() else None

        ret = self.maybe_prepare_mlp_sync_batch(ret)
        if ret:
            set_schedule_time_batch(ret)
        return ret

    def get_new_prebuilt_batch(self: Scheduler) -> Optional[ScheduleBatch]:
        """Create a schedulebatch for fake completed prefill"""
        if self.grammar_manager.has_waiting_grammars():
            ready_grammar_requests = self.grammar_manager.get_ready_grammar_requests()
            for req in ready_grammar_requests:
                self._add_request_to_queue(req)

        if len(self.waiting_queue) == 0:
            return None

        curr_batch_size = self.running_batch.batch_size()

        batch_size = min(self.req_to_token_pool.size, self.max_running_requests)

        num_not_used_batch = batch_size - curr_batch_size

        # pop req from waiting queue
        can_run_list: List[Req] = []
        waiting_queue: List[Req] = []

        for i in range(len(self.waiting_queue)):
            req = self.waiting_queue[i]
            # we can only add at least `num_not_used_batch` new batch to the running queue
            if i < num_not_used_batch:
                can_run_list.append(req)
                prepare_pd_decode_radix_key(self, req)
                req.init_next_round_input(self.tree_cache)
                # P->D still transfers one complete request snapshot into
                # request-private destination pages.  Dedup happens only
                # after DMA success, so transport remains all-or-nothing.
                reconcile_pd_decode_imported_prefix(self, req)
            else:
                waiting_queue.append(req)

        self.waiting_queue = waiting_queue
        if len(can_run_list) == 0:
            return None

        set_time_batch(can_run_list, "set_forward_entry_time")

        # construct a schedule batch with those requests and mark as decode
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )

        # construct fake completed prefill
        new_batch.prepare_for_prebuilt()
        new_batch.process_prebuilt(self.server_args, self.future_map)

        return new_batch

    def process_decode_queue(self: Scheduler):
        if self.server_args.disaggregation_decode_enable_offload_kvcache:
            self.decode_offload_manager.check_offload_progress()
            ready_responses = self.decode_offload_manager.pop_ready_responses()
            if ready_responses:
                for req in ready_responses:
                    req.time_stats.set_completion_time()
                self.stream_output(
                    ready_responses,
                    any(req.return_logprob for req in ready_responses),
                )

        # try to resume retracted requests if there are enough space for another `num_reserved_decode_tokens` decode steps
        resumed_reqs = self.disagg_decode_prealloc_queue.resume_retracted_reqs()
        self.waiting_queue.extend(resumed_reqs)
        if len(self.disagg_decode_prealloc_queue.retracted_queue) > 0:
            # if there are still retracted requests, we do not allocate new requests
            return

        if not hasattr(self, "polling_count"):
            self.polling_count = 0
            self.polling_interval = (
                self.server_args.disaggregation_decode_polling_interval
            )

        self.polling_count = (self.polling_count + 1) % self.polling_interval

        if self.polling_count % self.polling_interval == 0:
            req_conns, _ = self.disagg_decode_prealloc_queue.pop_preallocated()
            self.disagg_decode_transfer_queue.extend(req_conns)
            transferred_reqs = (
                self.disagg_decode_transfer_queue.pop_transferred()
            )  # the requests which kv has arrived
            if self.enable_hisparse:
                for req in transferred_reqs:
                    # Direct-to-host: KV data already in host pool, skip staging
                    self.hisparse_coordinator.admit_request_direct(req)
                self.waiting_queue.extend(transferred_reqs)
            else:
                self.waiting_queue.extend(transferred_reqs)
