# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A scheduler that manages a tensor parallel GPU worker."""

import copy
import dataclasses
import faulthandler
import heapq
import json
import logging
import os
import signal
import sys
import threading
import time
from array import array
from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import partial
from http import HTTPStatus
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

from sglang.srt.utils.common import suppress_noisy_warnings  # isort: skip

suppress_noisy_warnings()

import psutil  # isort: skip
import setproctitle
import torch

from sglang.srt.disaggregation.agentic_workset import (
    AgenticPWorksetLease,
    AgenticPWorksetLeaseBroker,
)
import torch.distributed
from torch.cuda import Stream as CudaStream
from torch.distributed import barrier

from sglang.jit_kernel.ngram_embedding import update_token_table
from sglang.srt.configs.model_config import ModelConfig, ModelImpl
from sglang.srt.constrained.grammar_manager import GrammarManager
from sglang.srt.debug_utils.pr_fix_toggle import maybe_revert_pr_fix
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.disaggregation.agentic_direct_transfer import (
    create_agentic_direct_runtime,
    debug_kv_digest,
    debug_kv_page_digests,
)
from sglang.srt.disaggregation.agentic_hybrid_transfer import (
    debug_mamba_digest,
    state_indices_for_workset,
    submit_reverse_receive,
    validate_agentic_mamba_tracking,
)
from sglang.srt.disaggregation.agentic_early_claim import AgenticEarlyClaimStore
from sglang.srt.disaggregation.agentic_host_staging import (
    AgenticPHostStagingManager,
    SharedHostStagingLedger,
    create_agentic_storage_controller,
    supports_agentic_kv_spill,
)
from sglang.srt.disaggregation.agentic_tp import (
    rank_env_int,
    rank_scoped_arena_directory,
    request_generation_key,
)
from sglang.srt.disaggregation.agentic_tp_control import TPGroupMailbox
from sglang.srt.disaggregation.p2d_host_staging import (
    AgenticPToDHostStagingManager,
)
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    AgenticRequestMetadata,
    RequestGeneration,
    SharedSnapshotEvictionController,
    SnapshotLifecycleError,
    SnapshotNotReadyError,
    SnapshotState,
    page_namespace,
    token_ids_digest,
    unpack_agentic_extra_key,
)
from sglang.srt.disaggregation.encode_receiver import create_mm_receiver
from sglang.srt.disaggregation.prefill import (
    PrefillBootstrapQueue,
    SchedulerDisaggregationPrefillMixin,
    maybe_release_metadata_buffer,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    FAKE_BOOTSTRAP_HOST,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    prepare_abort,
)
from sglang.srt.distributed import get_pp_group, get_world_group
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.attention.mamba.ops import (
    initialize_mamba_selective_state_update_backend,
)
from sglang.srt.layers.dp_attention import (
    compute_dp_attention_world_info,
    get_attention_cp_group,
    get_attention_tp_group,
)
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.lora.lora_drainer import LoRADrainer
from sglang.srt.lora.lora_overlap_loader import LoRAOverlapLoader
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.managers.io_struct import (
    AbortReq,
    ActiveRanksOutput,
    AddExternalCorpusReqInput,
    AddExternalCorpusReqOutput,
    AttachHiCacheStorageReqInput,
    AttachHiCacheStorageReqOutput,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    CheckWeightsReqInput,
    ClearHiCacheReqInput,
    ClearHiCacheReqOutput,
    CloseSessionReqInput,
    ConfigureLoggingReq,
    ContinueGenerationReqInput,
    DestroyWeightsUpdateGroupReqInput,
    DetachHiCacheStorageReqInput,
    DetachHiCacheStorageReqOutput,
    DumperControlReqInput,
    DumperControlReqOutput,
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    ExpertDistributionReqType,
    FlushCacheReqInput,
    FreezeGCReq,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    GetLoadsReqInput,
    GetWeightsByNameReqInput,
    HealthCheckOutput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsSendGroupForRemoteInstanceReqOutput,
    InitWeightsUpdateGroupReqInput,
    ListExternalCorporaReqInput,
    ListExternalCorporaReqOutput,
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterFromTensorsReqOutput,
    LoadLoRAAdapterReqInput,
    LoadLoRAAdapterReqOutput,
    OpenSessionReqInput,
    PauseGenerationReqInput,
    ProfileReq,
    ReleaseMemoryOccupationReqInput,
    RemoveExternalCorpusReqInput,
    RemoveExternalCorpusReqOutput,
    ResumeMemoryOccupationReqInput,
    RpcReqInput,
    RpcReqOutput,
    SendWeightsToRemoteInstanceReqInput,
    SendWeightsToRemoteInstanceReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    ShutdownReq,
    SlowDownReqInput,
    SlowDownReqOutput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UnloadLoRAAdapterReqInput,
    UnloadLoRAAdapterReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.load_snapshot import LoadSnapshot, create_load_snapshot_writer
from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
from sglang.srt.managers.overlap_utils import (
    decide_needs_cpu_seq_lens,
    resolve_forward_inputs,
)
from sglang.srt.managers.prefill_delayer import (
    PrefillDelayer,
    PrefillDelayerSinglePassExecutor,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    MultimodalInputs,
    Req,
    ScheduleBatch,
)
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
    SchedulePolicy,
)
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.dp_attn import SchedulerDPAttnAdapter
from sglang.srt.managers.scheduler_components.flush_wrapper import SchedulerFlushWrapper
from sglang.srt.managers.scheduler_components.idle_sleeper import IdleSleeper
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
    create_scheduler_watchdog,
)
from sglang.srt.managers.scheduler_components.ipc_channels import SchedulerIpcChannels
from sglang.srt.managers.scheduler_components.kv_events_publisher import (
    SchedulerKvEventsPublisher,
)
from sglang.srt.managers.scheduler_components.load_inquirer import SchedulerLoadInquirer
from sglang.srt.managers.scheduler_components.logprob_result_processor import (
    SchedulerLogprobResultProcessor,
)
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    RECORD_STEP_TIME,
    PrefillStats,
    SchedulerMetricsReporter,
)
from sglang.srt.managers.scheduler_components.new_token_ratio_tracker import (
    NewTokenRatioTracker,
)
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    SchedulerPoolStatsObserver,
)
from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)
from sglang.srt.managers.scheduler_components.request_receiver import (
    SchedulerRequestReceiver,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.managers.scheduler_input_blocker import SchedulerInputBlocker
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.managers.scheduler_recv_skipper import SchedulerRecvSkipper
from sglang.srt.managers.utils import (
    EmbeddingBatchResult,
    GenerationBatchResult,
    is_health_check_generate_req,
    validate_input_length,
)
from sglang.srt.mem_cache import kv_cache_builder
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.common import maybe_cache_unfinished_req, release_kv_cache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.model_loader.utils import get_resolved_model_impl
from sglang.srt.multiplex.multiplexing_mixin import SchedulerMultiplexMixin
from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector
from sglang.srt.observability.req_time_stats import (
    set_schedule_time_batch,
    set_time_batch,
)
from sglang.srt.observability.trace import process_tracing_init, trace_set_thread_info
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.platforms import current_platform
from sglang.srt.plugins import load_plugins
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import PortArgs, ServerArgs, get_global_server_args
from sglang.srt.session.session_controller import SessionController
from sglang.srt.speculative.dflash_utils import (
    resolve_dflash_prefill_refill_target,
    should_delay_dflash_prefill_for_batching,
    validate_dflash_request,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    DynamicGradMode,
    configure_gc_logger,
    configure_logger,
    freeze_gc,
    get_available_gpu_memory,
    get_bool_env_var,
    get_int_env_var,
    is_cuda,
    is_mps,
    kill_itself_when_parent_died,
    require_mlp_sync,
    set_gpu_proc_affinity,
    set_random_seed,
    suppress_other_loggers,
)
from sglang.srt.utils.common import is_npu
from sglang.srt.utils.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
)
from sglang.srt.utils.numa_utils import get_numa_node_if_available, numa_bind_to_node
from sglang.srt.utils.nvtx_utils import scheduler_nvtx_method
from sglang.srt.utils.tensor_bridge import use_mlx
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

if is_mps():
    CudaStreamContext = nullcontext
    from sglang.srt.hardware_backend.mlx.scheduler_mixin import SchedulerMlxOverlapMixin
else:
    from torch.cuda import StreamContext as CudaStreamContext

    class SchedulerMlxOverlapMixin:
        pass


logger = logging.getLogger(__name__)


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


# Test retract decode for debugging purposes
TEST_RETRACT = envs.SGLANG_TEST_RETRACT.get()
TEST_RETRACT_INTERVAL = envs.SGLANG_TEST_RETRACT_INTERVAL.get()
TEST_RETRACT_NO_PREFILL_BS = envs.SGLANG_TEST_RETRACT_NO_PREFILL_BS.get()

_is_npu = is_npu()


class Scheduler(
    SchedulerDisaggregationDecodeMixin,
    SchedulerDisaggregationPrefillMixin,
    SchedulerMultiplexMixin,
    SchedulerPPMixin,
    SchedulerDllmMixin,
    SchedulerMlxOverlapMixin,
):
    """A scheduler that manages a tensor parallel GPU worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        dp_rank: Optional[int],
    ):
        self.is_initializing = True
        # init_soft_watchdog starts a daemon thread that reads these on its first tick.
        self.forward_ct: int = 0
        self.cur_batch: Optional[ScheduleBatch] = None
        self.init_soft_watchdog(server_args)

        # Parse args
        self.server_args = server_args
        # Keep explicit aliases for the agentic TP control plane.  Upstream
        # stores these values in ``self.ps``, but ``init_running_status`` is
        # called before any compatibility attribute is otherwise created and
        # the request-generation mailboxes need the rank information there.
        self.tp_rank = tp_rank
        self.tp_size = server_args.tp_size
        self.pp_rank = pp_rank
        self.pp_size = server_args.pp_size
        self.gpu_id = gpu_id
        self.nccl_port = port_args.nccl_port
        self.schedule_policy = server_args.schedule_policy
        self.enable_priority_scheduling = server_args.enable_priority_scheduling
        self.abort_on_priority_when_disabled = (
            server_args.abort_on_priority_when_disabled
        )
        self.schedule_low_priority_values_first = (
            server_args.schedule_low_priority_values_first
        )
        self.priority_scheduling_preemption_threshold = (
            server_args.priority_scheduling_preemption_threshold
        )
        self.enable_lora = server_args.enable_lora
        self.enable_lora_overlap_loading = server_args.enable_lora_overlap_loading
        self.max_loras_per_batch = server_args.max_loras_per_batch
        self.enable_overlap = not server_args.disable_overlap_schedule and not use_mlx()
        self.enable_overlap_mlx = not server_args.disable_overlap_schedule and use_mlx()
        self.enable_pdmux = server_args.enable_pdmux
        self.skip_tokenizer_init = server_args.skip_tokenizer_init
        self.stream_interval = server_args.stream_interval
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.page_size = server_args.page_size
        custom_storage_only = envs.SGLANG_AGENTIC_KV_CUSTOM_STORAGE_ONLY.get()
        if custom_storage_only and server_args.enable_hierarchical_cache:
            raise ValueError(
                "new-method custom storage forbids native --enable-hierarchical-cache"
            )
        if (
            custom_storage_only
            and server_args.disaggregation_decode_enable_offload_kvcache
        ):
            raise ValueError(
                "new-method custom storage forbids native Decode KV offload"
            )
        self.enable_hierarchical_cache = server_args.enable_hierarchical_cache
        self.enable_hicache_storage = server_args.hicache_storage_backend is not None
        self.enable_decode_hicache = (
            server_args.disaggregation_decode_enable_radix_cache
            and self.enable_hierarchical_cache
        )
        self.max_recv_per_poll = envs.SGLANG_SCHEDULER_MAX_RECV_PER_POLL.get()
        self.enable_hisparse = server_args.enable_hisparse
        self.hisparse_coordinator: Optional[HiSparseCoordinator] = None

        # Set by the ShutdownReq handler to break the event loop for graceful shutdown.
        self.gracefully_exit = False

        # Distributed rank info
        attn_tp_rank, attn_tp_size, attn_dp_rank, attn_dp_size = (
            compute_dp_attention_world_info(
                server_args.enable_dp_attention,
                tp_rank,
                server_args.tp_size,
                server_args.dp_size,
                server_args.attn_cp_size,
            )
        )
        self.ps = ParallelState(
            tp_rank=tp_rank,
            tp_size=server_args.tp_size,
            pp_rank=pp_rank,
            pp_size=server_args.pp_size,
            dp_rank=dp_rank,
            dp_size=server_args.dp_size,
            attn_tp_rank=attn_tp_rank,
            attn_tp_size=attn_tp_size,
            attn_cp_rank=attn_cp_rank,
            attn_cp_size=server_args.attn_cp_size,
            attn_dp_rank=attn_dp_rank,
            attn_dp_size=attn_dp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=server_args.ep_size,
            moe_dp_rank=moe_dp_rank,
            moe_dp_size=server_args.moe_dp_size,
            gpu_id=gpu_id,
        )

        # Init model configs
        self.init_model_config()

        # Init metrics stats
        self.metrics_collector_context = SchedulerMetricsCollector.init_new(
            server_args=self.server_args,
            ps=self.ps,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank,
            enable_priority_scheduling=self.enable_priority_scheduling,
            enable_lora=self.enable_lora,
            enable_hierarchical_cache=self.enable_hierarchical_cache,
        )
        self.metrics_collector = self.metrics_collector_context.collector

        # Init inter-process communication
        self.init_ipc_channels(port_args)
        self.init_idle_sleeper()

        self.mm_receiver = None
        self.disagg_prefill_bootstrap_queue = None
        self.disagg_prefill_inflight_queue = None
        self.disagg_decode_prealloc_queue = None
        self.disagg_decode_transfer_queue = None

        # Init ZBAL, switch allocator should before any torch alloc action
        self.init_zbal_on_npu()

        # Init PD-multiplexing context
        if self.enable_pdmux:
            self.init_pdmux()

        # Init tokenizer
        self.init_tokenizer()

        # Init moe config and GEMM config (FP8 GEMM, etc.)
        self.init_moe_gemm_config()

        # Init mamba backend
        self.init_mamba_backend()

        # Must precede init_model_worker: revert targets like _init_pools run during it,
        # so patching them afterwards is a no-op.
        maybe_revert_pr_fix()

        # Launch a model worker and draft model worker if using speculative decoding
        self.init_model_worker()

        if (t := envs.SGLANG_TEST_STUCK_SCHEDULER_INIT.get()) > 0:
            time.sleep(t)

        # Init cache and memory pool
        result = kv_cache_builder.build_kv_cache(
            server_args=self.server_args,
            model_config=self.model_config,
            tp_worker=self.tp_worker,
            page_size=self.page_size,
            spec_algorithm=self.spec_algorithm,
            attn_tp_cpu_group=self.attn_tp_cpu_group,
            tp_cpu_group=self.tp_cpu_group,
            attn_cp_cpu_group=self.attn_cp_cpu_group,
            enable_metrics=self.server_args.enable_metrics,
            enable_kv_cache_events=bool(
                self.server_args.kv_events_config
                and self.ps.attn_tp_rank == 0
                and self.ps.attn_cp_rank == 0
            ),
            ps=self.ps,
            tp_group=self.tp_group,
            pp_group=self.pp_group,
            enable_hierarchical_cache=self.enable_hierarchical_cache,
        )
        self.is_hybrid_swa = result.is_hybrid_swa
        self.is_hybrid_ssm = result.is_hybrid_ssm
        self.sliding_window_size = result.sliding_window_size
        self.full_tokens_per_layer = result.full_tokens_per_layer
        self.swa_tokens_per_layer = result.swa_tokens_per_layer
        self.req_to_token_pool = result.req_to_token_pool
        self.token_to_kv_pool_allocator = result.token_to_kv_pool_allocator
        self.disable_radix_cache = result.disable_radix_cache
        self.tree_cache = result.tree_cache

        state_allocators = ()
        if envs.SGLANG_AGENTIC_KV_LIFECYCLE.get() and hasattr(
            self.req_to_token_pool, "mamba_allocator"
        ):
            state_allocators = (self.req_to_token_pool.mamba_allocator,)
        self.agentic_p_workset_broker = AgenticPWorksetLeaseBroker(
            self.page_size,
            state_allocators=state_allocators,
            mamba_req_to_token_pool=(
                self.req_to_token_pool if state_allocators else None
            ),
        )

        if (c := self.tp_worker.model_runner.canary_manager) is not None:
            c.attach_radix_cache(self.tree_cache)

        if self.enable_hisparse:
            # Coordinator was created inside ModelRunner.initialize() before CUDA graph capture
            self.hisparse_coordinator = self.tp_worker.model_runner.hisparse_coordinator
            self.hisparse_coordinator.set_decode_producer_stream(self.forward_stream)

        if self.server_args.disaggregation_mode == "decode" and (
            self.server_args.disaggregation_decode_enable_offload_kvcache
            or envs.SGLANG_AGENTIC_KV_LIFECYCLE.get()
        ):
            manager_class = DecodeKVCacheOffloadManager
            if envs.SGLANG_AGENTIC_KV_LIFECYCLE.get():
                from sglang.srt.disaggregation.agentic_decode_manager import (
                    DecodeKVCacheOffloadManager as AgenticDecodeKVCacheOffloadManager,
                )

                manager_class = AgenticDecodeKVCacheOffloadManager
            self.decode_offload_manager = manager_class(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                tp_group=(
                    self.attn_tp_cpu_group
                    if self.server_args.enable_dp_attention
                    else self.tp_cpu_group
                ),
                tree_cache=self.tree_cache,
                server_args=self.server_args,
            )
        else:
            self.decode_offload_manager = None

        # Register draft KV pool (when spec + HiCache co-enabled).
        kv_cache_builder.maybe_register_hicache_draft(
            tree_cache=self.tree_cache,
            draft_worker=self.draft_worker,
            spec_algorithm=self.spec_algorithm,
            server_args=self.server_args,
            enable_hierarchical_cache=self.enable_hierarchical_cache,
            page_size=self.page_size,
        )

        # Init running status
        self.init_running_status()

        # Init chunked prefill
        self.init_chunked_prefill()

        # Init diffusion LLM
        self.init_diffusion_llm()

        self.metrics_reporter = SchedulerMetricsReporter(
            scheduler=self,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank,
            metrics_collector_context=self.metrics_collector_context,
            metrics_collector=self.metrics_collector,
        )

        # Init schedule policy and new token estimation
        self.init_schedule_policy()

        # Init watchdog, memory saver, input blocker and recv skipper
        self.init_watch_dog_memory_saver_input_blocker()

        # Init profiler
        self.init_profiler()

        # Init prefill-decodedisaggregation
        self.init_disaggregation()

        # Init overlap schedule
        self.init_overlap()

        # Init Ngram Embedding
        self.maybe_init_ngram_embedding()

        # Init prefill kv split size when deterministic inference is enabled with various attention backends
        self.init_deterministic_inference_config()

        self.init_weight_updater()

        # Init request dispatcher
        self.init_request_dispatcher()

        # Init LoRA drainer for fair scheduling
        self.init_lora_drainer()

        # Init LoRA overlap loader
        self.init_lora_overlap_loader()

        # Init the grammar backend for constrained generation
        self.init_grammar_manager()

        self.maybe_init_scripted_scheduler_hook()

        self.init_request_receiver()

        self.init_dp_attn_adapter()

        self.init_pool_stats_observer()

        self.init_invariant_checker()

        self.init_kv_events_publisher()

        self.init_load_inquirer()

        self.init_output_streamer()

        self.init_batch_result_processor()

        self.is_initializing = False

    def init_zbal_on_npu(self):
        if _is_npu:
            from sglang.srt.hardware_backend.npu.utils import init_zbal

            if self.ps.pp_size > 1:
                logger.error("only zbal mix mode support pp_size > 1!")
            init_zbal(
                self.ps.tp_size, self.ps.gpu_id, self.ps.tp_rank
            )  # only switch allocator if is mix mode

    def init_model_config(self):
        self.model_config = ModelConfig.from_server_args(self.server_args)
        if _is_npu:
            # make sure the page size is not larger than block_size and chunked_prefill_size on NPU backend
            # the npu backend request the defined page size to be no larger than block_size and chunked_prefill_size
            from sglang.srt.dllm.config import DllmConfig

            self.dllm_config = (  # For diffusion LLM
                DllmConfig.from_server_args(self.server_args)
                if self.server_args.dllm_algorithm is not None
                else None
            )
            if self.dllm_config:
                if self.dllm_config.block_size < self.page_size:
                    logger.warning(
                        "WARNING: "
                        f"The page size {self.page_size} should not be larger than dllm block size {self.dllm_config.block_size}."
                        f"Page size now falls back to {self.dllm_config.block_size}"
                    )
                    self.page_size = self.dllm_config.block_size

    def init_ipc_channels(self, port_args: PortArgs):
        is_rank_zero = (
            self.ps.pp_rank == 0
            and self.ps.attn_tp_rank == 0
            and self.ps.attn_cp_rank == 0
        )
        self.ipc_channels = SchedulerIpcChannels.create(
            port_args=port_args,
            is_rank_zero=is_rank_zero,
            skip_tokenizer_init=self.server_args.skip_tokenizer_init,
            metrics_enabled=self.server_args.enable_metrics
            and (
                self.ps.attn_tp_rank == 0
                or self.server_args.enable_metrics_for_all_schedulers
            ),
            enable_scripted_runtime=envs.SGLANG_TEST_SCRIPTED_RUNTIME.get(),
        )

        self.load_snapshot_writer = None
        if not is_rank_zero:
            return

        dp_rank = self.ps.dp_rank if self.ps.dp_rank is not None else 0
        try:
            self.load_snapshot_writer = create_load_snapshot_writer(
                self.server_args,
                port_args,
                self.ps.dp_size,
                dp_rank,
                publish_interval=self.server_args.load_snapshot_publish_interval,
            )
        except Exception as e:
            logger.warning("load snapshot writer init failed: %s", e)

    def init_idle_sleeper(self) -> None:
        if (
            self.ps.pp_rank == 0
            and self.ps.attn_tp_rank == 0
            and self.ps.attn_cp_rank == 0
            and self.server_args.sleep_on_idle
        ):
            self.idle_sleeper = IdleSleeper(
                sockets=[
                    self.ipc_channels.recv_from_tokenizer,
                    self.ipc_channels.recv_from_rpc,
                ],
            )
        else:
            self.idle_sleeper = None

    def publish_load_snapshot(self, force: bool = False):
        writer = self.load_snapshot_writer
        if writer is None:
            return
        if not force:
            writer.publish_counter += 1
            if writer.publish_counter < writer.publish_interval:
                return
        writer.publish_counter = 0
        try:
            result = self.load_inquirer.get_loads(GetLoadsReqInput(include=["all"]))
            writer.write(LoadSnapshot.from_get_loads_output(result))
        except Exception as e:
            logger.warning("load snapshot publish failed: %s", e)

    def handle_get_loads_req(self, req: GetLoadsReqInput):
        return self.load_inquirer.get_loads(req)

    def init_tokenizer(self):
        server_args = self.server_args
        self.is_generation = self.model_config.is_generation

        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                    use_fast=not server_args.disable_fast_image_processor,
                    tokenizer_backend=server_args.tokenizer_backend,
                    model_name=server_args.model_path,
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                    tokenizer_backend=server_args.tokenizer_backend,
                )

        # Load multimodal processor for M-RoPE fallback computation.
        self._mm_processor = None
        if self.model_config.is_multimodal and self.processor is not None:
            try:
                import_processors("sglang.srt.multimodal.processors")
                self._mm_processor = get_mm_processor(
                    self.model_config.hf_config,
                    server_args,
                    self.processor,
                    "default",
                    skip_mm_pool=True,
                )
            except Exception:
                logger.warning(
                    "Failed to load multimodal processor in scheduler; "
                    "M-RoPE fallback will not be available."
                )

        # Set reasoning_parser and think_end_id if --reasoning_parser is enabled
        if self.server_args.reasoning_parser and self.tokenizer:
            reasoning_parser = ReasoningParser(
                model_type=self.server_args.reasoning_parser, stream_reasoning=False
            )
            self.model_config.think_end_id = self.tokenizer.encode(
                reasoning_parser.detector.think_end_token, add_special_tokens=False
            )[0]

    def init_mamba_backend(self) -> None:
        initialize_mamba_selective_state_update_backend(self.server_args)

    def init_moe_gemm_config(self):
        # For the MM models, check the text_config for MoE settings
        config_to_check = getattr(
            self.model_config.hf_config, "text_config", self.model_config.hf_config
        )

        # Different MoE architectures expose the per-token expert count under
        # different attribute names (e.g. Gemma4 uses ``top_k_experts``).
        moe_topk_attrs = (
            "num_experts_per_tok",
            "num_experts_per_token",
            "top_k_experts",
            "moe_top_k",
        )
        if any(hasattr(config_to_check, attr) for attr in moe_topk_attrs):
            initialize_moe_config(self.server_args)

        # Initialize GEMM-related configuration for FP8 and FP4 backends.
        initialize_fp8_gemm_config(self.server_args)
        initialize_fp4_gemm_config(self.server_args)

        # This must be called after initialize_moe_config
        self.require_mlp_sync = require_mlp_sync(self.server_args)

    def init_tp_model_worker(self):
        worker_kwargs = dict(
            server_args=self.server_args,
            gpu_id=self.ps.gpu_id,
            tp_rank=self.ps.tp_rank,
            moe_ep_rank=self.ps.moe_ep_rank,
            pp_rank=self.ps.pp_rank,
            attn_cp_rank=self.ps.attn_cp_rank,
            moe_dp_rank=self.ps.moe_dp_rank,
            dp_rank=self.ps.dp_rank,
            nccl_port=self.nccl_port,
        )

        # FIXME: move tp worker's init logic outside of the scheduler.
        if use_mlx():
            from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker

            self.tp_worker = MlxTpModelWorker(**worker_kwargs)
        else:
            from sglang.srt.managers.tp_worker import TpModelWorker

            self.tp_worker = TpModelWorker(**worker_kwargs)

    def maybe_init_draft_worker(self):
        if self.spec_algorithm.is_none():
            self.draft_worker = None
            self.external_corpus_manager = None
            return

        # Launch a draft worker for speculative decoding
        draft_worker_kwargs = dict(
            server_args=self.server_args,
            gpu_id=self.ps.gpu_id,
            tp_rank=self.ps.tp_rank,
            moe_ep_rank=self.ps.moe_ep_rank,
            nccl_port=self.nccl_port,
            target_worker=self.tp_worker,
            dp_rank=self.ps.dp_rank,
            attn_cp_rank=self.ps.attn_cp_rank,
            moe_dp_rank=self.ps.moe_dp_rank,
        )

        if self.server_args.speculative_draft_load_format is not None:
            self.server_args.load_format = (
                self.server_args.speculative_draft_load_format
            )
            logger.info(
                f"Using draft model load_format: '{self.server_args.speculative_draft_load_format}'"
            )

        DraftWorkerClass = self.spec_algorithm.create_worker(self.server_args)
        self.draft_worker = DraftWorkerClass(**draft_worker_kwargs)

        if self.spec_algorithm.is_ngram():
            from sglang.srt.speculative.external_corpus_manager import (
                ExternalCorpusManager,
            )

            self.external_corpus_manager = ExternalCorpusManager(
                self.draft_worker,
                self.ipc_channels.send_to_tokenizer.send_output,
            )
        else:
            self.external_corpus_manager = None

    def init_target_memory_pool(self):
        """Allocate target KV cache pools if they have not been allocated yet."""
        if (
            self.tp_worker.model_runner.memory_pool_config is not None
            and self.tp_worker.model_runner.req_to_token_pool is not None
            and self.tp_worker.model_runner.token_to_kv_pool_allocator is not None
        ):
            return
        self.tp_worker.alloc_memory_pool()

    def init_memory_pools(self):
        """Allocate KV cache pools for target and draft workers."""
        self.init_target_memory_pool()
        if self.draft_worker is not None:
            pool, allocator = self.tp_worker.get_memory_pool()
            self.draft_worker.alloc_memory_pool(
                memory_pool_config=self.tp_worker.model_runner.memory_pool_config,
                req_to_token_pool=pool,
                token_to_kv_pool_allocator=allocator,
            )

    def init_all_attention_backends(self):
        """Initialize attention backends for all workers."""
        self.tp_worker.init_attention_backends()
        if self.draft_worker is not None:
            self.draft_worker.init_attention_backends()

    def init_all_cuda_graphs(self):
        """Capture cuda graphs for all workers."""
        self.tp_worker.init_cuda_graphs()
        if self.draft_worker is not None:
            self.draft_worker.init_cuda_graphs()

    def init_model_worker(self):
        # Load model weights.
        self.init_tp_model_worker()
        if self.spec_algorithm.is_frozen_kv_mtp():
            # Frozen-KV MTP draft construction needs the target KV pool.
            self.init_target_memory_pool()
        self.maybe_init_draft_worker()

        # Allocate KV cache pools for all workers.
        self.init_memory_pools()

        # TODO: make memory profile consider cuda graph memory as well
        self.init_all_attention_backends()
        self.init_all_cuda_graphs()

        # Dispatch the model worker
        if self.spec_algorithm.is_none():
            self.model_worker = self.tp_worker
        else:
            self.model_worker = self.draft_worker

        # Get token and memory info from the model worker
        (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_queued_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            self.forward_stream,
            _,
            _,
            _,
        ) = self.tp_worker.get_worker_info()
        self.dflash_prefill_refill_target = (
            resolve_dflash_prefill_refill_target(self.max_running_requests)
            if self.spec_algorithm.is_dflash()
            else 1
        )
        if not get_global_server_args().pp_max_micro_batch_size:
            get_global_server_args().pp_max_micro_batch_size = max(
                self.max_running_requests // self.ps.pp_size, 1
            )

        self.tp_group = get_tp_group()
        self.tp_cpu_group = self.tp_group.cpu_group
        self.attn_tp_group = get_attention_tp_group()
        self.attn_tp_cpu_group = self.attn_tp_group.cpu_group
        self.attn_cp_group = get_attention_cp_group()
        self.attn_cp_cpu_group = self.attn_cp_group.cpu_group
        self.pp_group = get_pp_group()
        self.world_group = get_world_group()

        # NOTE: dp_tp_* are request/data-plane coordination groups (not tensor collectives).
        # When DP attention is enabled, scope to the attention-TP group; otherwise use
        # the base TP group. Entry rank is the local rank 0 in that group.
        # Use the CPU (gloo) group to broadcast VLM Python objects and avoid CUDA
        # stream/device coupling (#11910).
        self.dp_tp_group = (
            self.attn_tp_group
            if self.server_args.enable_dp_attention
            else self.tp_group
        )
        self.dp_tp_cpu_group = self.dp_tp_group.cpu_group

        # TODO(Jialin): Migrate pad_input_ids implementations to return array.
        self.pad_input_ids_func = self.tp_worker.get_pad_input_ids_func()
        set_random_seed(self.random_seed)

        # Print debug info
        avail_mem = get_available_gpu_memory(
            self.device, self.ps.gpu_id, empty_cache=False
        )
        if self.ps.tp_rank == 0:
            logger.info(
                f"max_total_num_tokens={self.max_total_num_tokens}, "
                f"chunked_prefill_size={self.server_args.chunked_prefill_size}, "
                f"max_prefill_tokens={self.max_prefill_tokens}, "
                f"max_running_requests={self.max_running_requests}, "
                f"context_len={self.model_config.context_len}, "
                f"{'available_cpu_mem' if self.device == 'cpu' else 'available_gpu_mem'}={avail_mem:.2f} GB"
            )

        if self.server_args.enable_metrics:
            self.metrics_collector.emit_constants(
                max_total_num_tokens=self.max_total_num_tokens,
                # TODO: max_running_requests_under_SLO has no setter — dead chain.
                max_running_requests_under_SLO=getattr(
                    self, "max_running_requests_under_SLO", None
                ),
                engine_startup_time=0.0,
                engine_load_weights_time=0.0,
                page_size=self.page_size,
                num_pages=self.max_total_num_tokens // self.page_size,
                context_len=self.model_config.context_len,
                startup_available_gpu_memory_gb=avail_mem,
            )

    def init_running_status(self):
        if envs.SGLANG_AGENTIC_KV_LIFECYCLE.get() and self.server_args.page_size < 64:
            raise ValueError(
                "SGLANG_AGENTIC_KV_LIFECYCLE V1 requires --page-size >= 64; "
                "smaller pages make request-level manifests unnecessarily large"
            )
        self.waiting_queue: List[Req] = []
        # Requests whose parent D snapshot is not committed yet.  They carry
        # metadata only and consume neither P host cache nor P GPU KV memory.
        self.agentic_kv_waiting_queue: List[Tuple[Req, float]] = []
        # Native TP requires every rank to construct identical model rows.
        # Direct/Slow completion is intentionally asynchronous per shard, so
        # rank-local completion time must never become Prefill queue order.
        # Requests receive this sequence when the broadcast request first
        # enters P; the sequence is therefore identical on every TP rank.
        self.agentic_tp_prefill_sequence = 0
        # TP=1 uses an edge-triggered ready pipeline. Background Direct/Slow
        # state machines publish snapshot completions; the scheduler indexes
        # metadata waiters and consumes only exact ready events instead of
        # rescanning and reclassifying the whole queue every model iteration.
        self.agentic_kv_waiting_by_rid: Dict[str, Tuple[Req, float]] = {}
        self.agentic_kv_waiting_by_parent: Dict[str, Dict[str, Req]] = {}
        self.agentic_kv_progress_queues: Dict[str, Deque[str]] = {
            "fast": deque(),
            "slow": deque(),
            "new": deque(),
        }
        self.agentic_kv_progress_enqueued: set[str] = set()
        self.agentic_kv_retry_heap: List[Tuple[float, int, str]] = []
        self.agentic_kv_retry_deadlines: Dict[str, float] = {}
        self.agentic_kv_retry_sequence = 0
        self.agentic_kv_waiting_tombstones = 0
        # Reverse D->P Direct transfers are discovered from the router's
        # lightweight arrival marker, before a tokenized Req exists.  Entries
        # remain allocator-owned until that Req arrives and binds the KV into
        # the P Radix cache.
        self.agentic_early_direct_receives: Dict[str, AgenticEarlyDirectReceive] = {}
        self.agentic_early_direct_terminal: Dict[str, float] = {}
        # Router arrivals are delivered by inotify into a FIFO admission
        # queue.  Transport completion has a separate queue consumed by the
        # GPU scheduler; neither path scans all marker files or all receivers.
        self.agentic_early_direct_admission_queue: Deque[
            Tuple[RequestGeneration, dict, Optional[Any]]
        ] = deque()
        self.agentic_early_direct_admission_ids: set[str] = set()
        self.agentic_early_direct_completion_queue: Deque[str] = deque()
        # Direct and Slow restore share the ordinary P KV pool.  Background
        # workers publish intents; only the scheduler services physical page
        # allocation and release, preserving allocator/Radix ownership rules.
        # Created after KV-pool construction with both Attention and hybrid
        # state allocators.  Do not overwrite it with an Attention-only broker.
        # TP rank 0 owns one ordered set of Direct admissions.  A dedicated
        # tmpfs mailbox grants the same request-generation to every rank's
        # background progress worker; each receives only its physical KV-head
        # shard.  Native scheduler broadcast is retained only for the final
        # synchronized Radix bind/clear boundary.
        self.agentic_tp_direct_admission_active: Dict[
            str,
            Tuple[
                RequestGeneration,
                float,
                Optional[int],
                int,
                Optional[AgenticPWorksetLease],
            ],
        ] = {}
        self.agentic_tp_direct_visible_order: List[str] = []
        self.agentic_tp_direct_command_visible = False
        self.agentic_tp_direct_group_status: Dict[str, int] = {}
        self.agentic_tp_direct_local_admitted: set[str] = set()
        self.agentic_tp_direct_local_failed: set[str] = set()
        self.agentic_tp_direct_local_rolled_back: set[str] = set()
        self.agentic_tp_host_active = None
        self.agentic_tp_host_active_since = 0.0
        self.agentic_tp_host_command_visible = False
        self.agentic_tp_host_group_status = 0
        # Slow restores are a bounded pipeline, not a single global baton.
        # The legacy scalar fields above remain compatibility aliases for
        # tests and external instrumentation; all scheduling decisions use
        # these request-generation keyed maps.
        self.agentic_tp_host_active_requests: Dict[str, RequestGeneration] = {}
        self.agentic_tp_host_active_since_by_snapshot: Dict[str, float] = {}
        self.agentic_tp_host_group_statuses: Dict[str, int] = {}
        self.agentic_tp_host_local_admitted: set[str] = set()
        self.agentic_tp_workset_retire_active: set[str] = set()
        self.agentic_tp_workset_retire_visible: set[str] = set()
        self.agentic_tp_workset_retire_group_statuses: Dict[str, int] = {}
        # Decode release is a level-triggered TP command.  Rank zero keeps the
        # selected snapshot active until every physical shard ACKs its local
        # allocator release; a one-tick scalar can strand a follower whenever
        # its Host/Direct I/O lock is busy at the broadcast boundary.
        self.agentic_tp_decode_release_active: Optional[str] = None
        if self.tp_size > 1 and envs.SGLANG_AGENTIC_KV_LIFECYCLE.get():
            mailbox_dir = os.getenv("SGLANG_PD_P_READY_DIR", "/dev/shm")
            common = {
                "tp_rank": self.tp_rank,
                "tp_size": self.tp_size,
                "directory": mailbox_dir,
            }
            # Snapshot/room identities are globally unique within one run, so
            # namespaces need not encode a rank-local engine id.  This lets P
            # and D exchange one logical receipt without another collective.
            self.agentic_tp_direct_mailbox = TPGroupMailbox("d2p-direct", **common)
            self.agentic_tp_host_mailbox = TPGroupMailbox("d2p-host", **common)
            self.agentic_tp_workset_retire_mailbox = TPGroupMailbox(
                "p-workset-retire", **common
            )
            self.agentic_tp_p2d_sender_mailbox = TPGroupMailbox("p2d-sender", **common)
            self.agentic_tp_p2d_receiver_mailbox = TPGroupMailbox(
                "p2d-receiver", **common
            )
            self.agentic_tp_p2d_admission_mailbox = TPGroupMailbox(
                "p2d-admission", **common
            )
            self.agentic_tp_p2d_cleanup_mailbox = TPGroupMailbox(
                "p2d-cleanup", **common
            )
            self.agentic_tp_decode_release_mailbox = TPGroupMailbox(
                "d-release", **common
            )
        else:
            self.agentic_tp_direct_mailbox = None
            self.agentic_tp_host_mailbox = None
            self.agentic_tp_workset_retire_mailbox = None
            self.agentic_tp_p2d_sender_mailbox = None
            self.agentic_tp_p2d_receiver_mailbox = None
            self.agentic_tp_p2d_admission_mailbox = None
            self.agentic_tp_p2d_cleanup_mailbox = None
            self.agentic_tp_decode_release_mailbox = None
        self.agentic_early_direct_arrival_watcher = None
        self.agentic_early_claim_store = None
        self.agentic_early_direct_poll_lock = threading.RLock()
        # P->D sender completion polling and reverse D->P Direct progress use
        # the same NIXL agent.  The Python binding performs manager-wide
        # control work, so allowing every P->D worker to enter it concurrently
        # can starve get_new_notifs() for seconds under a sustained burst.
        # Serialize only those short NIXL control calls; DMA itself remains
        # asynchronous and fully concurrent.  Direct sets the event before
        # taking the lock so P->D workers yield at the next progress step.
        self.agentic_nixl_control_lock = threading.Lock()
        self.agentic_direct_poll_requested = threading.Event()
        # Direct transport progress is owned exclusively by the background
        # worker. The GPU scheduler only inspects/binds completed entries and
        # must never wait for NIXL progress.
        self.agentic_early_direct_cycle_lock = threading.Lock()
        self.agentic_early_direct_progress_stop = threading.Event()
        self.agentic_early_direct_progress_thread = None
        # The running decoding batch for continuous batching
        self.running_batch: ScheduleBatch = ScheduleBatch(reqs=[], batch_is_full=False)
        # The current forward batch
        self.cur_batch: Optional[ScheduleBatch] = None
        # The last forward batch
        self.last_batch: Optional[ScheduleBatch] = None
        self.forward_ct = 0
        self.return_health_check_ipcs: Deque[Optional[str]] = deque()
        self.flush_wrapper = SchedulerFlushWrapper(
            flush_cache=self.flush_cache,
            is_fully_idle=self.is_fully_idle,
            ipc_channels=self.ipc_channels,
        )
        self.session_controller = SessionController(self.tree_cache)
        self.forward_sleep_time = None
        self._engine_paused = False

    def init_chunked_prefill(self):
        self.chunked_prefill_size = self.server_args.chunked_prefill_size
        uses_transformers_backend = (
            get_resolved_model_impl(self.model_config) == ModelImpl.TRANSFORMERS
        )
        if (
            self.chunked_prefill_size is not None
            and self.chunked_prefill_size > 0
            and self.model_config.is_multimodal
            and uses_transformers_backend
        ):
            logger.warning(
                "Chunked prefill is disabled for multimodal models with the "
                "Transformers backend to avoid partial multimodal chunk mismatches."
            )
            self.chunked_prefill_size = None
        elif self.chunked_prefill_size is not None and self.chunked_prefill_size <= 0:
            self.chunked_prefill_size = None
        self.chunked_req = None
        self._pending_chunked_abort_req = None
        self.is_mixed_chunk = (
            self.chunked_prefill_size is not None
            and self.server_args.enable_mixed_chunk
        )

        # Init the dynamic chunking predictor for PP
        self.enable_dynamic_chunking = (
            self.server_args.enable_dynamic_chunking and self.ps.pp_size > 1
        )
        if self.enable_dynamic_chunking:
            try:
                self.profile_and_init_predictor()
            except Exception as e:
                logger.warning(
                    f"[PP Dynamic Chunk] Failed to profile prefill latency: {e}. "
                    "Dynamic chunking will be disabled."
                )
                self.enable_dynamic_chunking = False

    def init_schedule_policy(self):
        # Init schedule policy and new token estimation
        self.policy = SchedulePolicy(
            self.schedule_policy,
            self.tree_cache,
            self.enable_hierarchical_cache,
            self.enable_priority_scheduling,
            self.schedule_low_priority_values_first,
        )
        self.prefill_delayer: Optional[PrefillDelayer] = None
        self.max_prefill_bs: int = 0
        if self.server_args.enable_prefill_delayer:
            if self.server_args.disaggregation_mode == "decode":
                logger.info(
                    "Ignoring --enable-prefill-delayer on decode engine "
                    "(no prefill scheduling path; delayer would be a no-op)."
                )
            else:
                self.prefill_delayer = PrefillDelayer(
                    dp_size=self.ps.dp_size,
                    attn_tp_size=self.ps.attn_tp_size,
                    cpu_group=self.tp_cpu_group,
                    device_group=self.tp_group.device_group,
                    server_args=self.server_args,
                    metrics_collector=(
                        self.metrics_collector
                        if self.metrics_reporter.enable_metrics
                        else None
                    ),
                    max_delay_passes=self.server_args.prefill_delayer_max_delay_passes,
                    token_usage_low_watermark=self.server_args.prefill_delayer_token_usage_low_watermark,
                    device=self.tp_group.device,
                )

        # NOTE: preemption is enabled by default for priority scheduling.
        self.enable_priority_preemption = (
            self.enable_priority_scheduling
            and not self.server_args.disable_priority_preemption
        )

        self.new_token_ratio_tracker = NewTokenRatioTracker.from_server_args(
            self.server_args
        )

    def init_soft_watchdog(self, server_args: ServerArgs):
        if (x := server_args.soft_watchdog_timeout) is not None:
            self.soft_watchdog = create_scheduler_watchdog(
                self, watchdog_timeout=x, soft=True
            )

    def init_watch_dog_memory_saver_input_blocker(self):
        # Start watchdog thread
        self.watchdog = create_scheduler_watchdog(
            self, watchdog_timeout=self.server_args.watchdog_timeout
        )

        # Init memory saver, profiler and metric stats
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.server_args.enable_memory_saver
        )

        # Init recv skipper and input blocker
        self.recv_skipper = SchedulerRecvSkipper.maybe_create(self.server_args)
        self.input_blocker = (
            SchedulerInputBlocker(noop=self.ps.attn_tp_rank != 0)
            if get_bool_env_var("SGLANG_ENABLE_COLOCATED_BATCH_GEN")
            else None
        )

        # Configure GC logger
        if envs.SGLANG_LOG_GC.get():
            configure_gc_logger()

    def init_disaggregation(self):
        self.disaggregation_mode = DisaggregationMode(
            self.server_args.disaggregation_mode
        )
        self.transfer_backend = TransferBackend(
            self.server_args.disaggregation_transfer_backend
        )

        # todo: should we fix this when enabling mtp or it doesn't matter since we only enable mtp in decode node thus we don't transfer draft kvs between P and D?
        draft_token_to_kv_pool, model_config = kv_cache_builder.get_draft_kv_pool(
            draft_worker=self.draft_worker,
            spec_algorithm=self.spec_algorithm,
            server_args=self.server_args,
        )
        # Default to the target model_config so the MetadataBuffers branches
        # below can always access it; overridden by the draft model_config
        # when this node runs a spec module.
        if model_config is None:
            model_config = self.model_config

        if (
            self.disaggregation_mode == DisaggregationMode.DECODE
        ):  # *2 for the headroom.
            buffer_size = (self.req_to_token_pool.size) * 2
            self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
                buffer_size
            )
            self.disagg_metadata_buffers = MetadataBuffers(
                buffer_size,
                hidden_size=(
                    model_config.spec_hidden_size
                    if self.spec_algorithm.carries_draft_hidden_states()
                    else 16  # minimal padding size for RDMA
                ),
                hidden_states_dtype=(
                    model_config.dtype
                    if self.spec_algorithm.carries_draft_hidden_states()
                    else torch.float32
                ),
                custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
            )

            # The decode requests polling kv cache
            self.disagg_decode_transfer_queue = DecodeTransferQueue(
                gloo_group=self.attn_tp_cpu_group,
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                tp_rank=self.ps.tp_rank,
                metadata_buffers=self.disagg_metadata_buffers,
                scheduler=self,
                tree_cache=self.tree_cache,
            )

            # The decode requests pending for pre-allocation
            self.disagg_decode_prealloc_queue = DecodePreallocQueue(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                draft_token_to_kv_pool=draft_token_to_kv_pool,
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                metadata_buffers=self.disagg_metadata_buffers,
                scheduler=self,
                transfer_queue=self.disagg_decode_transfer_queue,
                tree_cache=self.tree_cache,
                gloo_group=self.attn_tp_cpu_group,
                tp_rank=self.ps.tp_rank,
                tp_size=self.ps.tp_size,
                dp_size=self.server_args.dp_size,
                gpu_id=self.ps.gpu_id,
                bootstrap_port=self.server_args.disaggregation_bootstrap_port,
                max_total_num_tokens=self.max_total_num_tokens,
                pp_rank=self.ps.pp_rank,
                num_reserved_decode_tokens=self.server_args.num_reserved_decode_tokens,
                transfer_backend=self.transfer_backend,
            )
            if self.decode_offload_manager is not None:
                self.decode_offload_manager.attach_agentic_relay_manager(
                    self.disagg_decode_prealloc_queue.kv_manager,
                    self.req_to_metadata_buffer_idx_allocator,
                )
                self.decode_offload_manager.start_decode_io_progress_worker(
                    self.disagg_decode_prealloc_queue,
                    self.disagg_decode_transfer_queue,
                )

        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            # *2 for the headroom.
            buffer_size = self.max_running_requests * 2
            self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
                buffer_size
            )
            self.disagg_metadata_buffers = MetadataBuffers(
                buffer_size,
                hidden_size=(
                    model_config.spec_hidden_size
                    if self.spec_algorithm.carries_draft_hidden_states()
                    else 16  # minimal padding size for RDMA
                ),
                hidden_states_dtype=(
                    model_config.dtype
                    if self.spec_algorithm.carries_draft_hidden_states()
                    else torch.float32
                ),
                custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
            )

            self.disagg_prefill_bootstrap_queue = PrefillBootstrapQueue(
                token_to_kv_pool=self.token_to_kv_pool_allocator.get_kvcache(),
                draft_token_to_kv_pool=draft_token_to_kv_pool,
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                metadata_buffers=self.disagg_metadata_buffers,
                tp_rank=self.ps.tp_rank,
                tp_size=self.ps.tp_size,
                gpu_id=self.ps.gpu_id,
                bootstrap_port=self.server_args.disaggregation_bootstrap_port,
                gloo_group=self.attn_tp_cpu_group,
                max_total_num_tokens=self.max_total_num_tokens,
                scheduler=self,
                pp_rank=self.ps.pp_rank,
                pp_size=self.ps.pp_size,
                transfer_backend=self.transfer_backend,
            )
            self.agentic_direct_runtime = None
            self.agentic_host_staging_manager = None
            self.agentic_p2d_host_staging_manager = None
            if (
                envs.SGLANG_AGENTIC_KV_LIFECYCLE.get()
                and envs.SGLANG_AGENTIC_KV_FAST_TOOL_THRESHOLD.get() > 0
            ):
                kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
                self.agentic_direct_runtime = create_agentic_direct_runtime(
                    role=DisaggregationMode.DECODE,
                    kv_pool=kv_pool,
                    server_args=self.server_args,
                    engine_rank=self.tp_rank,
                    pp_rank=self.pp_rank,
                    gpu_id=self.gpu_id,
                    total_kv_heads=self.model_config.get_total_num_kv_heads(),
                    req_to_token_pool=self.req_to_token_pool,
                )
                validate_agentic_mamba_tracking(
                    self.agentic_direct_runtime.manager.kv_args,
                    self.server_args,
                )
                early_claim_dir = os.environ.get(
                    "SGLANG_AGENTIC_KV_EARLY_CLAIM_DIR", ""
                )
                if not early_claim_dir:
                    p_ready_dir = os.environ.get("SGLANG_PD_P_READY_DIR", "")
                    if p_ready_dir:
                        early_claim_dir = os.path.join(p_ready_dir, "early-claims")
                if early_claim_dir:
                    self.agentic_early_claim_store = AgenticEarlyClaimStore(
                        early_claim_dir
                    )
                if (
                    self.agentic_early_claim_store is not None
                    and not envs.SGLANG_AGENTIC_KV_FORCE_SLOW_PATH.get()
                ):
                    marker_max_age = max(
                        5.0,
                        envs.SGLANG_AGENTIC_KV_FAST_TOOL_THRESHOLD.get()
                        + envs.SGLANG_AGENTIC_KV_DIRECT_HANDSHAKE_TIMEOUT.get()
                        + 1.0,
                    )
                    # Every TP rank watches the same node-local arrival
                    # stream.  TP0 is still the only admission authority, but
                    # followers must be able to consume its background grant
                    # without waiting for the model scheduler's next native
                    # request broadcast.
                    self.agentic_early_direct_arrival_watcher = (
                        self.agentic_early_claim_store.watch_arrivals(
                            max_age_seconds=marker_max_age
                        )
                    )
                    logger.info(
                        "Agentic P unified workset leases enabled total_tokens=%d",
                        self.max_total_num_tokens,
                    )
                controller = None
                if envs.SGLANG_AGENTIC_KV_CUSTOM_STORAGE_ONLY.get():
                    # Direct and Slow share the same lifecycle metadata.  The
                    # controller exists even when Host staging is ablated,
                    # matching the validated pd behavior.
                    controller = create_agentic_storage_controller(
                        token_allocator=self.token_to_kv_pool_allocator,
                        server_args=self.server_args,
                        tp_rank=self.tp_rank,
                        tp_size=self.tp_size,
                        pp_rank=self.pp_rank,
                        pp_size=self.pp_size,
                        model_name=self.server_args.served_model_name,
                    )
                    self.agentic_storage_controller = controller
                if envs.SGLANG_AGENTIC_KV_HOST_STAGING.get():
                    ledger_base = envs.SGLANG_AGENTIC_KV_LEDGER_PATH.get()
                    staging_ledger_path = (
                        envs.SGLANG_AGENTIC_KV_STAGING_LEDGER_PATH.get()
                        or f"{ledger_base}.staging"
                    )
                    if (
                        not ledger_base
                        and not envs.SGLANG_AGENTIC_KV_STAGING_LEDGER_PATH.get()
                    ):
                        raise ValueError(
                            "P Host staging requires SGLANG_AGENTIC_KV_LEDGER_PATH "
                            "or SGLANG_AGENTIC_KV_STAGING_LEDGER_PATH"
                        )
                    if controller is None:
                        controller = getattr(self.tree_cache, "cache_controller", None)
                        if controller is None:
                            raise ValueError(
                                "P Host staging requires a storage controller"
                            )
                    expected_tool_seconds = {
                        str(name): float(seconds)
                        for name, seconds in json.loads(
                            envs.SGLANG_AGENTIC_KV_TOOL_MEAN_SECONDS.get()
                        ).items()
                    }
                    snapshot_store = controller.storage_backend.agentic_snapshot_store()
                    eviction_controller = None
                    if supports_agentic_kv_spill(controller.storage_backend):
                        eviction_controller = SharedSnapshotEvictionController(
                            snapshot_store,
                            ledger_path=ledger_base,
                            capacity_bytes=int(
                                envs.SGLANG_AGENTIC_KV_CAPACITY_GIB.get() * 1024**3
                            ),
                            high_watermark=(
                                envs.SGLANG_AGENTIC_KV_HIGH_WATERMARK.get()
                            ),
                            expected_tool_seconds=expected_tool_seconds,
                            reservation_ttl_seconds=(
                                envs.SGLANG_AGENTIC_KV_STALE_SECONDS.get()
                            ),
                        )
                    self.agentic_host_staging_manager = AgenticPHostStagingManager(
                        ledger=SharedHostStagingLedger(staging_ledger_path),
                        runtime=self.agentic_direct_runtime,
                        token_allocator=self.token_to_kv_pool_allocator,
                        workset_broker=self.agentic_p_workset_broker,
                        cache_controller=controller,
                        tree_cache=self.tree_cache,
                        page_size=self.server_args.page_size,
                        arena_directory=rank_scoped_arena_directory(
                            (
                                envs.SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_DIR.get()
                                or f"{staging_ledger_path}.arena"
                            ),
                            tp_rank=self.tp_rank,
                            tp_size=self.tp_size,
                            numa_node=rank_env_int(
                                "SGLANG_AGENTIC_KV_ARENA_NUMA_NODE",
                                "SGLANG_AGENTIC_KV_TP_NUMA_NODES",
                                tp_rank=self.tp_rank,
                            ),
                        ),
                        arena_capacity_bytes=int(
                            envs.SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_GIB.get() * 1024**3
                        ),
                        high_watermark=envs.SGLANG_AGENTIC_KV_P_HOST_HIGH_WATERMARK.get(),
                        low_watermark=envs.SGLANG_AGENTIC_KV_P_HOST_LOW_WATERMARK.get(),
                        hard_watermark=envs.SGLANG_AGENTIC_KV_P_HOST_HARD_WATERMARK.get(),
                        arena_numa_node=rank_env_int(
                            "SGLANG_AGENTIC_KV_ARENA_NUMA_NODE",
                            "SGLANG_AGENTIC_KV_TP_NUMA_NODES",
                            tp_rank=self.tp_rank,
                        ),
                        arena_domain=int(
                            os.environ.get("SGLANG_AGENTIC_KV_PREFILL_DOMAIN", "-1")
                        ),
                        tp_rank=self.tp_rank,
                        tp_size=self.tp_size,
                        expected_tool_seconds=expected_tool_seconds,
                        eviction_controller=eviction_controller,
                    )
                p2d_host_requested = os.getenv(
                    "SGLANG_AGENTIC_KV_P2D_HOST_STAGING", "0"
                ).lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                if p2d_host_requested:
                    p2d_ledger_path = os.getenv(
                        "SGLANG_AGENTIC_KV_P2D_STAGING_LEDGER_PATH",
                        f"{envs.SGLANG_AGENTIC_KV_STAGING_LEDGER_PATH.get()}.p2d",
                    )
                    p2d_arena_directory = os.getenv(
                        "SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_DIR",
                        f"{envs.SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_DIR.get()}.p2d",
                    )
                    p2d_numa_node = rank_env_int(
                        "SGLANG_AGENTIC_KV_ARENA_NUMA_NODE",
                        "SGLANG_AGENTIC_KV_TP_NUMA_NODES",
                        tp_rank=self.tp_rank,
                    )
                    self.agentic_p2d_host_staging_manager = (
                        AgenticPToDHostStagingManager(
                            ledger=SharedHostStagingLedger(p2d_ledger_path),
                            device_pool=kv_pool,
                            page_size=self.server_args.page_size,
                            arena_directory=rank_scoped_arena_directory(
                                p2d_arena_directory,
                                tp_rank=self.tp_rank,
                                tp_size=self.tp_size,
                                numa_node=p2d_numa_node,
                            ),
                            arena_capacity_bytes=int(
                                float(
                                    os.getenv(
                                        "SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_GIB",
                                        "32",
                                    )
                                )
                                * 1024**3
                            ),
                            prefill_domain=int(
                                os.getenv("SGLANG_AGENTIC_KV_PREFILL_DOMAIN", "0")
                            ),
                            numa_node=p2d_numa_node,
                            tp_rank=self.tp_rank,
                            tp_size=self.tp_size,
                            hard_watermark=float(
                                os.getenv(
                                    "SGLANG_AGENTIC_KV_P2D_HOST_HARD_WATERMARK",
                                    "0.90",
                                )
                            ),
                        )
                    )
                # Direct marker discovery and NIXL transport must continue
                # while the scheduler is inside a long Prefill forward. The
                # worker exclusively owns transport progress; the scheduler
                # only binds completed pages into Radix. This prevents a slow
                # transport operation from delaying the next GPU forward.
                if self.agentic_early_claim_store is not None:
                    self.agentic_early_direct_progress_thread = threading.Thread(
                        target=self._agentic_early_direct_progress_worker,
                        name=f"agentic-p-direct-{os.getpid()}",
                        daemon=True,
                    )
                    self.agentic_early_direct_progress_thread.start()
                    logger.info("Agentic P Direct transport progress worker enabled")
            # The prefill requests that are in the middle of kv sending
            self.disagg_prefill_inflight_queue: List[Req] = []
            self.start_prefill_transfer_progress_worker()

        # Init mm receiver for EPD disaggregation mode
        if (
            self.server_args.language_only
            and self.server_args.encoder_transfer_backend
            in ["zmq_to_scheduler", "mooncake"]
        ):
            self.mm_receiver = create_mm_receiver(
                self.server_args,
                dtype=self.model_config.dtype,
                hf_config=self.model_config.hf_config,
                pp_rank=self.ps.pp_rank,
                tp_rank=self.ps.tp_rank,
                tp_group=self.tp_group,
                scheduler=self,
            )

    def init_overlap(self):
        self.device_module = torch.get_device_module(self.device)

        # FutureMap is always-on: input_ids relay used in both modes.
        # Workers without the spec_v2_attn_backends override fall back to
        # target-only so the helper still produces a safe decision (no
        # accidental opt-out for unaudited shapes).
        if self.draft_worker is not None:
            attn_backends = getattr(
                self.draft_worker,
                "spec_v2_attn_backends",
                (self.tp_worker.model_runner.attn_backend,),
            )
        else:
            attn_backends = (self.tp_worker.model_runner.attn_backend,)
        needs_cpu_seq_lens = decide_needs_cpu_seq_lens(self.server_args, attn_backends)
        self.future_map = self.spec_algorithm.create_future_map(
            self.device,
            self.req_to_token_pool,
            needs_cpu_seq_lens=needs_cpu_seq_lens,
        )

        if use_mlx():
            # MLX uses its own overlap loop and does not create CUDA streams,
            # but the normal non-overlap scheduler path still relays decode
            # input IDs through FutureMap.
            self.result_queue: Deque = deque()
            return

        # forward_stream_ctx / copy_stream are also used by PP (non-overlap)
        # via scheduler_pp_mixin; init unconditionally to match main.
        self.forward_stream_ctx: CudaStreamContext = self.device_module.stream(
            self.forward_stream
        )
        self.copy_stream: CudaStream = self.device_module.Stream()
        self.copy_stream_ctx: CudaStreamContext = self.device_module.stream(
            self.copy_stream
        )

        if not self.enable_overlap:
            return

        self.batch_record_buf = [None] * 2
        self.batch_record_ct = 0

    def maybe_init_ngram_embedding(self):
        self.use_ngram_embedding = self.tp_worker.model_config.use_ngram_embedding
        if self.use_ngram_embedding:
            self.token_table = self.tp_worker.model_runner.token_table
            hf_config = self.tp_worker.model_config.hf_config
            self.ngram_embedding_n = hf_config.ngram_embedding_n
            self.ngram_embedding_k = hf_config.ngram_embedding_k

    def _maybe_prepare_ngram_embedding(
        self, batch: Optional[ScheduleBatch]
    ) -> Optional[ScheduleBatch]:
        """Fill the token table for ngram embedding before a forward pass."""
        if batch is None or not self.use_ngram_embedding:
            return batch
        batch.ne_token_table = self.token_table
        if batch.forward_mode == ForwardMode.EXTEND:
            all_tokens = []
            column_starts = []
            request_lengths = []
            for req in batch.reqs:
                start = len(req.prefix_indices)
                end = start + req.extend_input_len
                fill_ids = req.origin_input_ids + req.output_ids
                if start == 0:
                    tokens = fill_ids[start:end]
                    column_starts.append(0)
                elif start < self.ngram_embedding_n:
                    tokens = fill_ids[0:end]
                    column_starts.append(0)
                else:
                    # Prepend n-1 tokens before prefix_len for n-gram context
                    tokens = fill_ids[start - self.ngram_embedding_n + 1 : end]
                    column_starts.append(start - self.ngram_embedding_n + 1)
                all_tokens.extend(tokens)
                request_lengths.append(len(tokens))
            dtype = self.token_table.dtype
            device = self.token_table.device
            update_token_table(
                ne_token_table=self.token_table,
                tokens=torch.tensor(all_tokens, dtype=dtype, device=device),
                row_indices=batch.req_pool_indices,
                column_starts=torch.tensor(
                    column_starts, dtype=torch.int32, device=device
                ),
                req_lens=torch.tensor(
                    request_lengths, dtype=torch.int32, device=device
                ),
                ignore_tokens=None,
            )
        return batch

    def init_deterministic_inference_config(self):
        """Initialize deterministic inference configuration for different attention backends."""
        if not self.server_args.enable_deterministic_inference:
            self.truncation_align_size = None
            return

        backend_sizes = {
            "flashinfer": ("SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE", 4096),
            "triton": ("SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE", 4096),
        }
        env_var, default_size = backend_sizes.get(
            self.server_args.attention_backend, (None, None)
        )
        self.truncation_align_size = (
            get_int_env_var(env_var, default_size) if env_var else None
        )

    def init_request_dispatcher(self):
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.handle_generate_request),
                (TokenizedEmbeddingReqInput, self.handle_embedding_request),
                (BatchTokenizedGenerateReqInput, self.handle_batch_generate_request),
                (BatchTokenizedEmbeddingReqInput, self.handle_batch_embedding_request),
                (FlushCacheReqInput, self.flush_wrapper.handle),
                (ClearHiCacheReqInput, self.clear_hicache_storage_wrapped),
                (AttachHiCacheStorageReqInput, self.attach_hicache_storage_wrapped),
                (DetachHiCacheStorageReqInput, self.detach_hicache_storage_wrapped),
                (AbortReq, self.abort_request),
                (OpenSessionReqInput, self.open_session),
                (CloseSessionReqInput, self.close_session),
                (
                    UpdateWeightFromDiskReqInput,
                    self.weight_updater.update_weights_from_disk,
                ),
                (
                    InitWeightsUpdateGroupReqInput,
                    self.weight_updater.init_weights_update_group,
                ),
                (
                    DestroyWeightsUpdateGroupReqInput,
                    self.weight_updater.destroy_weights_update_group,
                ),
                (
                    InitWeightsSendGroupForRemoteInstanceReqInput,
                    self.init_weights_send_group_for_remote_instance,
                ),
                (
                    SendWeightsToRemoteInstanceReqInput,
                    self.send_weights_to_remote_instance,
                ),
                (
                    UpdateWeightsFromDistributedReqInput,
                    self.weight_updater.update_weights_from_distributed,
                ),
                (
                    UpdateWeightsFromTensorReqInput,
                    self.weight_updater.update_weights_from_tensor,
                ),
                (
                    UpdateWeightsFromIPCReqInput,
                    self.weight_updater.update_weights_from_ipc,
                ),
                (
                    GetWeightsByNameReqInput,
                    self.weight_updater.get_weights_by_name,
                ),
                (
                    ReleaseMemoryOccupationReqInput,
                    self.weight_updater.release_memory_occupation,
                ),
                (
                    ResumeMemoryOccupationReqInput,
                    self.weight_updater.resume_memory_occupation,
                ),
                (
                    CheckWeightsReqInput,
                    self.weight_updater.check_weights,
                ),
                (SlowDownReqInput, self.slow_down),
                (
                    ProfileReq,
                    lambda req: self.profiler_manager._profile(req),
                ),
                (FreezeGCReq, self.handle_freeze_gc),
                (ShutdownReq, self.handle_shutdown),
                (GetInternalStateReq, self.get_internal_state),
                (SetInternalStateReq, self.set_internal_state),
                (RpcReqInput, self.handle_rpc_request),
                (ExpertDistributionReq, self.expert_distribution_handle),
                (LoadLoRAAdapterReqInput, self.load_lora_adapter),
                (
                    LoadLoRAAdapterFromTensorsReqInput,
                    self.load_lora_adapter_from_tensors,
                ),
                (UnloadLoRAAdapterReqInput, self.unload_lora_adapter),
                (GetLoadsReqInput, self.handle_get_loads_req),
                (PauseGenerationReqInput, self.pause_generation),
                (ContinueGenerationReqInput, self.continue_generation),
                (ConfigureLoggingReq, self.configure_logging),
                (DumperControlReqInput, self.handle_dumper_control),
                (AddExternalCorpusReqInput, self.add_external_corpus),
                (
                    RemoveExternalCorpusReqInput,
                    self.remove_external_corpus,
                ),
                (
                    ListExternalCorporaReqInput,
                    self.list_external_corpora,
                ),
            ]
        )

    def _abort_on_running_timeout(self):
        # NOTE: this should be called before a batch is launched.
        timeout_s = envs.SGLANG_REQ_RUNNING_TIMEOUT.get()
        if timeout_s <= 0:
            return
        if self.running_batch.is_empty():
            return

        deadline = time.perf_counter() - timeout_s
        for req in self.running_batch.reqs:
            if not req.finished() and 0 < req.time_stats.forward_entry_time < deadline:
                req.to_finish = FINISH_ABORT(
                    "Request running timeout reached.", HTTPStatus.SERVICE_UNAVAILABLE
                )

    def get_init_info(self) -> Dict[str, Any]:
        """Return scheduler initialization info for handshake.

        This method provides the initialization info needed by the tokenizer manager
        and other components to verify the scheduler is ready.
        """
        result_dict = {
            "status": "ready",
            "max_total_num_tokens": self.max_total_num_tokens,
            "max_req_input_len": self.max_req_input_len,
        }

        return result_dict

    def release_host_resources(self) -> None:
        # Release pinned host buffers in userspace on graceful shutdown; see
        # HostKVCache.destroy. Called from run_scheduler_process's finally.
        if self.hisparse_coordinator is not None:
            self.hisparse_coordinator.destroy()

    def run_event_loop(self) -> None:
        """Run the scheduler's event loop.

        Sets up the schedule stream and dispatches to the appropriate event loop.
        The event loop blocks until shutdown.
        """
        if use_mlx():
            # MLX overlap uses mx.async_eval for CPU/GPU overlap,
            # not PyTorch MPS streams.
            dispatch_event_loop(self)
            return

        self.schedule_stream = self.device_module.Stream(priority=0)
        if self.device == "cpu":
            self.schedule_stream.synchronize = lambda: None  # No-op for CPU
        # DFLASH fences its shared req_to_token writes with verify_done /
        # plan-stream deps, so the global WAR barrier only serializes plan
        # overlap. TODO: generalize this global-barrier enablement policy.
        self._war_barrier_enabled = (
            is_cuda() or envs.SGLANG_ENABLE_WAR_BARRIER.get()
        ) and not self.spec_algorithm.is_dflash()
        with self.device_module.StreamContext(self.schedule_stream):
            dispatch_event_loop(self)

    def _apply_war_barrier(self):
        # Wait for the prev forward to finish reading the shared buffers this
        # iter's schedule will overwrite. Fast path: wait on the read-done event
        # the forward published after its snapshot (non-spec: decode graph;
        # spec: draft_extend), then clear it. Else fall back to whole-forward
        # wait_stream.
        if not self._war_barrier_enabled:
            return
        runner = self.model_worker.war_fastpath_runner
        ev = runner.war_fastpath_read_done_event
        if ev is not None:
            self.schedule_stream.wait_event(ev)
            runner.war_fastpath_read_done_event = None
        else:
            self.schedule_stream.wait_stream(self.forward_stream)

    @DynamicGradMode()
    def event_loop_normal(self):
        """A normal scheduler loop."""
        while True:
            if self.gracefully_exit:
                break

            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue

            # Get the next batch to run
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # When the server is idle, do self-check and re-init some states.
                self.on_idle()

            # Update last_batch
            self.last_batch = batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.invariant_checker.self_check_during_busy()

    @DynamicGradMode()
    def event_loop_overlap(self):
        """A scheduler loop that overlaps the CPU processing and GPU computation."""
        self.result_queue: Deque[
            Tuple[ScheduleBatch, Union[GenerationBatchResult, EmbeddingBatchResult]]
        ] = deque()

        def pop_and_process():
            # Process the results of the last batch
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        while True:
            if self.gracefully_exit:
                break

            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue

            self._apply_war_barrier()

            # Get the next batch to run
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch
            disable_overlap_for_batch = self.is_disable_overlap_for_batch(batch)

            # If we do not need to overlap the current batch with the last batch,
            # we can process the last batch immediately.
            if disable_overlap_for_batch:
                pop_and_process()

            # Launch the current batch
            if batch:
                batch_result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None

            # Process the last batch
            if self.last_batch:
                if not disable_overlap_for_batch:
                    pop_and_process()
            elif batch is None:
                # When the server is idle, do self-check and re-init some states
                self.on_idle()

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            if self.is_generation:
                self.launch_batch_sample_if_needed(batch_result)

            # Update last_batch
            self.last_batch = batch

            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.invariant_checker.self_check_during_busy()

    def is_disable_overlap_for_batch(self, batch: ScheduleBatch) -> bool:
        # For two consecutive prefill batches, we disable overlap to improve the TTFT of the first batch.
        # This might slightly hurt the throughput, so we use an environment variable to control it.
        # In DP attention mode, use the globally synchronized is_extend_in_batch
        # so all DP ranks make the same overlap decision (avoiding deadlock).
        # In non-DP mode, use the local forward_mode directly.
        if self.require_mlp_sync:
            is_extend = lambda b: b and b.is_extend_in_batch
        else:
            is_extend = lambda b: b and b.forward_mode.is_extend()

        batch_is_extend = is_extend(batch)
        last_batch_is_extend = is_extend(self.last_batch)

        disable_overlap_for_batch = (
            envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.get()
            and batch_is_extend
            and last_batch_is_extend
        )

        # We do not support overlap + spec + grammar yet,
        # so we need to turn off overlap for this batch.
        # TODO(lsyin): support overlap + spec + grammar
        need_grammar_sync = (
            batch
            and not batch.spec_algorithm.is_none()
            and batch.has_grammar
            and batch.forward_mode.is_decode()
            and len(self.result_queue) > 0
        )

        return disable_overlap_for_batch or need_grammar_sync

    @scheduler_nvtx_method("scheduler.process_input_requests")
    def process_input_requests(self, recv_reqs: List):
        now = time.monotonic()
        self.session_controller.maybe_reap(now)
        for recv_req in recv_reqs:
            # Skip health check when server is busy — ongoing requests already carry health info.
            if is_health_check_generate_req(recv_req) and not self.is_fully_idle(
                for_health_check=True
            ):
                self.return_health_check_ipcs.append(
                    getattr(recv_req, "http_worker_ipc", None)
                )
                continue

            output = self._request_dispatcher(recv_req)
            if output is not None:
                if not isinstance(output, RpcReqOutput):
                    self.ipc_channels.send_to_tokenizer.send_output(output, recv_req)
                else:
                    if self.ipc_channels.recv_from_rpc is not None:
                        self.ipc_channels.recv_from_rpc.send_pyobj(output)

        self.flush_wrapper.check_pending()
        if self.external_corpus_manager is not None:
            self.external_corpus_manager.check_pending_load()
        host_staging = getattr(self, "agentic_host_staging_manager", None)
        if host_staging is not None:
            host_staging.poll()
        self._drain_agentic_kv_waiting_queue()
        # Agentic request-generation ownership is advanced through the native
        # TP request broadcast.  Every rank must publish the physical progress
        # it made *after* draining this scheduler tick, so TP0 can derive the
        # next group command on a later broadcast.  Leaving these reducers
        # disconnected strands Slow H2D at PREPARE (and workset retirement at
        # its first phase): TP0 keeps broadcasting the old command because it
        # never observes the rank-local mailbox acknowledgements.
        if self.tp_size > 1 and envs.SGLANG_AGENTIC_KV_LIFECYCLE.get():
            self._agentic_tp_reduce_direct_status()
            self._agentic_tp_reduce_host_status()
            self._agentic_tp_reduce_workset_retire_status()

    def init_profiler(self) -> None:
        self.profiler_manager = SchedulerProfilerManager(
            ps=self.ps,
            dp_tp_cpu_group=self.dp_tp_cpu_group,
            get_forward_ct=lambda: self.forward_ct,
        )

    def init_weight_updater(self) -> None:
        self.weight_updater = SchedulerWeightUpdaterManager(
            tp_worker=self.tp_worker,
            draft_worker=self.draft_worker,
            tp_cpu_group=self.tp_cpu_group,
            memory_saver_adapter=self.memory_saver_adapter,
            flush_cache=self.flush_cache,
            is_fully_idle=self.is_fully_idle,
            scheduler=self,
            metrics_collector=self.metrics_collector,
        )

    def init_lora_drainer(self) -> None:
        if self.server_args.lora_drain_wait_threshold > 0.0:
            self.lora_drainer = LoRADrainer(
                self.server_args.max_loras_per_batch,
                self.server_args.lora_drain_wait_threshold,
            )
        else:
            self.lora_drainer = None

    def init_lora_overlap_loader(self) -> None:
        if self.enable_lora_overlap_loading:
            self.lora_overlap_loader = LoRAOverlapLoader(
                self.tp_worker.model_runner.lora_manager
            )

    def init_grammar_manager(self) -> None:
        self.grammar_manager = GrammarManager(self)

    def maybe_init_scripted_scheduler_hook(self) -> None:
        if envs.SGLANG_TEST_SCRIPTED_RUNTIME.get():
            from sglang.test.scripted_runtime.scheduler_hook import (
                ScriptedSchedulerHook,
            )

            self.scripted_scheduler_hook = ScriptedSchedulerHook(
                scheduler=self,
                tokenizer_recv_proxy=self.ipc_channels.recv_from_tokenizer,
            )
        else:
            self.scripted_scheduler_hook = None

    def init_request_receiver(self) -> None:
        agentic_tp_control = (
            self.tp_size > 1 and envs.SGLANG_AGENTIC_KV_LIFECYCLE.get()
        )
        if agentic_tp_control and self.server_args.enable_dp_attention:
            raise RuntimeError(
                "Agentic PD TP control does not yet support DP-attention; "
                "disable DP-attention or run the baseline lifecycle"
            )
        self.request_receiver = SchedulerRequestReceiver(
            recv_from_tokenizer=self.ipc_channels.recv_from_tokenizer,
            recv_from_rpc=self.ipc_channels.recv_from_rpc,
            recv_skipper=self.recv_skipper,
            input_blocker=self.input_blocker,
            mm_receiver=self.mm_receiver,
            ps=self.ps,
            tp_group=self.tp_group,
            tp_cpu_group=self.tp_cpu_group,
            attn_tp_group=self.attn_tp_group,
            attn_tp_cpu_group=self.attn_tp_cpu_group,
            attn_cp_group=self.attn_cp_group,
            attn_cp_cpu_group=self.attn_cp_cpu_group,
            world_group=self.world_group,
            server_args=self.server_args,
            model_config=self.model_config,
            max_recv_per_poll=self.max_recv_per_poll,
            stream_output=lambda *a, **kw: self.output_streamer.stream_output(*a, **kw),
            get_last_forward_mode=lambda: (
                self.last_batch.forward_mode if self.last_batch is not None else None
            ),
            scripted_scheduler_hook=self.scripted_scheduler_hook,
            prepare_tp_control=(
                self._agentic_tp_prepare_admission_control
                if agentic_tp_control
                else None
            ),
            consume_tp_control=(
                self._agentic_tp_consume_admission_control
                if agentic_tp_control
                else None
            ),
        )

    def init_dp_attn_adapter(self) -> None:
        self.dp_attn_adapter = SchedulerDPAttnAdapter(
            tp_group=self.tp_group,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            offload_tags=self.weight_updater.offload_tags,
            ps=self.ps,
            server_args=self.server_args,
            model_config=self.model_config,
            enable_overlap=self.enable_overlap,
            spec_algorithm=self.spec_algorithm,
            get_require_mlp_sync=lambda: self.require_mlp_sync,
        )

    def init_pool_stats_observer(self) -> None:
        self.pool_stats_observer = SchedulerPoolStatsObserver(
            tree_cache=self.tree_cache,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            req_to_token_pool=self.req_to_token_pool,
            session_controller=self.session_controller,
            hisparse_coordinator=self.hisparse_coordinator,
            is_hybrid_swa=self.is_hybrid_swa,
            is_hybrid_ssm=self.is_hybrid_ssm,
            enable_hisparse=self.enable_hisparse,
            full_tokens_per_layer=self.full_tokens_per_layer,
            swa_tokens_per_layer=self.swa_tokens_per_layer,
            max_total_num_tokens=self.max_total_num_tokens,
            get_last_batch=lambda: self.last_batch,
            get_running_batch=lambda: self.running_batch,
        )

    def init_invariant_checker(self) -> None:
        self.invariant_checker = SchedulerInvariantChecker(
            is_hybrid_swa=self.is_hybrid_swa,
            is_hybrid_ssm=self.is_hybrid_ssm,
            disaggregation_mode=self.disaggregation_mode,
            page_size=self.page_size,
            full_tokens_per_layer=self.full_tokens_per_layer,
            swa_tokens_per_layer=self.swa_tokens_per_layer,
            max_total_num_tokens=self.max_total_num_tokens,
            server_args=self.server_args,
            tree_cache=self.tree_cache,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            req_to_token_pool=self.req_to_token_pool,
            pool_stats_observer=self.pool_stats_observer,
            get_last_batch=lambda: self.last_batch,
            get_running_batch=lambda: self.running_batch,
        )

    def init_kv_events_publisher(self) -> None:
        self.kv_events_publisher = SchedulerKvEventsPublisher(
            kv_events_config=self.server_args.kv_events_config,
            ps=self.ps,
            attn_tp_rank=self.ps.attn_tp_rank,
            attn_cp_rank=self.ps.attn_cp_rank,
            attn_dp_rank=self.ps.attn_dp_rank,
            dp_rank=self.ps.dp_rank,
            tree_cache=self.tree_cache,
            send_metrics_from_scheduler=self.ipc_channels.send_metrics_from_scheduler,
            max_running_requests=self.max_running_requests,
            max_total_num_tokens=self.max_total_num_tokens,
            get_stats=lambda: self.metrics_reporter.stats,
        )

    def init_load_inquirer(self) -> None:
        self.load_inquirer = SchedulerLoadInquirer(
            disaggregation_mode=self.disaggregation_mode,
            ps=self.ps,
            server_args=self.server_args,
            max_total_num_tokens=self.max_total_num_tokens,
            max_running_requests=self.max_running_requests,
            pool_stats_observer=self.pool_stats_observer,
            tp_worker=self.tp_worker,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            spec_algorithm=self.spec_algorithm,
            get_running_batch=lambda: self.running_batch,
            get_waiting_queue=lambda: self.waiting_queue,
            get_stats=lambda: self.metrics_reporter.stats,
            get_chunked_req=lambda: self.chunked_req,
            get_disagg_prefill_bootstrap_queue=lambda: self.disagg_prefill_bootstrap_queue,
            get_disagg_prefill_inflight_queue=lambda: self.disagg_prefill_inflight_queue,
            get_disagg_decode_prealloc_queue=lambda: self.disagg_decode_prealloc_queue,
            get_disagg_decode_transfer_queue=lambda: self.disagg_decode_transfer_queue,
            get_spec_total_num_accept_tokens=lambda: self.metrics_reporter.spec_total_num_accept_tokens,
            get_spec_total_num_forward_ct=lambda: self.metrics_reporter.spec_total_num_forward_ct,
        )

    def init_output_streamer(self) -> None:
        self.output_streamer = SchedulerOutputStreamer(
            send_to_detokenizer=self.ipc_channels.send_to_detokenizer,
            tree_cache=self.tree_cache,
            ps=self.ps,
            server_args=self.server_args,
            is_generation=self.is_generation,
            spec_algorithm=self.spec_algorithm,
            disaggregation_mode=self.disaggregation_mode,
            enable_hicache_storage=lambda: self.enable_hicache_storage,
        )

    def init_batch_result_processor(self) -> None:
        self.batch_result_processor = SchedulerBatchResultProcessor(
            is_generation=self.is_generation,
            disaggregation_mode=self.disaggregation_mode,
            enable_overlap=self.enable_overlap,
            enable_overlap_mlx=self.enable_overlap_mlx,
            server_args=self.server_args,
            model_config=self.model_config,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            hisparse_coordinator=self.hisparse_coordinator,
            req_to_token_pool=self.req_to_token_pool,
            decode_offload_manager=self.decode_offload_manager,
            metrics_collector=self.metrics_collector,
            metrics_reporter=self.metrics_reporter,
            draft_worker=self.draft_worker,
            model_worker=self.model_worker,
            logprob_result_processor=SchedulerLogprobResultProcessor(
                server_args=self.server_args, model_config=self.model_config
            ),
            output_streamer=self.output_streamer,
            abort_request=self.abort_request,
        )

    def init_req_max_new_tokens(self, req):
        input_len = len(req.origin_input_ids)
        # Keep this bound consistent with PrefillAdder's admission budget:
        # ceil_page(input_len) + max_new_tokens + page_size must be strictly
        # smaller than max_total_num_tokens. Otherwise a request can be accepted
        # into the waiting queue but can never be scheduled, blocking the queue
        # and eventually making health checks fail.
        paged_input_len = -(-input_len // self.page_size) * self.page_size
        req.sampling_params.max_new_tokens = max(
            0,
            min(
                (
                    req.sampling_params.max_new_tokens
                    if req.sampling_params.max_new_tokens is not None
                    else 1 << 30
                ),
                self.max_req_len - input_len - 1,
                self.max_total_num_tokens - paged_input_len - self.page_size - 1,
            ),
        )

    def _process_and_broadcast_mm_inputs(
        self,
        raw_mm_inputs,
    ):
        """Materialize MultimodalInputs once on the entry rank and broadcast to others.

        Entry rank:
        - constructs MultimodalInputs.from_processor_output() once
        - broadcasts to other ranks in self.cpu_group (if world_size > 1)

        Non-entry ranks:
        - receive the object via broadcast (if world_size > 1)
        - otherwise (single-rank / no group) fall back to local from_processor_output

        Returns:
            MultimodalInputs | None
        """
        if raw_mm_inputs is None:
            return None

        group_world_size = 1
        try:
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and self.dp_tp_cpu_group is not None
            ):
                group_world_size = torch.distributed.get_world_size(
                    group=self.dp_tp_cpu_group
                )
        except Exception as e:
            logger.warning(
                f"Failed to get world size in mm_inputs handling with {e}, fallback to 1."
            )

        # In case tp size > 1, all the Scheduler TP ranks runs the duplicated computing
        # process in CPU which occupies the main thread CPU cycle. This computing logic
        # merely needs to be run on TP0 and be broadcast to other TP ranks.
        # Since the Scheduler is single-threaded, any large CPU cost will impact
        # handling of other messages. For example, CPU hits 99.9% can significantly
        # increase the CUDA kernel launch time.
        if self.dp_tp_group.rank_in_group == 0:
            # Only the entry rank materializes once from dict.
            image_inputs = MultimodalInputs.from_processor_output(raw_mm_inputs)
            # Broadcast to other TP ranks (use src=0 within the group).
            if group_world_size > 1:
                obj_list = [image_inputs]
                torch.distributed.broadcast_object_list(
                    obj_list,
                    src=self.dp_tp_group.first_rank,
                    group=self.dp_tp_cpu_group,
                )
                image_inputs = obj_list[0]
        else:
            # Non-entry ranks: receive if group size > 1; otherwise materialize locally.
            if group_world_size > 1:
                obj_list = [None]
                torch.distributed.broadcast_object_list(
                    obj_list,
                    src=self.dp_tp_group.first_rank,
                    group=self.dp_tp_cpu_group,
                )
                image_inputs = obj_list[0]
            else:
                image_inputs = MultimodalInputs.from_processor_output(raw_mm_inputs)

        return image_inputs

    def _get_multimodal_inputs(self, mm_inputs_dict):
        if self.server_args.enable_broadcast_mm_inputs_process:
            return self._process_and_broadcast_mm_inputs(mm_inputs_dict)
        else:
            return MultimodalInputs.from_processor_output(mm_inputs_dict)

    @staticmethod
    def _try_apply_padded_mm_input_ids(recv_req, req, image_inputs) -> bool:
        """setup origin_input_ids with trying to reuse existing MultimodalInputs.padded_input_ids first,
        if absent, call pad_input_ids_func"""
        padded_input_ids = image_inputs.padded_input_ids
        if padded_input_ids is None or recv_req.input_ids is None:
            return False

        recv_input_len = len(recv_req.input_ids)
        if len(padded_input_ids) != recv_input_len:
            return False

        prefix_len = len(req.origin_input_ids) - recv_input_len
        if prefix_len < 0:
            return False

        padded_input_ids = array("q", padded_input_ids)
        if prefix_len == 0:
            req.origin_input_ids = padded_input_ids
        else:
            req.origin_input_ids = req.origin_input_ids[:prefix_len] + padded_input_ids
        return True

    def _maybe_compute_mrope_positions(self, req) -> None:
        """Compute M-RoPE positions when they are missing (e.g. gRPC preprocessed path)."""
        if self._mm_processor is None:
            return
        mm = req.multimodal_inputs
        if mm is None or mm.mrope_positions is not None:
            return

        mrope_positions, mrope_position_delta = (
            self._mm_processor.compute_mrope_positions(
                req.origin_input_ids, mm.mm_items
            )
        )
        if mrope_positions is not None:
            mm.mrope_positions = mrope_positions
            mm.mrope_position_delta = mrope_position_delta

    def _maybe_clear_mm_inputs(self, batch: ScheduleBatch) -> None:
        for req in batch.reqs:
            if not req.finished() or not (mm_inputs := req.multimodal_inputs):
                continue
            # For session requests, keep mm_inputs for the next request
            if req.session:
                continue
            # For non-session requests, clear features and mm_inputs
            mm_inputs.release_features()
            req.multimodal_inputs = None

    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
    ):
        # The Rust PD router currently preserves extra_key but drops
        # sampling_params.custom_params.  Restore the validated lifecycle
        # envelope before constructing Req, then retain only the stable key for
        # radix matching across generations.
        try:
            agentic_envelope = unpack_agentic_extra_key(recv_req.extra_key)
        except ValueError as exc:
            logger.warning("Ignoring invalid AgenticKV envelope: %s", exc)
            agentic_envelope = None
        if agentic_envelope is not None:
            stable_extra_key, agentic_custom_params = agentic_envelope
            sampling_params = copy.copy(recv_req.sampling_params)
            existing_custom_params = dict(sampling_params.custom_params or {})
            existing_custom_params.update(agentic_custom_params)
            sampling_params.custom_params = existing_custom_params
            recv_req.sampling_params = sampling_params
            recv_req.extra_key = stable_extra_key

        # Route: normal request / session request / session-not-found
        session_id = (
            recv_req.session_params.id if recv_req.session_params is not None else None
        )

        if session_id is None:
            # Normal non-session request
            if recv_req.input_embeds is not None:
                # Generate fake input_ids based on the length of input_embeds
                seq_length = len(recv_req.input_embeds)
                recv_req.input_ids = array("q", [1]) * seq_length

            if recv_req.bootstrap_port is None:
                # Use default bootstrap port
                recv_req.bootstrap_port = self.server_args.disaggregation_bootstrap_port

            req = Req(
                recv_req.rid,
                recv_req.input_text,
                recv_req.input_ids,
                recv_req.sampling_params,
                return_logprob=recv_req.return_logprob,
                top_logprobs_num=recv_req.top_logprobs_num,
                token_ids_logprob=recv_req.token_ids_logprob,
                stream=recv_req.stream,
                lora_id=recv_req.lora_id,
                input_embeds=recv_req.input_embeds,
                positional_embed_overrides=recv_req.positional_embed_overrides,
                token_type_ids=recv_req.token_type_ids,
                custom_logit_processor=recv_req.custom_logit_processor,
                require_reasoning=recv_req.require_reasoning,
                return_hidden_states=recv_req.return_hidden_states,
                return_routed_experts=recv_req.return_routed_experts,
                routed_experts_start_len=recv_req.routed_experts_start_len,
                return_indexer_topk=recv_req.return_indexer_topk,
                eos_token_ids=self.model_config.hf_eos_token_id,
                bootstrap_host=recv_req.bootstrap_host,
                bootstrap_port=recv_req.bootstrap_port,
                bootstrap_room=recv_req.bootstrap_room,
                disagg_mode=self.disaggregation_mode,
                routed_dp_rank=recv_req.routed_dp_rank,
                disagg_prefill_dp_rank=recv_req.disagg_prefill_dp_rank,
                vocab_size=self.model_config.vocab_size,
                priority=recv_req.priority,
                metrics_collector=(
                    self.metrics_collector
                    if self.metrics_reporter.enable_metrics
                    else None
                ),
                extra_key=recv_req.extra_key,
                routing_key=recv_req.routing_key,
                http_worker_ipc=recv_req.http_worker_ipc,
                dllm_config=self.dllm_config,
                time_stats=recv_req.time_stats,
                multi_item_delimiter_indices=recv_req.multi_item_delimiter_indices,
            )
            req.tokenizer = self.tokenizer

            if self.disaggregation_mode != DisaggregationMode.NULL:
                # Invalid request for disaggregated mode
                if (
                    recv_req.bootstrap_room is None
                    and self.transfer_backend != TransferBackend.FAKE
                ):
                    error_msg = (
                        f"Invalid request: Disaggregated request received without "
                        f"bootstrap room id. {req.rid=}"
                    )
                    logger.error(error_msg)
                    recv_req.time_stats.trace_ctx.abort(
                        abort_info={"reason": error_msg}
                    )
                    prepare_abort(req, error_msg, status_code=HTTPStatus.BAD_REQUEST)
                    self.output_streamer.stream_output([req], req.return_logprob)
                    return

        elif (
            session_id in self.session_controller
            and not self.session_controller.get(session_id).close_on_finish
        ):
            # Session exists and is not closing: create request from session
            session = self.session_controller.get(session_id)
            req = session.create_req(
                recv_req,
                self.tokenizer,
                self.model_config.vocab_size,
                eos_token_ids=self.model_config.hf_eos_token_id,
            )
            # TODO: set trace context
            if self.metrics_reporter.enable_metrics:
                req.time_stats.set_metrics_collector(self.metrics_collector)
            if isinstance(req.finished_reason, FINISH_ABORT):
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return

        else:
            # Session not found, or session is closing
            if session_id in self.session_controller:
                error_msg = (
                    f"Invalid request: close was requested for session {session_id}"
                )
            else:
                error_msg = f"Invalid request: session id {session_id} does not exist"
            req = Req(
                recv_req.rid,
                recv_req.input_text,
                recv_req.input_ids,
                recv_req.sampling_params,
                vocab_size=self.model_config.vocab_size,
                http_worker_ipc=recv_req.http_worker_ipc,
            )
            req.tokenizer = self.tokenizer
            req.set_finish_with_abort(error_msg)
            self.init_req_max_new_tokens(req)
            self._add_request_to_queue(req)
            return

        if self.spec_algorithm.is_dflash():
            error_msg = validate_dflash_request(req, self.enable_overlap)
            if error_msg is not None:
                req.set_finish_with_abort(error_msg)
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return
        # Handle multimodal inputs
        if recv_req.mm_inputs is not None:
            image_inputs = self._get_multimodal_inputs(recv_req.mm_inputs)

            SessionController.adjust_mm_offsets(recv_req, req, image_inputs)

            # The following steps are already fast, execute locally on each rank.
            # Expand a single image token into multiple dummy tokens for receiving image embeddings.
            # The pad function is model-specific and can be None for some backends.
            if (
                not self._try_apply_padded_mm_input_ids(recv_req, req, image_inputs)
                and self.pad_input_ids_func
            ):
                req.origin_input_ids = array(
                    "q", self.pad_input_ids_func(req.origin_input_ids, image_inputs)
                )
            req.extend_image_inputs(image_inputs)
            self._maybe_compute_mrope_positions(req)

            if len(req.origin_input_ids) >= self.max_req_input_len:
                req.set_finish_with_abort(
                    error_msg=(
                        "Multimodal prompt is too long after expanding multimodal tokens. "
                        f"After expanding {len(req.origin_input_ids_unpadded)=} => {len(req.origin_input_ids)} >= {self.max_req_input_len}."
                    )
                )
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return

        # initialize before returning
        self.init_req_max_new_tokens(req)

        # Validate prompt length
        error_msg = validate_input_length(
            req,
            self.max_req_input_len,
            self.server_args.allow_auto_truncate,
        )
        if error_msg:
            req.set_finish_with_abort(error_msg)
            self._add_request_to_queue(req)
            return

        if not recv_req.return_logprob and recv_req.logprob_start_len != -1:
            # When return_logprob is False, logprob_start_len should be ignored
            recv_req.logprob_start_len = -1

        if recv_req.logprob_start_len == -1:
            if recv_req.return_logprob and recv_req.token_ids_logprob is None:
                # If logprob is required but neither token_ids_logprob nor logprob_start_len is
                # set, return the logprobs for output tokens by default
                req.logprob_start_len = len(req.origin_input_ids)
            elif req.is_prefill_only:
                # For prefill-only requests with logprob_start_len == -1, set logprob_start_len
                # beyond input sequence to skip input logprob computation entirely
                req.logprob_start_len = len(req.origin_input_ids)
            else:
                # If return_logprob is False, only the last token requires logprob computation
                req.logprob_start_len = -1
        else:
            req.logprob_start_len = recv_req.logprob_start_len

        if req.logprob_start_len > len(req.origin_input_ids):
            error_msg = f"{req.logprob_start_len=} is higher than the number of input tokens {len(req.origin_input_ids)=}. Please use a smaller logprob_start_len."
            req.logprob_start_len = -1
            req.set_finish_with_abort(error_msg)
            self._add_request_to_queue(req)
            return

        if recv_req.return_routed_experts:
            error_msg = None
            if recv_req.routed_experts_start_len < 0:
                error_msg = (
                    f"{recv_req.routed_experts_start_len=} is lower than 0. "
                    "Please use a non-negative routed_experts_start_len."
                )

            if recv_req.routed_experts_start_len > len(req.origin_input_ids):
                error_msg = (
                    f"{recv_req.routed_experts_start_len=} is higher than the "
                    f"number of input tokens {len(req.origin_input_ids)=}. Please "
                    f"use a smaller routed_experts_start_len."
                )

            if error_msg is not None:
                req.routed_experts_start_len = 0
                req.set_finish_with_abort(error_msg)
                self._add_request_to_queue(req)
                return

        added_to_grammar_queue = self.grammar_manager.process_req_with_grammar(req)
        if not added_to_grammar_queue:
            self._add_request_to_queue(req)

    def handle_batch_generate_request(
        self,
        recv_req: BatchTokenizedGenerateReqInput,
    ):
        """Handle optimized batch generate request."""
        logger.debug(f"Processing batch generate request with {len(recv_req)} requests")

        # Process each request in the batch
        for tokenized_req in recv_req:
            self.handle_generate_request(tokenized_req)

    def _prefetch_kvcache(self, req: Req):
        if self.enable_hicache_storage:
            agentic_manifest = getattr(req, "_agentic_kv_manifest", None)
            if (
                envs.SGLANG_AGENTIC_KV_CUSTOM_STORAGE_ONLY.get()
                and agentic_manifest is None
            ):
                return
            req.init_next_round_input(self.tree_cache, cow_mamba=False)
            last_host_node = req.last_host_node
            if last_host_node.backuped or last_host_node is self.tree_cache.root_node:
                last_hash = last_host_node.get_last_hash_value()
                matched_len = len(req.prefix_indices) + req.host_hit_length
                match_end = req._compute_max_prefix_len(
                    len(req.full_untruncated_fill_ids)
                )
                new_input_tokens = req.full_untruncated_fill_ids[matched_len:match_end]

                prefix_keys = (
                    last_host_node.get_prefix_hash_values(last_host_node.parent)
                    if self.tree_cache.hicache_storage_pass_prefix_keys
                    else None
                )
                agentic_expected_tokens = (
                    max(0, agentic_manifest.token_count - matched_len)
                    if agentic_manifest is not None
                    else None
                )
                if agentic_expected_tokens == 0:
                    return
                self.tree_cache.prefetch_from_storage(
                    req.rid,
                    last_host_node,
                    new_input_tokens,
                    last_hash,
                    prefix_keys,
                    getattr(req, "_agentic_kv_storage_namespace", None),
                    agentic_expected_tokens,
                    req.extra_key if agentic_manifest is not None else None,
                )

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

    def _agentic_release_restore_pin_after_admission(self, req: Req) -> None:
        """Drop transport's temporary pin after PrefillAdder owns the prefix."""

        for attr in (
            "_agentic_direct_parent_pin_node",
            "_agentic_kv_host_pin_node",
        ):
            node = getattr(req, attr, None)
            if node is None:
                continue
            self.tree_cache.dec_lock_ref(node)
            delattr(req, attr)

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
            if chunked_req is not None and (
                private_suffix is None or private_suffix.numel() == 0
            ):
                # ``fill_len`` is the prefix length through the chunk that
                # just ran;
                # ``origin_input_ids + output_ids`` is the complete logical
                # prompt.  Page-round the unprocessed suffix exactly as the
                # allocator will do on subsequent chunks.
                remaining = max(
                    0,
                    len(chunked_req.origin_input_ids)
                    + len(chunked_req.output_ids)
                    - int(
                        getattr(
                            chunked_req,
                            "fill_len",
                            len(getattr(chunked_req, "fill_ids", ())),
                        )
                    ),
                )
                reserve_tokens = (
                    (remaining + self.page_size - 1) // self.page_size
                ) * self.page_size
            broker.service(
                self.token_to_kv_pool_allocator,
                reserve_tokens=reserve_tokens,
            )

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
                submit_reverse_receive(
                    receiver,
                    workset_lease,
                    runtime.manager.kv_args.state_types,
                )
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
                if self.tp_size > 1:
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
        self.agentic_p_workset_broker.request_release(
            entry.request.snapshot_id, entry.workset_lease
        )
        if release_claim:
            try:
                current = snapshot_store.load(entry.request, require_ready=False)
                if (
                    current is not None
                    and current.state
                    in {
                        SnapshotState.DIRECT_LOADING,
                        SnapshotState.P_RECEIVED,
                    }
                    and current.claim_id == entry.claim_id
                ):
                    if current.state is SnapshotState.P_RECEIVED:
                        snapshot_store.release_received_direct(current, entry.claim_id)
                    else:
                        snapshot_store.release_direct_claim(current, entry.claim_id)
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
                if reason == "tp_group_abort":
                    # The native TP abort command already rolled back this
                    # rank's prepared Radix branch before entering this
                    # helper.  If teardown had to wait for the NIXL fence,
                    # that command returned before it could publish status 6.
                    # Publish the deferred rollback ACK here once transport
                    # is terminal; leaving status -1 makes TP0 wait forever
                    # and strands the authoritative D snapshot.
                    if failed is not None:
                        failed.discard(entry.request.snapshot_id)
                    rolled_back = getattr(
                        self, "agentic_tp_direct_local_rolled_back", None
                    )
                    if rolled_back is None:
                        rolled_back = set()
                        self.agentic_tp_direct_local_rolled_back = rolled_back
                    rolled_back.add(entry.request.snapshot_id)
                    mailbox = getattr(self, "agentic_tp_direct_mailbox", None)
                    if mailbox is not None:
                        # ``-1`` is intentionally terminal for ordinary
                        # progress writers.  This is the ordered native-abort
                        # completion boundary, so it must explicitly replace
                        # that failure marker with the rollback ACK.
                        mailbox.publish_local(entry.request.snapshot_id, 6)
                elif failed is not None:
                    failed.add(entry.request.snapshot_id)
        logger.warning(
            "AgenticKV early_direct_drop snapshot=%s reason=%s",
            entry.request.snapshot_id,
            reason,
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
                    # Router can retarget the same arrival marker after D
                    # crosses Direct -> Slow and chooses another P domain.
                    # This worker may already have published an allocator
                    # intent from the previous, locally targeted marker.  A
                    # scheduler-safe grant can land between those two marker
                    # observations, so dropping only the queue item strands
                    # either the intent or its newly active lease forever.
                    # Cancel the exact Direct owner on the old P; the Host
                    # restore on the new P has an independent Slow owner.
                    self.agentic_p_workset_broker.cancel_unstarted(
                        request.snapshot_id,
                        owner=AgenticPWorksetLeaseBroker.direct_owner(
                            request.snapshot_id
                        ),
                    )
                    continue
            else:
                # Preserve the established 1P behavior: its arrival markers
                # are untargeted and require no route-resolution handshake.
                target_domain = None

            broker = self.agentic_p_workset_broker
            workset_owner = AgenticPWorksetLeaseBroker.direct_owner(request.snapshot_id)
            if broker.owner_is_superseded(request.snapshot_id, owner=workset_owner):
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
                    owner=AgenticPWorksetLeaseBroker.direct_owner(request.snapshot_id),
                )
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
            authoritative_manifest = snapshot_store.load(request, require_ready=False)
            authoritative_eligible_states = (
                {SnapshotState.DIRECT_READY, SnapshotState.DIRECT_LOADING}
                if tp_size > 1 and tp_rank != 0
                else {SnapshotState.DIRECT_READY}
            )
            if (
                authoritative_manifest is None
                or authoritative_manifest.state not in authoritative_eligible_states
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
                    if manifest.state not in {
                        SnapshotState.DIRECT_READY,
                        SnapshotState.DIRECT_LOADING,
                    }:
                        broker.cancel_unstarted(
                            request.snapshot_id,
                            owner=workset_owner,
                        )
                        continue
                    with poll_lock:
                        queue.append((request, payload, manifest))
                        pending.add(request.snapshot_id)
                    continue
                if int(receipt) < 0 or int(receipt) >= 4:
                    broker.request_release(request.snapshot_id, workset_lease)
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
        self._agentic_admit_queued_direct_receives(
            snapshot_store, direct_timeout, poll_lock
        )
        if self.tp_size > 1:
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
                if now - entry.completed_at >= bind_timeout and not getattr(
                    entry, "bind_wait_warned", False
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
                    state_digest = debug_mamba_digest(
                        self.req_to_token_pool,
                        state_indices_for_workset(
                            entry.workset_lease,
                            self.agentic_direct_runtime.manager.kv_args.state_types,
                        ),
                    )
                    if state_digest is not None:
                        logger.info(
                            "AgenticKV p_received_state_digest snapshot=%s "
                            "rank=%d digest=%s",
                            snapshot_id,
                            self.tp_rank,
                            state_digest,
                        )
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

        if self.tp_size > 1:
            self._agentic_commit_tp_direct_groups(snapshot_store)

        # Retain short-lived terminal ids only to avoid repeatedly reopening a
        # marker while Decode is about to remove it.
        with poll_lock:
            for snapshot_id, terminal_at in tuple(
                self.agentic_early_direct_terminal.items()
            ):
                if now - terminal_at >= 10.0:
                    self.agentic_early_direct_terminal.pop(snapshot_id, None)

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
        the mailbox proves all ranks acknowledged status 6; only then may TP0
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
                snapshot_store.release_received_direct(current, expected_claim_id)
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
                    mailbox.publish_local_progress(snapshot_id, 6)
                elif snapshot_id in getattr(self, "agentic_tp_direct_local_failed", ()):
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
            group_status = mailbox.group_status(snapshot_id)
            if group_status is None:
                continue
            group_status = int(group_status)
            receipt = mailbox.receipt(snapshot_id)
            if receipt is not None and int(receipt) < 0:
                if int(receipt) <= -2:
                    continue
                if group_status >= 6:
                    Scheduler._agentic_abort_tp_direct_grant(
                        self,
                        active_item[0],
                        snapshot_store,
                        reason="all_ranks_rolled_back",
                        rolled_back=True,
                    )
                continue
            if group_status < 0:
                Scheduler._agentic_abort_tp_direct_grant(
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
                current = snapshot_store.load(active_item[0], require_ready=False)
                if current is not None and current.state is SnapshotState.P_RECEIVED:
                    try:
                        snapshot_store.commit_direct_bound(current, current.claim_id)
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
            else debug_kv_digest(
                direct_runtime.kv_pool,
                entry.device_indices[: entry.manifest.token_count],
            )
        )
        if restored_digest is not None:
            logger.info(
                "AgenticKV p_restored_digest snapshot=%s digest=%s",
                request.snapshot_id,
                restored_digest,
            )
        restored_pages = (
            None
            if direct_runtime is None
            else debug_kv_page_digests(
                direct_runtime.kv_pool,
                entry.device_indices[: entry.manifest.token_count],
            )
        )
        if restored_pages is not None:
            logger.info(
                "AgenticKV p_restored_page_digests snapshot=%s rank=%d pages=%s",
                request.snapshot_id,
                self.tp_rank,
                restored_pages,
            )

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
                    "AgenticKV permanent Direct failure commit retry " "snapshot=%s",
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
                    entry,
                    req,
                    reason=(
                        "workset_marker_underestimated:"
                        f"reserved={entry.workset_lease.prompt_tokens}:"
                        f"actual={len(req.origin_input_ids)}"
                    ),
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
        state_value = (
            entry.workset_lease.state_device_indices[0].clone()
            if entry.workset_lease.state_device_indices
            else None
        )
        try:
            result = self.tree_cache.insert(
                InsertParams(
                    key=RadixKey(parent_tokens, req.extra_key),
                    value=entry.device_indices,
                    mamba_value=state_value,
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
            request.snapshot_id,
            entry.workset_lease,
            state_donated_to_radix=(state_value is not None and not result.mamba_exist),
            state_duplicate=(state_value is not None and result.mamba_exist),
        )
        # insert() makes the restored parent visible to the Radix LRU.  Pin the
        # exact request-generation before returning to the queue.  Native
        # Prefill acquires its ordinary request lock first and only then drops
        # this temporary pin, so there is no evictable gap between ownerships.
        try:
            self.agentic_p_workset_broker.attach_runtime_state_for_bind(
                request.snapshot_id, req, entry.workset_lease
            )
            parent_match = self.tree_cache.match_prefix(
                MatchPrefixParams(
                    key=RadixKey(parent_tokens, req.extra_key),
                    req=req,
                    cow_mamba=self.tree_cache.supports_mamba(),
                )
            )
            if self.tree_cache.supports_mamba():
                self.agentic_p_workset_broker.stage_runtime_checkpoint_cow_for_bind(
                    request.snapshot_id, req, entry.workset_lease
                )
            if len(parent_match.device_indices) != len(parent_tokens):
                raise RuntimeError(
                    "Early Direct parent disappeared before request protection"
                )
            self.tree_cache.inc_lock_ref(parent_match.last_device_node)
            req._agentic_direct_parent_pin_node = parent_match.last_device_node
            req._agentic_direct_parent_token_count = len(parent_tokens)
            entry.radix_prepared = True
            entry.existing_tokens = (
                0 if self.tree_cache.supports_mamba() else int(result.prefix_len)
            )
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
            submit_reverse_receive(
                receiver,
                workset_lease,
                runtime.manager.kv_args.state_types,
            )
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
            state_value = (
                workset_lease.state_device_indices[0].clone()
                if workset_lease.state_device_indices
                else None
            )
            try:
                result = self.tree_cache.insert(
                    InsertParams(
                        key=RadixKey(keys, req.extra_key),
                        value=device_indices,
                        mamba_value=state_value,
                        priority=getattr(req, "priority", 0) or 0,
                    )
                )
                inserted = True
                self.agentic_p_workset_broker.commit_parent_bound(
                    manifest.snapshot_id,
                    workset_lease,
                    state_donated_to_radix=(
                        state_value is not None and not result.mamba_exist
                    ),
                    state_duplicate=(state_value is not None and result.mamba_exist),
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
                    MatchPrefixParams(
                        key=RadixKey(keys, req.extra_key),
                        req=req,
                        cow_mamba=self.tree_cache.supports_mamba(),
                    )
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
                    completed = snapshot_store.commit_direct_bound(completed, claim_id)
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
                    and host_action
                    in {"prepare", "start", "bind", "commit", "finalize"}
                )
                allow_start_io = bool(
                    allow_start_io
                    and host_action is not None
                    and host_action in {"start", "bind", "commit", "finalize"}
                )
                allow_bind_io = bool(
                    host_action is not None
                    and host_action in {"bind", "commit", "finalize"}
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
                # D retains the complete attention+Mamba snapshot until its
                # independent worker commits it to Shared Host.  Do not claim
                # Direct in this pd-compatible ablation.
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
            if time.monotonic() - started_at >= timeout and not getattr(
                req, "_agentic_parent_wait_warned", False
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
            if (
                manifest.state is SnapshotState.FAILED
                and manifest.failure_reason == "p_direct_capacity"
            ):
                # This experimental fail-open path deliberately discards the
                # parent KV when P cannot admit a Direct receive.  The child
                # must compete with fresh work, below Direct and slow recovery,
                # rather than retaining its former fast-parent priority.
                req._agentic_kv_queue_class = "new"
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
        if time.monotonic() - started_at >= timeout and not getattr(
            req, "_agentic_parent_missing_warned", False
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

    def _agentic_track_waiter(self, req: Req, started_at: float) -> None:
        """Index one metadata-only P request for edge-triggered progress."""

        self.agentic_kv_waiting_by_rid[req.rid] = (req, float(started_at))
        metadata = AgenticRequestMetadata.from_req(req)
        parent = metadata.parent if metadata is not None else None
        if parent is not None:
            self.agentic_kv_waiting_by_parent.setdefault(parent.snapshot_id, {})[
                req.rid
            ] = req
        self._agentic_enqueue_progress(req)

    def _agentic_forget_waiter(self, req: Req) -> None:
        """Remove indexes immediately; compact the compatibility list lazily."""

        if self.agentic_kv_waiting_by_rid.pop(req.rid, None) is None:
            return
        metadata = AgenticRequestMetadata.from_req(req)
        parent = metadata.parent if metadata is not None else None
        if parent is not None:
            waiters = self.agentic_kv_waiting_by_parent.get(parent.snapshot_id)
            if waiters is not None:
                waiters.pop(req.rid, None)
                if not waiters:
                    self.agentic_kv_waiting_by_parent.pop(parent.snapshot_id, None)
        self.agentic_kv_progress_enqueued.discard(req.rid)
        self.agentic_kv_retry_deadlines.pop(req.rid, None)
        self.agentic_kv_waiting_tombstones += 1

    def _agentic_enqueue_progress(self, req: Req) -> None:
        """Publish one idempotent runnable/control edge to the scheduler."""

        if (
            req.rid not in self.agentic_kv_waiting_by_rid
            or req.rid in self.agentic_kv_progress_enqueued
        ):
            return
        queue_class = self._agentic_queue_class(req)
        self.agentic_kv_progress_queues[queue_class].append(req.rid)
        self.agentic_kv_progress_enqueued.add(req.rid)

    def _agentic_enqueue_snapshot_waiters(self, snapshot_id: str) -> None:
        for req in tuple(
            self.agentic_kv_waiting_by_parent.get(str(snapshot_id), {}).values()
        ):
            self._agentic_enqueue_progress(req)

    def _agentic_schedule_retry(self, req: Req, delay_seconds: float = 0.05) -> None:
        """Backstop lost filesystem edges without repeatedly scanning waiters."""

        if req.rid not in self.agentic_kv_waiting_by_rid:
            return
        deadline = time.monotonic() + max(0.001, float(delay_seconds))
        previous = self.agentic_kv_retry_deadlines.get(req.rid)
        if previous is not None and previous <= deadline:
            return
        self.agentic_kv_retry_sequence += 1
        self.agentic_kv_retry_deadlines[req.rid] = deadline
        heapq.heappush(
            self.agentic_kv_retry_heap,
            (deadline, self.agentic_kv_retry_sequence, req.rid),
        )

    def _agentic_ingest_progress_events(self) -> None:
        """Convert background completion edges into exact waiter wakeups."""

        completions = getattr(self, "agentic_early_direct_completion_queue", None)
        if completions is not None:
            with getattr(self, "agentic_early_direct_poll_lock", nullcontext()):
                direct = tuple(completions)
                completions.clear()
            for snapshot_id in direct:
                self._agentic_enqueue_snapshot_waiters(snapshot_id)

        host_staging = getattr(self, "agentic_host_staging_manager", None)
        if host_staging is not None:
            drain_events = getattr(host_staging, "drain_scheduler_events", None)
            if drain_events is not None:
                for _, snapshot_id in drain_events():
                    self._agentic_enqueue_snapshot_waiters(snapshot_id)

        broker = getattr(self, "agentic_p_workset_broker", None)
        if broker is not None:
            drain_grants = getattr(broker, "drain_grant_events", None)
            if drain_grants is not None:
                for snapshot_id in drain_grants():
                    self._agentic_enqueue_snapshot_waiters(snapshot_id)

        now = time.monotonic()
        while self.agentic_kv_retry_heap:
            deadline, _, rid = self.agentic_kv_retry_heap[0]
            if deadline > now:
                break
            heapq.heappop(self.agentic_kv_retry_heap)
            if self.agentic_kv_retry_deadlines.get(rid) != deadline:
                continue
            self.agentic_kv_retry_deadlines.pop(rid, None)
            waiter = self.agentic_kv_waiting_by_rid.get(rid)
            if waiter is not None:
                self._agentic_enqueue_progress(waiter[0])

    def _agentic_compact_waiting_queue(self, *, force: bool = False) -> None:
        """Keep legacy diagnostics/TP iteration accurate off the hot path."""

        tombstones = self.agentic_kv_waiting_tombstones
        if (
            not force
            and tombstones < 64
            and tombstones * 2 < len(self.agentic_kv_waiting_queue)
        ):
            return
        if not tombstones:
            return
        live = self.agentic_kv_waiting_by_rid
        self.agentic_kv_waiting_queue = [
            entry
            for entry in self.agentic_kv_waiting_queue
            if (
                entry[0].rid in live
                and live[entry[0].rid][0] is entry[0]
                and live[entry[0].rid][1] == entry[1]
            )
        ]
        self.agentic_kv_waiting_tombstones = 0

    @staticmethod
    def _agentic_slow_aging_seconds() -> float:
        try:
            return max(
                0.0,
                float(os.environ.get("SGLANG_AGENTIC_KV_SLOW_AGING_SECONDS", "2")),
            )
        except ValueError:
            logger.exception("Invalid agentic slow aging setting")
            raise

    @staticmethod
    def _agentic_new_aging_seconds() -> float:
        try:
            return max(
                0.0,
                float(os.environ.get("SGLANG_AGENTIC_KV_NEW_AGING_SECONDS", "10")),
            )
        except ValueError:
            logger.exception("Invalid agentic new-request aging setting")
            raise

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
        return None

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

    _AGENTIC_TP_CONTROL_KEY = "__sglang_agentic_tp_admission_v1__"

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
            if request.snapshot_id in getattr(
                self, "agentic_tp_host_local_admitted", ()
            ):
                local_status = 5
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
                if getattr(req, "_agentic_tp_host_handoff_ready", False):
                    local_status = max(local_status, 4)
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

        visible = tuple(getattr(self, "agentic_tp_workset_retire_visible", ()))
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
        if int(group_status) >= 5:
            return "clear"
        if int(group_status) >= 4:
            return "finalize"
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
            release_mailbox = self.agentic_tp_decode_release_mailbox
            snapshot_id = getattr(self, "agentic_tp_decode_release_active", None)
            if snapshot_id is not None and release_mailbox.group_status(snapshot_id) == 1:
                release_mailbox.clear_group(snapshot_id)
                self.agentic_tp_decode_release_active = None
                snapshot_id = None
            if snapshot_id is None and offload_manager is not None:
                snapshot_id = offload_manager.tp_pending_release_snapshot()
                if snapshot_id is not None:
                    snapshot_id = str(snapshot_id)
                    self.agentic_tp_decode_release_active = snapshot_id
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
                        p_ready_dir
                        and decode_req.req.bootstrap_host != FAKE_BOOTSTRAP_HOST
                        and not getattr(decode_req, "_async_p_ready", False)
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
        workset_plan_epoch = int(getattr(self, "_agentic_tp_workset_plan_epoch", 0)) + 1
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
        if workset_broker is not None and hasattr(workset_broker, "prepare_tp_control"):
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
        for req in recv_reqs:
            if isinstance(req, dict) and req.get(self._AGENTIC_TP_CONTROL_KEY):
                control = req
            else:
                ordinary.append(req)
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
                if not self.agentic_p_workset_broker.commit_tp_handoff(snapshot_id):
                    raise RuntimeError(
                        "TP workset handoff commit reached a non-handed local "
                        f"lease for {snapshot_id}"
                    )
            retire_visible = getattr(self, "agentic_tp_workset_retire_visible", None)
            if retire_commands and retire_visible is None:
                retire_visible = set()
                self.agentic_tp_workset_retire_visible = retire_visible
            retire_mailbox = getattr(self, "agentic_tp_workset_retire_mailbox", None)
            for command in retire_commands:
                snapshot_id = str(command["snapshot"])
                action = str(command["action"])
                if action == "prepare":
                    self.agentic_p_workset_broker.prepare_tp_retire(snapshot_id)
                    retire_visible.add(snapshot_id)
                elif action == "commit":
                    if not self.agentic_p_workset_broker.commit_tp_retire(snapshot_id):
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
                    raise RuntimeError(f"unknown TP workset retire action {action}")
        decode_release_snapshot = control.get("decode_release_snapshot")
        if decode_release_snapshot is not None:
            offload_manager = getattr(self, "decode_offload_manager", None)
            if offload_manager is None:
                raise RuntimeError("TP Decode release lost its offload manager")
            decode_release_snapshot = str(decode_release_snapshot)
            release_mailbox = self.agentic_tp_decode_release_mailbox
            if release_mailbox.local_status(decode_release_snapshot) != 1:
                if offload_manager.commit_tp_release(decode_release_snapshot):
                    release_mailbox.publish_local(decode_release_snapshot, 1)
        if self.disaggregation_mode is DisaggregationMode.DECODE:
            offload_manager = getattr(self, "decode_offload_manager", None)
            if offload_manager is not None:
                apply_commands = getattr(
                    offload_manager, "apply_tp_candidate_commands", None
                )
                if apply_commands is not None:
                    apply_commands(control.get("decode_agentic_commands", ()))
            Scheduler._agentic_tp_latch_decode_admit_keys(
                self,
                control.get("decode_admit_keys", ())
            )
            transfer_keys = control.get("decode_transfer_keys")
            if transfer_keys is None:
                transfer_rid = control.get("decode_transfer_rid")
                transfer_room = control.get("decode_transfer_room")
                transfer_keys = (
                    []
                    if transfer_rid is None or transfer_room is None
                    else [(transfer_rid, transfer_room)]
                )
            Scheduler._agentic_tp_latch_decode_transfer_control(
                self,
                transfer_keys,
                control.get("decode_transfer_statuses", ()),
            )
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
                    rolled_back = getattr(
                        self, "agentic_tp_direct_local_rolled_back", None
                    )
                    if rolled_back is None:
                        rolled_back = set()
                        self.agentic_tp_direct_local_rolled_back = rolled_back
                    rolled_back.add(snapshot_id)
                    self.agentic_tp_direct_local_failed.discard(snapshot_id)
                    mailbox = getattr(self, "agentic_tp_direct_mailbox", None)
                    if mailbox is not None:
                        # Native rollback is complete and the receiver is no
                        # longer transport-owned.  Override the earlier -1
                        # failure marker with the ordered rollback ACK.
                        mailbox.publish_local(snapshot_id, 6)
                    # TP0 returns P_RECEIVED ownership only after every rank
                    # has published this rollback ACK.  Keep the command
                    # active until the background worker publishes receipt -2.
                    continue
                with direct_poll_lock:
                    self.agentic_tp_direct_local_admitted.discard(snapshot_id)
                    self.agentic_tp_direct_local_failed.discard(snapshot_id)
                    getattr(self, "agentic_tp_direct_local_rolled_back", set()).discard(
                        snapshot_id
                    )
                    if active_direct.get(snapshot_id) is active_item:
                        active_direct.pop(snapshot_id, None)
                    group_status.pop(snapshot_id, None)
                    mailbox = getattr(self, "agentic_tp_direct_mailbox", None)
                    if mailbox is not None:
                        mailbox.clear_local(snapshot_id)
                        if self.tp_rank == 0:
                            mailbox.clear_group(snapshot_id)
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
        commit_snapshots = set()
        finalize_snapshots = set()
        mailbox = getattr(self, "agentic_tp_host_mailbox", None)
        for command in host_commands:
            host_snapshot = str(command["snapshot"])
            host_action = command.get("action")
            if host_action == "clear":
                self.agentic_tp_host_local_admitted.discard(host_snapshot)
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
                commit_snapshots.add(host_snapshot)
            elif host_action == "finalize":
                finalize_snapshots.add(host_snapshot)
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
        self._agentic_tp_host_commit_snapshots = commit_snapshots
        self._agentic_tp_host_commit_snapshot = (
            None if not commit_snapshots else next(iter(commit_snapshots))
        )
        host_staging = getattr(self, "agentic_host_staging_manager", None)
        if host_staging is not None:
            host_staging.tp_host_commit_snapshots = commit_snapshots
            host_staging.tp_host_commit_snapshot = self._agentic_tp_host_commit_snapshot
            host_staging.tp_host_finalize_snapshots = finalize_snapshots
        host_timeout_snapshot = control.get("host_timeout_snapshot")
        self._agentic_tp_host_timeout_snapshot = (
            None if host_timeout_snapshot is None else str(host_timeout_snapshot)
        )
        return ordinary

    def _agentic_tp_latch_decode_admit_keys(self, keys) -> None:
        """Latch TP0 admission commands until this rank applies them.

        Native scheduler broadcasts can advance faster than a follower's
        disaggregation polling interval.  Replacing the previous command on
        every control epoch could therefore let TP0 allocate a destination
        while TP1 missed that one-epoch command forever.  Admission is a
        rank-local physical transition, so keep each ordered command until
        ``pop_preallocated`` has applied it on this rank.
        """

        current = list(getattr(self, "_agentic_tp_decode_admit_keys", ()))
        seen = set(current)
        for rid, room in keys:
            key = (str(rid), int(room))
            if key not in seen:
                current.append(key)
                seen.add(key)
        self._agentic_tp_decode_admit_keys = current

    def _agentic_tp_complete_decode_admit_keys(self, decode_reqs) -> None:
        """Retire only admission commands physically applied by this rank."""

        if not decode_reqs:
            return
        completed = {
            (str(decode_req.req.rid), int(decode_req.req.bootstrap_room))
            for decode_req in decode_reqs
        }
        self._agentic_tp_decode_admit_keys = [
            key
            for key in getattr(self, "_agentic_tp_decode_admit_keys", ())
            if key not in completed
        ]

    def _agentic_tp_latch_decode_transfer_control(self, keys, statuses) -> None:
        """Keep transfer commands/status until the local receiver commits."""

        current = list(getattr(self, "_agentic_tp_decode_transfer_keys", ()))
        status_map = dict(
            getattr(self, "_agentic_tp_decode_transfer_group_status", {})
        )
        seen = set(current)
        for (rid, room), status in zip(keys, statuses):
            key = (str(rid), int(room))
            if key not in seen:
                current.append(key)
                seen.add(key)
            old_status = status_map.get(key)
            if old_status not in (int(KVPoll.Success), int(KVPoll.Failed)):
                status_map[key] = int(status)
        self._agentic_tp_decode_transfer_keys = current
        self._agentic_tp_decode_transfer_group_status = status_map

    def _agentic_tp_reconcile_decode_transfer_keys(self, transfer_queue) -> None:
        """Drop latched terminals only after this rank removed its receiver."""

        live = {
            (str(entry.req.rid), int(entry.req.bootstrap_room))
            for entry in transfer_queue.queue
        }
        self._agentic_tp_decode_transfer_keys = [
            key
            for key in getattr(self, "_agentic_tp_decode_transfer_keys", ())
            if key in live
        ]
        statuses = getattr(self, "_agentic_tp_decode_transfer_group_status", {})
        self._agentic_tp_decode_transfer_group_status = {
            key: status for key, status in statuses.items() if key in live
        }

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

    def _drain_agentic_kv_waiting_queue_tp1(self) -> None:
        """Consume edge-triggered Direct/Slow readiness without queue scans."""

        self._agentic_ingest_progress_events()
        try:
            admission_batch = max(
                1, int(os.environ.get("SGLANG_AGENTIC_KV_ADMISSION_BATCH", "8"))
            )
        except ValueError:
            logger.exception("Invalid agentic KV admission setting")
            raise

        def pop_waiter(queue_class: str):
            """Pop one live edge, repairing a stale priority classification."""

            progress = self.agentic_kv_progress_queues[queue_class]
            while progress:
                rid = progress.popleft()
                self.agentic_kv_progress_enqueued.discard(rid)
                waiter = self.agentic_kv_waiting_by_rid.get(rid)
                if waiter is None:
                    continue
                if self._agentic_queue_class(waiter[0]) != queue_class:
                    self._agentic_enqueue_progress(waiter[0])
                    continue
                return waiter
            return None

        def oldest_wait_age(queue_class: str) -> Optional[float]:
            """Read only the queue head; never rescan all metadata waiters."""

            progress = self.agentic_kv_progress_queues[queue_class]
            while progress:
                rid = progress[0]
                waiter = self.agentic_kv_waiting_by_rid.get(rid)
                if waiter is None:
                    progress.popleft()
                    self.agentic_kv_progress_enqueued.discard(rid)
                    continue
                if self._agentic_queue_class(waiter[0]) != queue_class:
                    progress.popleft()
                    self.agentic_kv_progress_enqueued.discard(rid)
                    self._agentic_enqueue_progress(waiter[0])
                    continue
                return max(0.0, time.monotonic() - waiter[1])
            return None

        # Direct remains the normal first choice. Once Slow recovery or fresh
        # work crosses its existing aging bound, reserve one slot before the
        # usual priority pass. This keeps the path O(1) and prevents a steady
        # Direct stream from starving Shared-Arena recovery indefinitely.
        promoted = []
        slow_age = oldest_wait_age("slow")
        if slow_age is not None and slow_age >= self._agentic_slow_aging_seconds():
            promoted.append("slow")
        new_age = oldest_wait_age("new")
        if new_age is not None and new_age >= self._agentic_new_aging_seconds():
            promoted.append("new")

        processed = 0

        def consume_one(queue_class: str) -> bool:
            nonlocal processed
            waiter = pop_waiter(queue_class)
            if waiter is None:
                return False
            req, started_at = waiter
            processed += 1
            try:
                deferred = self._agentic_should_defer(
                    req, started_at, allow_start_io=True
                )
            except (SnapshotNotReadyError, SnapshotLifecycleError):
                deferred = True
            if deferred:
                host_staging = getattr(self, "agentic_host_staging_manager", None)
                # Slow DMA and early Direct publish completion edges. A
                # legacy request-owned receiver still needs polling, while
                # missing/transitioning manifests get a low-rate timer
                # backstop in case an external file event is lost.
                if getattr(req, "_agentic_direct_receiver", None) is not None:
                    self._agentic_schedule_retry(req, 0.005)
                elif host_staging is not None and req.rid in host_staging.loads:
                    load = host_staging.loads[req.rid]
                    if (
                        load.get("ledger_prepare_pending")
                        or load.get("io_complete")
                        or load.get("io_error") is not None
                    ):
                        # Pure DMA wait is background-owned and emits an edge.
                        # Ledger prepare/retry, Radix bind, handoff and failure
                        # cleanup are scheduler-owned idempotent boundaries;
                        # keep retrying those without expecting another DMA
                        # completion notification.
                        self._agentic_schedule_retry(req, 0.005)
                else:
                    self._agentic_schedule_retry(req, 0.05)
                return True

            req._agentic_kv_wait_enqueued = False
            self._agentic_forget_waiter(req)
            self._agentic_publish_p_scheduled(req)
            self._add_request_to_queue(req)
            return True

        for queue_class in promoted:
            if processed >= admission_batch:
                break
            consume_one(queue_class)

        for queue_class in ("fast", "slow", "new"):
            while processed < admission_batch and consume_one(queue_class):
                pass
        self._agentic_compact_waiting_queue()

    def _drain_agentic_kv_waiting_queue(self) -> None:
        """Progress active KV I/O and admit pending work in arrival order.

        A request in this queue owns metadata only.  Each scheduler iteration
        first polls already-started transfers. New Direct, Slow, and initial
        requests then share one arrival-ordered compute-admission queue; their
        I/O engines and credits remain independent. A small admission batch
        amortizes scheduler ticks that contain long Prefill kernels.
        """
        if getattr(self, "tp_size", 1) == 1:
            self._drain_agentic_kv_waiting_queue_tp1()
            return

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

        active = []
        fast = []
        slow = []
        new = []
        for entry in self.agentic_kv_waiting_queue:
            req = entry[0]
            if self._agentic_io_active(req):
                active.append(entry)
            elif self._agentic_queue_class(req) == "new":
                new.append(entry)
            elif self._agentic_queue_class(req) == "slow":
                slow.append(entry)
            else:
                fast.append(entry)

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
            for entry in active + fast + slow + new:
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
            inactive = sorted(fast + slow + new, key=lambda entry: entry[1])
            selected = active + inactive[:scan_limit]
            untouched = inactive[scan_limit:]
        still_waiting = []
        new_io_started = 0
        newly_admitted = 0
        # A selected request may overlap its receive/load with the current
        # Prefill batch, but requests that have not been selected remain
        # metadata-only.  This is a global cap, not a per-tick cap: an active
        # receive from an earlier scheduler iteration consumes the slot.
        active_direct = sum(self._agentic_io_kind(req) == "direct" for req, _ in active)
        active_slow = sum(self._agentic_io_kind(req) == "slow" for req, _ in active)
        direct_starts_left = max(0, direct_io_cap - active_direct)
        slow_starts_left = max(0, slow_io_cap - active_slow)
        for req, started_at in selected:
            previous_kind = self._agentic_io_kind(req)
            was_active = previous_kind is not None
            if not was_active and newly_admitted >= admission_batch:
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
        # Put unscanned entries first so the next bounded pass starts there.
        self.agentic_kv_waiting_queue = untouched + still_waiting

    def _prioritize_agentic_prefill_ready(self) -> None:
        """Prioritize requests that can make progress with their own KV pages.

        ``fast`` describes where a parent snapshot came from, not whether the
        request is currently runnable.  A failed Direct parent may retain the
        fast label while requiring ordinary KV for full recomputation.  It
        must never head-of-line block a handed Direct/Slow workset whose
        suffix pages are already owned by that exact request.
        """

        combined = self.waiting_queue
        if getattr(self, "tp_size", 1) > 1:
            # A TP group may observe Direct/Slow shard completion in a
            # different wall-clock order.  A handed workset, however, is a
            # group-committed ownership fact: both ranks have already bound
            # the same parent and own the same suffix pages.  Put those
            # requests ahead of ordinary allocation-dependent work, then use
            # the rank-independent request order assigned at native ingress.
            # Without this first key, older fresh requests can exhaust the
            # ordinary allocator and head-of-line block the exact worksets
            # already occupying that allocator.  Never sort on mutable local
            # fast/slow completion state.
            def tp_order(req):
                workset_backed = bool(
                    getattr(req, "_agentic_workset_backed", False)
                    and getattr(req, "_agentic_workset_suffix_indices", None)
                    is not None
                )
                sequence = getattr(req, "_agentic_tp_prefill_sequence", None)
                if sequence is None:
                    # Compatibility for requests created by out-of-tree code.
                    # Production agentic requests always have a sequence; rid
                    # remains a deterministic fail-safe rather than a local
                    # timestamp or cache-completion order.
                    return (
                        0 if workset_backed else 1,
                        int(getattr(req, "_agentic_tp_prefill_priority", 1)),
                        1,
                        0,
                        str(req.rid),
                    )
                return (
                    0 if workset_backed else 1,
                    int(getattr(req, "_agentic_tp_prefill_priority", 1)),
                    0,
                    int(sequence),
                    str(req.rid),
                )

            self.waiting_queue = sorted(combined, key=tp_order)
            return

        owned = [
            req
            for req in combined
            if (
                getattr(req, "_agentic_workset_backed", False)
                and getattr(req, "_agentic_workset_suffix_indices", None) is not None
                and req._agentic_workset_suffix_indices.numel() > 0
            )
        ]
        owned_ids = {id(req) for req in owned}
        fast = [
            req
            for req in combined
            if id(req) not in owned_ids
            if getattr(req, "_agentic_kv_queue_class", None) == "fast"
        ]
        slow = [
            req
            for req in combined
            if id(req) not in owned_ids
            if getattr(req, "_agentic_kv_queue_class", None) == "slow"
        ]
        new = [
            req
            for req in combined
            if id(req) not in owned_ids
            if getattr(req, "_agentic_kv_queue_class", None) not in {"fast", "slow"}
        ]
        self.waiting_queue = owned + fast + slow + new

    def _merge_disagg_prefill_ready(self, reqs: List[Req]) -> None:
        """Maintain stable fast > slow > new priority without head-of-line blocking."""

        if not reqs:
            return
        self.waiting_queue.extend(reqs)
        self._prioritize_agentic_prefill_ready()

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
                if current is not None and current.state in {
                    SnapshotState.DIRECT_LOADING,
                    SnapshotState.P_RECEIVED,
                }:
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
            self.agentic_p_workset_broker.release_handed(
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
            host_staging.abort_request(req.rid, metadata.parent, req=req)
            if getattr(self, "tp_size", 1) > 1:
                snapshot_id = metadata.parent.snapshot_id
                getattr(self, "agentic_tp_host_active_requests", {}).pop(
                    snapshot_id, None
                )
                getattr(self, "agentic_tp_host_active_since_by_snapshot", {}).pop(
                    snapshot_id, None
                )
                getattr(self, "agentic_tp_host_group_statuses", {}).pop(
                    snapshot_id, None
                )
                getattr(self, "agentic_tp_host_local_admitted", set()).discard(
                    snapshot_id
                )
                mailbox = getattr(self, "agentic_tp_host_mailbox", None)
                if mailbox is not None:
                    mailbox.clear_local(snapshot_id)
                    if self.tp_rank == 0:
                        mailbox.clear_group(snapshot_id)

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

    def _add_request_to_queue(self, req: Req, is_retracted: bool = False):
        if (
            self.disaggregation_mode == DisaggregationMode.PREFILL
            and getattr(self, "tp_size", 1) > 1
            and not is_retracted
            and not hasattr(req, "_agentic_tp_prefill_sequence")
        ):
            metadata = AgenticRequestMetadata.from_req(req)
            req._agentic_tp_prefill_priority = int(
                metadata is None or metadata.parent is None
            )
            req._agentic_tp_prefill_sequence = self.agentic_tp_prefill_sequence
            self.agentic_tp_prefill_sequence += 1
        if self.disaggregation_mode == DisaggregationMode.PREFILL and not is_retracted:
            self._agentic_publish_p_accepted(req)
        if (
            self.disaggregation_mode == DisaggregationMode.PREFILL
            and not is_retracted
            and not hasattr(req, "_agentic_kv_wait_started_at")
        ):
            # Timestamp every P request, including initial requests that have
            # no parent snapshot.  The timestamp survives bootstrap so the
            # ready queue can promote a starved new request deterministically.
            req._agentic_kv_wait_started_at = time.monotonic()
        if (
            self.disaggregation_mode == DisaggregationMode.PREFILL
            and not is_retracted
            and not getattr(req, "_agentic_kv_wait_enqueued", False)
            and not getattr(req, "_agentic_kv_gate_complete", False)
        ):
            metadata = AgenticRequestMetadata.from_req(req)
            if metadata is not None:
                if metadata.parent is not None:
                    receives = getattr(self, "agentic_early_direct_receives", None)
                    entry = (
                        receives.get(metadata.parent.snapshot_id) if receives else None
                    )
                    if entry is not None and entry.completed_at is not None:
                        self._agentic_bind_early_direct_receive(
                            req, metadata.parent, allow_tp_commit=False
                        )
                if not getattr(req, "_agentic_kv_gate_complete", False):
                    # Every P request first enters one scheduler-owned,
                    # metadata-only queue. Parent turns start as Direct and
                    # may be reclassified as Slow; initial requests stay last.
                    req._agentic_kv_queue_class = (
                        "fast" if metadata.parent is not None else "new"
                    )
                    req._agentic_kv_wait_enqueued = True
                    enqueued_at = time.monotonic()
                    req._agentic_kv_wait_started_at = enqueued_at
                    self.agentic_kv_waiting_queue.append((req, enqueued_at))
                    if getattr(self, "tp_size", 1) == 1:
                        self._agentic_track_waiter(req, enqueued_at)
                    return
                # Direct is already resident; enter native Prefill admission.
                self._agentic_publish_p_scheduled(req)
        if self.disaggregation_mode == DisaggregationMode.NULL:
            if not self._set_or_validate_priority(req):
                return
            if self._abort_on_queued_limit(req):
                return
            self._prefetch_kvcache(req)
            self.waiting_queue.append(req)
            req.time_stats.set_wait_queue_entry_time()
        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            host_staging = getattr(self, "agentic_host_staging_manager", None)
            try:
                # This is the common boundary where a request leaves any
                # metadata-only lifecycle gate and enters the native Prefill
                # bootstrap pipeline.  Publish the Router queue ACK for every
                # request here, including initial generations whose wire
                # metadata was stripped or could not be parsed.  The helper is
                # idempotent for Direct/Slow requests that already published
                # the same boundary above.
                self._agentic_publish_p_scheduled(req)
                self._prefetch_kvcache(req)
                self.disagg_prefill_bootstrap_queue.add(
                    req, self.model_config.num_key_value_heads
                )
            except Exception:
                # Normal ownership crosses the bootstrap queue and ends only
                # after PrefillAdder has acquired req.last_node.  Release the
                # temporary Host pin here solely when that handoff cannot be
                # established at all.
                if host_staging is not None:
                    host_staging.release_request_pin(req)
                raise
            req.time_stats.set_prefill_bootstrap_queue_entry_time()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.disagg_decode_prealloc_queue.add(req, is_retracted=is_retracted)
            if not is_retracted:
                req.time_stats.set_decode_prealloc_queue_entry_time()
            else:
                req.time_stats.set_retract_time()
        else:
            raise ValueError(f"Invalid {self.disaggregation_mode=}")

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

    def _set_or_validate_priority(self, req: Req) -> bool:
        """Set the default priority value, or abort the request based on the priority scheduling mode."""
        if self.enable_priority_scheduling and req.priority is None:
            if self.schedule_low_priority_values_first:
                req.priority = sys.maxsize
            else:
                req.priority = -sys.maxsize - 1
        elif (
            not self.enable_priority_scheduling
            and req.priority is not None
            and self.abort_on_priority_when_disabled
        ):
            abort_req = AbortReq(
                finished_reason={
                    "type": "abort",
                    "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                    "message": "Using priority is disabled for this server. Please send a new request without a priority.",
                },
                rid=req.rid,
            )
            req.time_stats.trace_ctx.abort(abort_info=abort_req.finished_reason)
            self.ipc_channels.send_to_tokenizer.send_output(abort_req, req)
            return False
        return True

    def _abort_on_queued_limit(self, recv_req: Req) -> bool:
        """Abort an incoming or existing request if the waiting queue is full. Returns True if the incoming request is aborted."""
        if (
            self.max_queued_requests is None
            or len(self.waiting_queue) + 1 <= self.max_queued_requests
        ):
            return False

        # Reject the incoming request by default.
        req_to_abort = recv_req
        message = "The request queue is full."
        if self.enable_priority_scheduling:
            # With priority scheduling, consider aboritng an existing request based on the priority.
            # direction = 1  => smaller number = higher priority; -1 => larger number = higher priority.
            # max(...) + (direction * priority, queue_time_start) picks the least-preferred request.
            # Tie: later queue_time_start (newer) is evicted first. Preempt only if strictly better.
            direction = 1 if self.schedule_low_priority_values_first else -1
            key_fn = lambda item: (
                direction * item[1].priority,
                item[1].time_stats.wait_queue_entry_time,
            )
            idx, candidate_req = max(enumerate(self.waiting_queue), key=key_fn)
            abort_existing_req = (
                direction * recv_req.priority < direction * candidate_req.priority
            )
            if abort_existing_req:
                if self.enable_hicache_storage:
                    # Release prefetch events associated with the request
                    self.tree_cache.release_aborted_request(candidate_req.rid)
                elif self.enable_hierarchical_cache:
                    self.tree_cache.terminate_prefetch(candidate_req.rid)
                self.waiting_queue.pop(idx)
                req_to_abort = candidate_req
                message = "The request is aborted by a higher priority request."

        self.ipc_channels.send_to_tokenizer.send_output(
            AbortReq(
                finished_reason={
                    "type": "abort",
                    "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                    "message": message,
                },
                rid=req_to_abort.rid,
            ),
            req_to_abort,
        )
        req_to_abort.time_stats.trace_ctx.abort(abort_info={"reason": message})
        return req_to_abort.rid == recv_req.rid

    def _abort_on_waiting_timeout(self):
        if (timeout_s := envs.SGLANG_REQ_WAITING_TIMEOUT.get()) <= 0:
            return

        deleted_reqs = set()
        deadline = time.perf_counter() - timeout_s
        for req in self.waiting_queue:
            entry_time = req.time_stats.wait_queue_entry_time
            if 0 < entry_time < deadline:
                if self.enable_hicache_storage:
                    # Release prefetch events associated with the request
                    self.tree_cache.release_aborted_request(req.rid)
                self.ipc_channels.send_to_tokenizer.send_output(
                    AbortReq(
                        finished_reason={
                            "type": "abort",
                            "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                            "message": "Request waiting timeout reached.",
                        },
                        rid=req.rid,
                    ),
                    req,
                )
                deleted_reqs.add(req)

        if deleted_reqs:
            self.waiting_queue = [
                req for req in self.waiting_queue if req not in deleted_reqs
            ]

    def handle_embedding_request(
        self,
        recv_req: TokenizedEmbeddingReqInput,
    ):
        req = Req(
            recv_req.rid,
            recv_req.input_text,
            recv_req.input_ids,
            recv_req.sampling_params,
            positional_embed_overrides=recv_req.positional_embed_overrides,
            token_type_ids=recv_req.token_type_ids,
            routed_dp_rank=recv_req.routed_dp_rank,
            priority=recv_req.priority,
            dimensions=recv_req.dimensions,
            lora_id=recv_req.lora_id,
            http_worker_ipc=recv_req.http_worker_ipc,
            time_stats=recv_req.time_stats,
            return_pooled_hidden_states=recv_req.return_pooled_hidden_states,
            multi_item_delimiter_indices=recv_req.multi_item_delimiter_indices,
        )
        req.tokenizer = self.tokenizer

        # Handle multimodal inputs
        if recv_req.image_inputs is not None:
            image_inputs = self._get_multimodal_inputs(recv_req.image_inputs)
            # Expand a single image token into multiple dummy tokens for receiving image embeddings
            # The `pad_input_ids_func` is model-specific and may be None for
            # embedding models or models not requiring special padding.
            # If None, `req.origin_input_ids` is expected to be correctly populated already.
            if (
                not self._try_apply_padded_mm_input_ids(recv_req, req, image_inputs)
                and self.pad_input_ids_func
            ):
                # See companion call site above for the array.array wrap rationale.
                req.origin_input_ids = array(
                    "q", self.pad_input_ids_func(req.origin_input_ids, image_inputs)
                )

            req.extend_image_inputs(image_inputs)
            self._maybe_compute_mrope_positions(req)

            if len(req.origin_input_ids) >= self.max_req_input_len:
                req.set_finish_with_abort(
                    error_msg=(
                        "Multimodal prompt is too long after expanding multimodal tokens. "
                        f"After expanding {len(req.origin_input_ids_unpadded)=} => {len(req.origin_input_ids)} >= {self.max_req_input_len}."
                    )
                )
                self._add_request_to_queue(req)
                return

        # Validate prompts length
        error_msg = validate_input_length(
            req,
            self.max_req_input_len,
            self.server_args.allow_auto_truncate,
        )
        if error_msg:
            self._add_request_to_queue(req)
            return

        # Copy more attributes
        req.logprob_start_len = -1
        self._add_request_to_queue(req)

    def handle_batch_embedding_request(
        self,
        recv_req: BatchTokenizedEmbeddingReqInput,
    ):
        """Handle optimized batch embedding request."""
        logger.debug(
            f"Processing batch embedding request with {len(recv_req)} requests"
        )

        # Process each request in the batch
        for tokenized_req in recv_req:
            self.handle_embedding_request(tokenized_req)

    def stash_chunked_request(self, req: Req):
        maybe_cache_unfinished_req(req, self.tree_cache, chunked=True)

    def process_pending_chunked_abort(self) -> None:
        """Abort an in-flight chunked-prefill request once it is safe to do so.

        ``abort_request`` only records the target in ``_pending_chunked_abort_req``
        (tearing it down mid-iteration is unsafe). Clearing ``chunked_req`` here at
        the top of the scheduling step stops the next chunk from launching; the
        chunk already launched is drained when its result is resolved. Under overlap
        the result lands a step later, so the batch-result processors keep
        ``inflight_middle_chunks`` accounting intact and skip the aborted chunk:
        ``process_batch_result_disagg_prefill`` via its ``is_aborted`` drop, and
        ``process_batch_result_prefill`` via its chunked branch (the finished req
        is excluded from streaming and its logprob offset is still accounted).
        Mirrors ``handle_bootstrap_failure``.
        """
        req = self._pending_chunked_abort_req
        if req is None:
            return
        if self.chunked_req is not req:
            # Already past chunked prefill; the running-batch abort path handles
            # it. Drop the marker once the request is actually gone.
            if req.finished() or req.req_pool_idx is None:
                self._pending_chunked_abort_req = None
            return

        prepare_abort(req, "Aborted")
        req.time_stats.trace_ctx.abort(abort_info={"reason": "Aborted"})
        req.to_finish = None
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            req.disagg_kv_sender.abort()
            maybe_release_metadata_buffer(
                req, self.req_to_metadata_buffer_idx_allocator
            )
            req.pending_bootstrap = False
        if self.enable_hicache_storage:
            self.tree_cache.release_aborted_request(req.rid)
        if (
            req.req_pool_idx is not None or self.tree_cache.supports_mamba()
        ) and not req.kv_committed_freed:
            release_kv_cache(req, self.tree_cache, is_insert=False)

        self.chunked_req = None
        self._chunked_req_scheduled_last_iter = False
        self._pending_chunked_abort_req = None
        self.ipc_channels.send_to_tokenizer.send_output(AbortReq(rid=req.rid), req)
        logger.debug(f"Abort chunked prefill request. {req.rid=}")

    def _build_hisparse_decode_batch(self, reqs):
        """Build a ScheduleBatch for hisparse requests transitioning from staging to decode."""
        device = self.device

        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=self.enable_overlap,
            spec_algorithm=self.spec_algorithm,
        )

        req_pool_indices = [r.req_pool_idx for r in reqs]
        batch.req_pool_indices = torch.tensor(
            req_pool_indices, dtype=torch.int64, device=device
        )
        batch.req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
        seq_lens = [len(r.origin_input_ids) + len(r.output_ids) - 1 for r in reqs]
        batch.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=device)
        batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        batch.seq_lens_sum = sum(seq_lens)
        # Stash last token into relay; resolve_forward_inputs will gather.
        last_tokens = torch.tensor(
            [r.output_ids[-1] for r in reqs], dtype=torch.int64, device=device
        )
        self.future_map.stash(batch.req_pool_indices, last_tokens)
        batch.input_ids = None

        if batch.return_logprob:
            batch.top_logprobs_nums = [r.logprob.top_logprobs_num for r in reqs]
            batch.token_ids_logprobs = [list(r.origin_input_ids) for r in reqs]

        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch, self.model_config.vocab_size
        )
        # todo hisparse, maybe other info to contain for the new batch
        return batch

    @scheduler_nvtx_method("scheduler.get_next_batch_to_run")
    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
        self.process_pending_chunked_abort()

        if self.enable_fpm:
            self._fpm_batch_t0 = time.monotonic()
        self._abort_on_waiting_timeout()
        self._abort_on_running_timeout()
        if self.dllm_config is not None:
            self.dllm_manager.filter_finished_reqs()

        # Merge the prefill batch into the running batch
        chunked_req_to_exclude = set()

        if self.dllm_config is not None and self.dllm_manager.any_staging_reqs():
            chunked_req_to_exclude.update(self.dllm_manager.staging_queue)
            for req in self.dllm_manager.staging_queue:
                self.stash_chunked_request(req)

        if self.chunked_req is not None:
            # Move the chunked request out of the batch so that we can merge
            # only finished requests to running_batch.
            chunked_req_to_exclude.add(self.chunked_req)

            # Stash (cache) the previous chunk only when it produced new KV
            # beyond what is already cached. A parked chunk (add_chunked_req
            # hybrid-SWA early-return) leaves fill_len == len(prefix_indices),
            # so there is nothing new to cache and stashing would be a no-op.
            if self.chunked_req.fill_len > len(self.chunked_req.prefix_indices):
                self.stash_chunked_request(self.chunked_req)

        # HiSparse has its own prefill-to-decode transition; skip last_batch merge.
        if self.enable_hisparse:
            ready_reqs = self.hisparse_coordinator.collect_ready_reqs()
            if len(ready_reqs) > 0:
                new_batch = self._build_hisparse_decode_batch(ready_reqs)
                if self.running_batch.is_empty():
                    self.running_batch = new_batch
                else:
                    self.running_batch.merge_batch(new_batch)
                self.running_batch.hisparse_coordinator = self.hisparse_coordinator
            # Reset batch_is_full so the scheduler can schedule more prefills.
            self.running_batch.batch_is_full = False

        if (
            not self.enable_hisparse
            and self.last_batch
            and self.last_batch.forward_mode.is_extend()
        ):
            if self.last_batch.chunked_req is not None:
                # In the context pipeline parallelism, after the last chunk, the current microbatch still track outdated chunked_req.
                # We need to discard it.
                chunked_req_to_exclude.add(self.last_batch.chunked_req)

            if self.dllm_config is not None and self.last_batch.reqs:
                chunked_req_to_exclude.update(self.last_batch.reqs)

            # Filter batch
            last_bs = self.last_batch.batch_size()
            self.last_batch.filter_batch(
                chunked_req_to_exclude=list(chunked_req_to_exclude)
            )
            if self.last_batch.batch_size() < last_bs:
                self.running_batch.batch_is_full = False

            # Merge the new batch into the running batch.
            if not self.last_batch.is_empty():
                if self.running_batch.is_empty():
                    self.running_batch = self.last_batch
                else:
                    # Merge running_batch with prefill batch
                    self.running_batch.merge_batch(self.last_batch)

        # For prefill-only batch, filter out finished requests since they
        # won't go through the decode step. This keeps running_batch accurate
        # for load reporting (num_running_reqs via /v1/loads).
        # Runs outside the last_batch block so stale requests are cleaned
        # even when no new batches arrive (e.g. traffic stops).
        if self.running_batch.is_prefill_only:
            self.running_batch.filter_batch()
            if self.running_batch.is_empty():
                self.running_batch.batch_is_full = False

        if self.dllm_config is not None:
            new_batch = self.get_new_batch_dllm()
        else:
            new_batch = self.get_new_batch_prefill()

        need_mlp_sync = self.require_mlp_sync
        if (
            need_mlp_sync
            and not self.spec_algorithm.is_none()
            and not self.server_args.speculative_skip_dp_mlp_sync
        ):
            # NOTE: This branch makes sure prefill and decode batches will not be mixed when spec and dp-attn is enabled.
            # Before merging the new batch into running batch:
            # 1. All new batches are none -> need_mlp_sync remains true (sync is needed for decode batch).
            # 2. All new batches are some (prefill / idle) -> we do not need prepare mlp sync one more time.
            new_batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(new_batch)
            need_mlp_sync = new_batch is None

        if new_batch is not None:
            # Run prefill first if possible
            ret = new_batch
        else:
            # Run decode (skip for prefill-only batches)
            if (
                not self.running_batch.is_empty()
                and not self.running_batch.is_prefill_only
            ):
                self.running_batch = self.update_running_batch(self.running_batch)
                ret = self.running_batch if not self.running_batch.is_empty() else None
            else:
                ret = None

        # Handle DP attention and log stats
        ret = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(
            ret, need_sync=need_mlp_sync
        )

        # Handle ngram embedding
        ret = self._maybe_prepare_ngram_embedding(ret)

        if ret:
            set_schedule_time_batch(ret)
            if self.enable_fpm:
                ret.fpm_start_time = self._fpm_batch_t0

        return ret

    def get_num_allocatable_reqs(self, running_bs):
        res = get_global_server_args().pp_max_micro_batch_size - running_bs
        res = min(res, self.req_to_token_pool.available_size())
        return res

    def _should_delay_dflash_prefill_for_batching(self, running_bs: int) -> bool:
        if not self.spec_algorithm.is_dflash():
            return False
        if running_bs <= 0 or self.chunked_req is not None:
            return False

        return should_delay_dflash_prefill_for_batching(
            running_bs=running_bs,
            num_allocatable_reqs=self.get_num_allocatable_reqs(running_bs),
            max_running_requests=self.max_running_requests,
            prefill_refill_target=self.dflash_prefill_refill_target,
        )

    def get_new_batch_prefill(self) -> Optional[ScheduleBatch]:
        prefill_delayer_single_pass = None
        if self.prefill_delayer:
            # Get max usage across all pools for prefill delay decision
            max_pool_usage = (
                self.pool_stats_observer.get_pool_stats().get_max_pool_usage()
            )
            prefill_delayer_single_pass = PrefillDelayerSinglePassExecutor(
                self.prefill_delayer, token_usage=max_pool_usage
            )

        ret = self._get_new_batch_prefill_raw(
            prefill_delayer_single_pass=prefill_delayer_single_pass
        )

        if self.prefill_delayer:
            prefill_delayer_single_pass.finalize(actual_prefill=ret is not None)

        return ret

    def _get_new_batch_prefill_raw(
        self, prefill_delayer_single_pass: Optional[PrefillDelayerSinglePassExecutor]
    ) -> Optional[ScheduleBatch]:
        host_staging = getattr(self, "agentic_host_staging_manager", None)
        if host_staging is not None:
            host_staging.poll()

        # Check if the grammar is ready in the grammar queue
        if self.grammar_manager.has_waiting_grammars():
            ready_grammar_requests = self.grammar_manager.get_ready_grammar_requests()
            for req in ready_grammar_requests:
                self._add_request_to_queue(req)

        if self.enable_hierarchical_cache:
            self.tree_cache.check_hicache_events()

        if self.enable_priority_preemption or self.is_hybrid_swa:
            # Reset batch_is_full to try preemption with a prefill adder.
            self.running_batch.batch_is_full = False

        if (
            self.running_batch.batch_is_full or len(self.waiting_queue) == 0
        ) and self.chunked_req is None:
            return None

        running_bs = len(self.running_batch.reqs)
        if self._should_delay_dflash_prefill_for_batching(running_bs):
            return None

        # Ignore the check if self.chunked_req is not None.
        # In the non-PP case, when self.chunked_req is not None, num_allocatable_reqs should always be greater than 0,
        # as the space for the chunked requests has just been released.
        # In PP case, chunked requests (or dllm requests) can start in one microbatch and end in another microbatch, so the max_running_requests per microbatch should not be strict.
        # Instead, we should always allow chunked requests to be added, otherwise, there will be a memory leak.
        if (
            self.get_num_allocatable_reqs(running_bs) <= 0
            and self.chunked_req is None
            and not self.enable_priority_preemption
        ):
            self.running_batch.batch_is_full = True
            return None

        # Get priority queue
        self.policy.calc_priority(self.waiting_queue, self.running_batch)
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # Cache-aware/FCFS policy sorting above must not erase the
            # agentic pipeline order.  Requests whose KV is still loading are
            # skipped below, so ordinary ready work remains work-conserving.
            self._prioritize_agentic_prefill_ready()

        if TEST_RETRACT and running_bs > TEST_RETRACT_NO_PREFILL_BS:
            # If we are testing retraction and the running batch size exceeds
            # TEST_RETRACT_NO_PREFILL_BS, we skip the prefill to keep the requests
            # in the waiting queue.
            return None

        # Determine chunked_prefill_size for this batch
        chunked_prefill_size = self.chunked_prefill_size
        if self.chunked_req is not None and self.enable_dynamic_chunking:
            history_len = len(self.chunked_req.prefix_indices)
            dynamic_size = self.predict_next_chunk_size(history_len)
            if dynamic_size is not None:
                chunked_prefill_size = dynamic_size

        # Prefill policy
        adder = PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            self.running_batch,
            self.new_token_ratio_tracker.current,
            self.max_prefill_tokens,
            chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
            self.priority_scheduling_preemption_threshold,
            max_prefill_bs=self.max_prefill_bs,
            max_running_requests=self.max_running_requests,
            prefill_max_requests=self.server_args.prefill_max_requests,
            prefill_delayer_single_pass=prefill_delayer_single_pass,
            dllm_config=self.dllm_config,
            waiting_queue_len=len(self.waiting_queue),
        )

        if self.chunked_req is not None:
            # The native chunk continuation path assumes that finishing the
            # previous chunk released enough KV for the next one.  A
            # disaggregated Prefill worker keeps completed/P-ready prompts
            # resident until P->D finishes, so that assumption is false under
            # downstream backpressure.  When no page is currently allocatable,
            # defer the continuation and let the transfer consumer release
            # some P KV instead of forcing a chunk into an empty allocator.
            # There is deliberately no percentage watermark here: any real
            # allocatable capacity remains usable.
            workset_suffix_indices = getattr(
                self.chunked_req,
                "_agentic_workset_suffix_indices",
                None,
            )
            has_workset_suffix = (
                getattr(self.chunked_req, "_agentic_workset_backed", False)
                and workset_suffix_indices is not None
                and workset_suffix_indices.numel() > 0
            )
            if (
                self.disaggregation_mode == DisaggregationMode.PREFILL
                and adder.rem_total_tokens <= 0
                and not has_workset_suffix
            ):
                logger.info(
                    "Deferring disaggregated Prefill chunk: no allocatable "
                    "KV tokens (inflight=%d)",
                    len(self.disagg_prefill_inflight_queue),
                )
                return None
            self.chunked_req.init_next_round_input()
            self.chunked_req = adder.add_chunked_req(self.chunked_req)

        if self.enable_lora:
            running_loras = {
                req.lora_id for req in self.running_batch.reqs if not req.finished()
            }
            # Account for LoRAs that are already loaded in the adder, such as chunked requests
            running_loras.update(req.lora_id for req in adder.can_run_list)

            if self.lora_drainer:
                self.lora_drainer.update_draining_state(
                    self.waiting_queue,
                    self.running_batch.reqs,
                )

        mamba_allocator = getattr(self.req_to_token_pool, "mamba_allocator", None)
        if mamba_allocator is not None:
            mamba_allocator.alloc_group_begin(len(self.waiting_queue))
        # Get requests from the waiting queue to a new prefill batch
        for req in self.waiting_queue:
            if self.enable_lora and not self._can_schedule_lora_req(req, running_loras):
                continue

            running_bs = len(self.running_batch.reqs)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                self.running_batch.batch_is_full = True
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                # In prefill mode, prealloc queue and transfer queue can also take memory,
                # so we need to check if the available size for the actual available size.
                if len(adder.can_run_list) >= self.req_to_token_pool.available_size():
                    self.running_batch.batch_is_full = True

            if self.running_batch.batch_is_full:
                if (
                    not self.enable_priority_preemption
                    or not adder.preempt_to_schedule(req, self.server_args)
                ):
                    break

            if self.enable_hicache_storage:
                prefetch_done = self.tree_cache.check_prefetch_progress(req.rid)
                if not prefetch_done:
                    # skip staging requests that are ongoing prefetch
                    continue
                # Pop the number of tokens loaded from storage (L3 hits)
                req.storage_hit_length = self.tree_cache.pop_prefetch_loaded_tokens(
                    req.rid
                )
                pop_failed = getattr(self.tree_cache, "pop_prefetch_failed", None)
                if pop_failed is not None and pop_failed(req.rid):
                    self._agentic_abandon_load(req)

            workset_suffix_indices = getattr(
                req, "_agentic_workset_suffix_indices", None
            )
            has_private_workset = bool(
                getattr(req, "_agentic_workset_backed", False)
                and workset_suffix_indices is not None
                and workset_suffix_indices.numel() > 0
            )
            if adder.can_run_list:
                first_workset_suffix = getattr(
                    adder.can_run_list[0],
                    "_agentic_workset_suffix_indices",
                    None,
                )
                batch_has_private_workset = bool(
                    getattr(adder.can_run_list[0], "_agentic_workset_backed", False)
                    and first_workset_suffix is not None
                    and first_workset_suffix.numel() > 0
                )
                if batch_has_private_workset != has_private_workset:
                    # Private workset pages and allocator-owned pages have
                    # different allocation/rollback contracts.  Keeping each
                    # Prefill batch homogeneous makes ownership explicit and
                    # still remains work-conserving because workset requests
                    # are ordered first.
                    break

            req.init_next_round_input(self.tree_cache)
            p_ready_credit = getattr(self, "_p_ready_compute_credit_tokens", None)
            # Once scheduled, the request's complete matched + new Prompt KV
            # becomes protected until P->D transfer finishes.  Account for the
            # full prompt, including device/Host cache hits, not only the new
            # tokens sent through the model.  This is deliberately conservative
            # when two requests share a prefix; V1 values smooth bounded
            # admission over squeezing the final few cache pages from a batch.
            p_ready_is_new = getattr(req, "_agentic_kv_queue_class", "new") == "new"
            p_ready_protected_tokens = (
                -(-int(req.fill_len) // self.page_size) * self.page_size
                if p_ready_credit is not None and p_ready_is_new
                else 0
            )
            if p_ready_credit is not None and p_ready_protected_tokens > p_ready_credit:
                # Do not let a large head-of-line prompt prevent smaller work
                # later in the queue from using the available credit.
                continue
            if (
                getattr(req, "_agentic_kv_direct_hit_tokens", 0)
                or getattr(req, "_agentic_kv_manifest", None) is not None
            ):
                logger.info(
                    "AgenticKV p_rematch req=%s device_tokens=%d host_tokens=%d "
                    "storage_tokens=%d expected_tokens=%d extra_key=%s",
                    req.rid,
                    len(req.prefix_indices),
                    req.host_hit_length,
                    getattr(req, "storage_hit_length", 0),
                    getattr(
                        getattr(req, "_agentic_kv_manifest", None),
                        "token_count",
                        getattr(req, "_agentic_kv_direct_hit_tokens", 0),
                    ),
                    req.extra_key,
                )
            self._agentic_mark_p_host(req)
            if (
                getattr(getattr(req, "_agentic_kv_manifest", None), "state", None)
                is SnapshotState.P_LOADING
            ):
                # No operation, a revoked operation, or an incomplete GET.
                # The request continues by recomputing, but the claimed
                # snapshot must not remain pinned indefinitely.
                self._agentic_abandon_load(req)
            self._agentic_consume_if_already_on_gpu(req)
            can_run_before = len(adder.can_run_list)
            if has_private_workset:
                res = adder.add_one_req_with_private_token_credit(
                    req,
                    has_chunked_req=(self.chunked_req is not None),
                    truncation_align_size=self.truncation_align_size,
                    private_token_credit=workset_suffix_indices.numel(),
                )
            else:
                res = adder.add_one_req(
                    req,
                    has_chunked_req=(self.chunked_req is not None),
                    truncation_align_size=self.truncation_align_size,
                )
            admitted = len(adder.can_run_list) > can_run_before
            if admitted:
                # add_one_req() has acquired the native request lock.  It is
                # now safe to remove Direct/Slow's transport pin without an
                # evictable gap, including the paired Mamba checkpoint lock.
                self._agentic_release_restore_pin_after_admission(req)
            if p_ready_credit is not None and admitted:
                p_ready_credit -= p_ready_protected_tokens
                self._p_ready_compute_credit_tokens = p_ready_credit

            if self.enable_lora:
                running_loras.add(req.lora_id)

            if res != AddReqResult.CONTINUE:
                added = len(adder.can_run_list) > 0 and req is adder.can_run_list[-1]
                if res == AddReqResult.NO_TOKEN:
                    if self.enable_hierarchical_cache:
                        # Set batch_is_full after making sure there are requests that can be served
                        self.running_batch.batch_is_full = len(
                            adder.can_run_list
                        ) > 0 or (not self.running_batch.is_empty())
                    else:
                        self.running_batch.batch_is_full = True
                # revert matched mamba idx to avoid memory leak, if req is not added.
                # Only free if the slot was freshly allocated in this batch (not
                # pre-existing from a session). Session-held slots have their own
                # lifecycle and freeing them here causes double-free.
                added = len(adder.can_run_list) > 0 and req is adder.can_run_list[-1]
                if (
                    not added
                    and req.mamba_pool_idx is not None
                    and not getattr(req, "session", None)
                ):
                    self.tree_cache.req_to_token_pool.mamba_allocator.free(
                        req.mamba_pool_idx.unsqueeze(-1)
                    )
                    req.mamba_pool_idx = None
                break

        if mamba_allocator is not None:
            mamba_allocator.alloc_group_end()

        can_run_list: List[Req] = adder.can_run_list
        if len(can_run_list) == 0:
            return None

        # A private reverse-KV workset changes allocator ownership.  Every TP
        # rank must validate the same all-private batch before any rank calls
        # alloc_for_extend(), whose successful transaction consumes leases.
        # Running this consensus for ordinary agentic Prefill batches too
        # prevents one corrupt rank from returning early while peers enter a
        # collective for a private batch.
        if getattr(self, "agentic_p_workset_broker", None) is not None:
            self._validate_agentic_workset_batch_tp(can_run_list)

        # Update waiting queue

        can_run_set = set(can_run_list)
        self.waiting_queue = [x for x in self.waiting_queue if x not in can_run_set]
        if adder.preempt_list:
            for req in adder.preempt_list:
                self._add_request_to_queue(req)

        if adder.new_chunked_req is not None:
            # Update chunked prefill
            assert self.chunked_req is None
            self.chunked_req = adder.new_chunked_req

        if self.chunked_req is not None:
            self.chunked_req.inflight_middle_chunks += 1

        set_time_batch(can_run_list, "set_forward_entry_time")

        # Create a new batch
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            chunked_req=self.chunked_req,
        )

        new_batch.contains_last_prefill_chunk = (
            self.chunked_req is None or len(can_run_list) != 1
        )

        self.max_prefill_bs = max(self.max_prefill_bs, len(can_run_list))
        if self.enable_hierarchical_cache:
            # todo (zhiqiang): disable cuda graph execution if hicache loading triggered
            new_batch.hicache_consumer_index = (
                self.tree_cache.ready_to_load_host_cache()
            )

        new_batch.prepare_for_extend()

        # Record prefill stats for logging after forward.
        new_batch.prefill_stats = PrefillStats.from_adder(
            adder,
            self.running_batch.reqs,
            self.enable_priority_scheduling,
            num_pending_tokens=self.load_inquirer._get_num_pending_tokens(
                chunk_deduct=(
                    self.chunked_req.extend_input_len
                    if self.chunked_req is not None
                    else 0
                ),
            ),
        )

        # Mixed-style chunked prefill
        if (
            self.is_mixed_chunk
            and not self.running_batch.is_empty()
            and not (new_batch.return_logprob or self.running_batch.return_logprob)
            # mix_with_running cats input_ids but not input_embeds — shapes would mismatch
            and new_batch.input_embeds is None
        ):
            # TODO (lianmin): support return_logprob + mixed chunked prefill
            self.running_batch.filter_batch()
            if not self.running_batch.is_empty():
                self.running_batch.prepare_for_decode()
                new_batch.mix_with_running(self.running_batch)
                new_batch.decoding_reqs = self.running_batch.reqs
            self.running_batch = ScheduleBatch(
                reqs=[], batch_is_full=self.running_batch.batch_is_full
            )
        else:
            new_batch.decoding_reqs = None

        return new_batch

    def _validate_agentic_workset_batch_tp(self, reqs: List[Req]) -> None:
        flags = [
            bool(
                getattr(req, "_agentic_workset_backed", False)
                and getattr(req, "_agentic_p_workset_lease", None) is not None
                and getattr(req, "_agentic_p_workset_broker", None) is not None
            )
            for req in reqs
        ]
        report = None
        try:
            if any(flags) and not all(flags):
                raise RuntimeError("mixed private/ordinary Prefill batch")
            if flags and all(flags):
                brokers = {
                    id(req._agentic_p_workset_broker): req._agentic_p_workset_broker
                    for req in reqs
                }
                if len(brokers) != 1:
                    raise RuntimeError("private Prefill batch has multiple brokers")
                broker = next(iter(brokers.values()))
                operations = tuple(
                    (
                        req._agentic_p_workset_lease,
                        int(req.extend_input_len),
                        int(req.fill_len) >= len(req.full_untruncated_fill_ids),
                    )
                    for req in reqs
                )
                local_signature = broker.validate_suffix_batch(operations)
                # lease_id is a rank-local allocator identity.  Each broker
                # has already validated its exact physical lease above, but
                # independent TP ranks are neither required nor expected to
                # allocate the same numeric id.  The collective preflight
                # compares only the logical operation and its per-rank shape.
                signature = tuple(
                    (
                        snapshot_id,
                        start,
                        end,
                        final_prompt_chunk,
                        physical_suffix_tokens,
                    )
                    for (
                        snapshot_id,
                        _lease_id,
                        start,
                        end,
                        final_prompt_chunk,
                        physical_suffix_tokens,
                    ) in local_signature
                )
                report = ("private", tuple(req.rid for req in reqs), signature, "")
            else:
                report = ("ordinary", tuple(req.rid for req in reqs), (), "")
        except Exception as exc:
            report = ("invalid", tuple(req.rid for req in reqs), (), repr(exc))

        reports = [report]
        if self.tp_size > 1:
            reports = [None for _ in range(self.tp_size)]
            torch.distributed.all_gather_object(
                reports,
                report,
                group=self.tp_cpu_group,
            )
        if any(item[0] == "invalid" for item in reports) or any(
            item != reports[0] for item in reports[1:]
        ):
            raise RuntimeError(
                "agentic private-workset TP preflight failed before ownership "
                f"commit: reports={reports}"
            )

    def _can_schedule_lora_req(
        self, req: Req, running_loras: set[Optional[str]]
    ) -> bool:
        """
        Check if a LoRA request can be scheduled.

        This method checks two conditions:
        1. The drainer allows scheduling (based on draining state)
        2. The LoRA adapter can be loaded (either already running or can be added)
        """
        if self.lora_drainer and not self.lora_drainer.can_schedule(req):
            return False

        if req.lora_id in running_loras:
            return True

        if self.enable_lora_overlap_loading:
            # For overlapping loading of LoRA weights with computation, we will load each
            # adapter one at a time, as opposed to loading them in one batch
            return self.lora_overlap_loader.try_overlap_load_lora(
                req.lora_id, running_loras
            )
        else:
            new_lora_set = {req.lora_id} | running_loras
            return self.tp_worker.model_runner.lora_manager.validate_lora_batch(
                new_lora_set
            )

    def update_running_batch(self, batch: ScheduleBatch) -> Optional[ScheduleBatch]:
        """Update the current running decoding batch."""
        initial_bs = batch.batch_size()

        batch.filter_batch()
        if batch.is_empty():
            batch.batch_is_full = False
            return batch

        # Eagerly release lock_ref on completed write-through nodes so they
        # become evictable, improving batch scheduling headroom.
        if self.enable_hierarchical_cache:
            self.tree_cache.flush_write_through_acks()

        # Check if decode out of memory
        if (kv_full_retract_flag := not batch.check_decode_mem()) or (
            TEST_RETRACT and self.forward_ct % TEST_RETRACT_INTERVAL == 0
        ):
            old_available_tokens = self.token_to_kv_pool_allocator.available_size()
            old_ratio = self.new_token_ratio_tracker.current
            mamba_allocator = getattr(
                self.tree_cache.req_to_token_pool, "mamba_allocator", None
            )
            old_mamba_available = (
                mamba_allocator.available_size()
                if mamba_allocator is not None
                else None
            )
            retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode(
                self.server_args
            )
            new_available_tokens = self.token_to_kv_pool_allocator.available_size()
            new_token_gained = new_available_tokens - old_available_tokens
            mamba_num_gained = (
                mamba_allocator.available_size() - old_mamba_available
                if mamba_allocator is not None
                else None
            )

            self.metrics_reporter.num_retracted_reqs = len(retracted_reqs)
            if self.metrics_reporter.enable_metrics and len(retracted_reqs) > 0:
                self.metrics_reporter.metrics_collector.increment_retracted_reqs(
                    num_retracted_reqs=len(retracted_reqs),
                    num_retracted_input_tokens=sum(
                        len(r.origin_input_ids) for r in retracted_reqs
                    ),
                    num_retracted_output_tokens=sum(
                        len(r.output_ids) for r in retracted_reqs
                    ),
                )
            self.new_token_ratio_tracker.current = new_token_ratio
            for req in reqs_to_abort:
                abort_reason: FINISH_ABORT = req.to_finish
                self.ipc_channels.send_to_tokenizer.send_output(
                    AbortReq(
                        finished_reason=abort_reason.to_json(),
                        rid=req.rid,
                    ),
                    req,
                )

            msg_prefix = (
                "KV cache pool is full. Retract requests. "
                if kv_full_retract_flag
                else "Testing retraction. "
            )
            msg_details = f"#retracted_reqs: {len(retracted_reqs)}, #new_tokens_gained: {new_token_gained}"
            if mamba_num_gained is not None:
                msg_details += f", #mamba_num_gained: {mamba_num_gained}"
            if kv_full_retract_flag:
                msg_details += (
                    f", #new_token_ratio: {old_ratio:.4f} -> {new_token_ratio:.4f}"
                )
            logger.warning(msg_prefix + msg_details)

            for req in retracted_reqs:
                self._add_request_to_queue(req, is_retracted=True)
        else:
            self.new_token_ratio_tracker.decay_step()

        if batch.batch_size() < initial_bs:
            batch.batch_is_full = False

        if batch.is_empty():
            return batch

        # Update batch tensors
        batch.prepare_for_decode()
        return batch

    def record_batch_in_overlap(self, batch: ScheduleBatch):
        # FIXME(lsyin): hacky way to keep a reference to avoid GPU tensors being freed by torch GC
        # NOTE: More Reliable: record all tensors into the forward stream
        # NOTE: - for all future tensors, we shall always read from future map
        #       - for all non-future tensors (produced only by schedule stream),
        #       we shall keep its reference not being release during all the forwarding pass
        # Snapshot all fields: spec V2 rebinds seq_lens / spec_info mid-forward.
        attr_snapshot = [
            getattr(batch, f.name, None) for f in dataclasses.fields(batch)
        ]
        self.batch_record_ct = (self.batch_record_ct + 1) % 2
        # List (not tuple) so that workers can register additional refs via
        # GenerationBatchResult.extra_keep_alive_refs after forward returns.
        self.batch_record_buf[self.batch_record_ct] = [batch, attr_snapshot]

    @contextmanager
    def _forward_isolation(self, batch: ScheduleBatch, *, overlap: bool):
        """Make SB transactional across one forward (overlap and non-overlap).

        1. Snapshot SB fields so V2's mid-forward mutations (forward_mode /
           input_ids / seq_lens / spec_info / ...) can be undone. V1 / non-spec
           only need sampling_info restored - V1 carries spec_info forward as
           next-iter draft input.
        2. Substitute sampling_info with a forward-only copy (orchestrator=None,
           shares the pre-accumulated penalty buffer) so V2's multiple init_new
           calls don't double-accumulate penalties.
        3. (overlap=True only) Pin (batch, snapshot) into batch_record_buf
           for 2 iters so GPU tensors in the snapshot survive the caching
           allocator past the forward stream. Must run AFTER the sampling_info
           swap so the forward-only copy gets pinned. The non-overlap (sync) path
           runs on a single stream and doesn't allocate batch_record_buf, so it
           passes overlap=False.
        """
        # 1. snapshot
        snapshot_v2_full = not batch.spec_algorithm.is_none()
        sched_snapshot = (
            {f.name: getattr(batch, f.name) for f in dataclasses.fields(batch)}
            if snapshot_v2_full
            else None
        )
        sched_sampling_info = batch.sampling_info

        # 2. sampling_info substitute
        if sched_sampling_info is not None:
            batch.sampling_info = sched_sampling_info.copy_for_forward()

        # 3. pin for 2-iter tensor lifetime (overlap path only)
        if overlap:
            self.record_batch_in_overlap(batch)

        try:
            yield
        finally:
            if snapshot_v2_full:
                for name, value in sched_snapshot.items():
                    setattr(batch, name, value)
            else:
                batch.sampling_info = sched_sampling_info

    @scheduler_nvtx_method("scheduler.run_batch")
    def run_batch(
        self,
        batch: ScheduleBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[GenerationBatchResult, EmbeddingBatchResult]:
        """Run a batch."""
        self.forward_ct += 1
        batch.forward_iter = self.forward_ct

        if self.scripted_scheduler_hook is not None:
            self.scripted_scheduler_hook.on_run_batch(batch)

        agentic_tp_debug = (
            self.tp_size > 1
            and self.server_args.disaggregation_mode == "prefill"
            and os.environ.get("SGLANG_AGENTIC_KV_TP_DEBUG_BATCH", "0") == "1"
        )
        if agentic_tp_debug:
            logger.info(
                "AgenticTP batch_enter ct=%d mode=%s rids=%s seq_lens=%s "
                "extend_lens=%s",
                self.forward_ct,
                batch.forward_mode,
                [req.rid for req in batch.reqs],
                [int(value) for value in batch.seq_lens_cpu],
                [int(req.extend_input_len) for req in batch.reqs],
            )

        # Whether to run the profiler
        self.profiler_manager._profile_batch_predicate(batch)
        if self.forward_sleep_time is not None:
            logger.info(f"Scheduler.run_batch sleep {self.forward_sleep_time}s")
            time.sleep(self.forward_sleep_time)

        # Place holder handling for pd-disagg decode event loop
        if batch.forward_mode.is_prebuilt():
            return self._run_batch_prebuilt(batch)

        # Run forward
        if self.is_generation:
            if self.enable_overlap:
                # Self-gates on batch.spec_info.future_indices; non-spec_v2
                # no-ops (ForwardBatch.init_new lazily computes the sum).
                self.future_map.resolve_seq_lens_cpu(batch)

                with self.forward_stream_ctx:
                    self.forward_stream.wait_stream(self.schedule_stream)
                    # resolve consumes SB staging (prefill_input_ids_cpu /
                    # mix_running_indices). Run OUTSIDE isolation so the
                    # snapshot captures the post-consume state — restoring
                    # post-forward must not un-consume staging.
                    resolve_forward_inputs(batch, self.future_map)

                    with self._forward_isolation(batch, overlap=True):
                        future_indices = batch.req_pool_indices

                        # Spec_v2 fires on_publish mid-worker (between verify and
                        # draft_extend) so schedule prep can overlap with draft_extend.
                        # Non-spec has no later work — scheduler publishes after return.
                        fwd_kwargs = (
                            {
                                "on_publish": partial(
                                    self.future_map.publish, future_indices
                                )
                            }
                            if not batch.spec_algorithm.is_none()
                            else {}
                        )

                        # FIXME: pp is not compatible with overlap
                        batch_result = self.model_worker.forward_batch_generation(
                            batch, **fwd_kwargs
                        )
                        if batch.spec_algorithm.is_none():
                            self.future_map.publish(future_indices, batch.seq_lens + 1)
                        # Park any refs the worker wants kept alive 2 iters
                        # (cross-stream tensor lifetime; pinned in the same
                        # ring slot as the SB attr snapshot).
                        if batch_result.extra_keep_alive_refs:
                            self.batch_record_buf[self.batch_record_ct].extend(
                                batch_result.extra_keep_alive_refs
                            )
                        # FIXME(lsyin): maybe move this to forward_batch_generation
                        batch_result.copy_done = self.device_module.Event()
                        if batch_result.delay_sample_func is None:
                            stash_payload = (
                                batch_result.next_draft_input
                                if not batch.spec_algorithm.is_none()
                                else batch_result.next_token_ids
                            )
                            self.future_map.stash(future_indices, stash_payload)
                            batch_result.copy_to_cpu(
                                return_logprob=batch.return_logprob,
                                return_hidden_states=batch.return_hidden_states,
                            )
                        else:
                            batch_result.future_indices = future_indices

                # Next-iter input_ids relayed via future_map.
                batch.input_ids = None

                if not batch.spec_algorithm.is_none():
                    batch.spec_info = batch_result.next_draft_input
                    batch.spec_info.future_indices = future_indices
            elif self.enable_pdmux and batch.forward_mode.is_split_prefill():
                resolve_forward_inputs(batch, self.future_map)
                batch_result = self.tp_worker.forward_batch_split_prefill(batch)
                if isinstance(batch_result.next_token_ids, torch.Tensor):
                    self.future_map.stash(
                        batch.req_pool_indices, batch_result.next_token_ids
                    )
                batch.input_ids = None
            elif not batch.spec_algorithm.is_none():
                # Non-overlap: drive the V2 worker synchronously (no
                # future_map relay / on_publish).
                resolve_forward_inputs(batch, self.future_map)
                with self._forward_isolation(batch, overlap=False):
                    batch_result = self.model_worker.forward_batch_generation(batch)
                # The isolation restore reverted the worker's in-forward SB edits;
                # re-apply what must carry to the next iter.
                batch.spec_info = batch_result.next_draft_input
                if batch_result.new_seq_lens is not None:
                    batch.seq_lens = batch_result.new_seq_lens
                    if batch.seq_lens_cpu is not None:
                        batch.seq_lens_cpu = batch_result.new_seq_lens.to("cpu")
                        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
                batch.input_ids = None  # rebuilt next iter from draft_token
                self.update_cache_from_scheduler(batch, batch_result)
                # Sync D2H so the result processor can read CPU tensors.
                batch_result.copy_done = self.device_module.Event()
                batch_result.copy_to_cpu(
                    return_logprob=batch.return_logprob,
                    return_hidden_states=batch.return_hidden_states,
                )
            else:
                kwargs = (
                    {"pp_proxy_tensors": pp_proxy_tensors}
                    if self.spec_algorithm.is_none()
                    else {}
                )
                resolve_forward_inputs(batch, self.future_map)
                batch_result = self.model_worker.forward_batch_generation(
                    batch, **kwargs
                )
                if isinstance(batch_result.next_token_ids, torch.Tensor):
                    # Non-spec: relay via future_map, gathered next iter.
                    self.future_map.stash(
                        batch.req_pool_indices, batch_result.next_token_ids
                    )
                    batch.input_ids = None
                self.update_cache_from_scheduler(batch, batch_result)

            # These 2 values are needed for processing the output, but the values can be
            # modified by overlap schedule. So we have to copy them here so that
            # we can use the correct values in output processing.
            if batch.return_logprob:
                batch_result.extend_input_len_per_req = [
                    req.extend_input_len for req in batch.reqs
                ]
                batch_result.extend_logprob_start_len_per_req = [
                    req.extend_logprob_start_len for req in batch.reqs
                ]
            else:
                batch_result.extend_input_len_per_req = None
                batch_result.extend_logprob_start_len_per_req = None

            ret = batch_result
        else:  # embedding or reward model
            if self.enable_overlap:
                self.record_batch_in_overlap(batch)
                with self.forward_stream_ctx:
                    self.forward_stream.wait_stream(self.schedule_stream)
                    resolve_forward_inputs(batch, self.future_map)
                    pooler_output = self.tp_worker.forward_batch_embedding(batch)
                    ret = EmbeddingBatchResult(
                        embeddings=pooler_output.embeddings,
                        pooled_hidden_states=pooler_output.pooled_hidden_states,
                    )
                    ret.copy_to_cpu()
            else:
                resolve_forward_inputs(batch, self.future_map)
                pooler_output = self.tp_worker.forward_batch_embedding(batch)
                ret = EmbeddingBatchResult(
                    embeddings=pooler_output.embeddings,
                    pooled_hidden_states=pooler_output.pooled_hidden_states,
                )

        self._maybe_report_active_ranks()

        return ret

    def _maybe_report_active_ranks(self) -> None:
        if not (
            self.server_args.enable_dp_attention
            and self.server_args.elastic_ep_backend is not None
        ):
            return
        # Get the tensors indicating rank activeness
        tp_active_ranks = self.tp_group.active_ranks.detach().cpu().numpy()
        tp_active_ranks_cpu = self.tp_group.active_ranks_cpu.detach().numpy()
        tp_active_ranks &= tp_active_ranks_cpu
        dp_active_ranks = tp_active_ranks.reshape(self.ps.dp_size, -1).prod(axis=1)
        self.ipc_channels.send_to_tokenizer.send_output(
            ActiveRanksOutput(status=dp_active_ranks.tolist())
        )

    def launch_batch_sample_if_needed(
        self, batch_result: GenerationBatchResult
    ) -> Union[GenerationBatchResult]:
        # TODO(lsyin): make the delayed sample a default behavior after
        # unifying the forward_batch_generation interface (related to spec V2).
        if batch_result is None or batch_result.delay_sample_func is None:
            return

        with self.forward_stream_ctx:
            self.forward_stream.wait_stream(self.schedule_stream)
            _batch_result = batch_result.delay_sample_func()
            assert _batch_result is batch_result
            # Delay-sample is non-spec only; stash takes next_token_ids tensor.
            self.future_map.stash(
                batch_result.future_indices, batch_result.next_token_ids
            )
            batch_result.copy_to_cpu(
                return_logprob=self.cur_batch.return_logprob,
                return_hidden_states=self.cur_batch.return_hidden_states,
            )

        # Release the closure and large GPU tensors that are no longer needed.
        # The delay_sample_func closure captures forward_batch (which holds
        # sampling_info with vocab_mask) and logits_output (which holds
        # next_token_logits). Without clearing these, they stay alive via
        # batch_result in result_queue and batch_record_buf until the next
        # iteration, causing a steady VRAM leak with structured output.
        batch_result.delay_sample_func = None
        if batch_result.logits_output is not None:
            batch_result.logits_output.next_token_logits = None

    @scheduler_nvtx_method("scheduler.process_batch_result")
    def process_batch_result(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        self.publish_load_snapshot(force=batch.forward_mode.is_extend())

        if batch.forward_mode.is_decode():
            self.batch_result_processor.process_batch_result_decode(batch, result)
        elif batch.forward_mode.is_extend():
            if batch.is_dllm():
                self.process_batch_result_dllm(batch, result)
            elif self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.process_batch_result_disagg_prefill(batch, result)
            else:
                self.batch_result_processor.process_batch_result_prefill(batch, result)
        elif batch.forward_mode.is_prebuilt():
            self.batch_result_processor.process_batch_result_prebuilt(batch)
        elif batch.forward_mode.is_idle():
            self.batch_result_processor.process_batch_result_idle(batch, result)

        self.metrics_reporter.log_batch_result_stats(batch, result)

        # Emit forward pass metrics (every iteration when enabled)
        if self.enable_fpm:
            self.metrics_reporter._emit_forward_pass_metrics(batch, result)

        self._maybe_clear_mm_inputs(batch)
        self.maybe_send_health_check_signal()
        self.metrics_reporter.update_device_timer()

    def maybe_send_health_check_signal(self):
        if self.return_health_check_ipcs:
            # Return some signal for the health check.
            # This is used to prevent the health check signal being blocked by long context prefill.
            # However, one minor issue is that this code path does not check the status of detokenizer manager.
            self.ipc_channels.send_to_tokenizer.send_output(
                HealthCheckOutput(
                    http_worker_ipc=self.return_health_check_ipcs.popleft()
                )
            )

    def add_external_corpus(
        self, recv_req: AddExternalCorpusReqInput
    ) -> Optional[AddExternalCorpusReqOutput]:
        if self.external_corpus_manager is None:
            return AddExternalCorpusReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        return self.external_corpus_manager.add(recv_req)

    def remove_external_corpus(
        self, recv_req: RemoveExternalCorpusReqInput
    ) -> RemoveExternalCorpusReqOutput:
        if self.external_corpus_manager is None:
            return RemoveExternalCorpusReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        return self.external_corpus_manager.remove(recv_req)

    def list_external_corpora(
        self, recv_req: ListExternalCorporaReqInput
    ) -> ListExternalCorporaReqOutput:
        if self.external_corpus_manager is None:
            return ListExternalCorporaReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        return self.external_corpus_manager.list(recv_req)

    def clear_hicache_storage_wrapped(self, recv_req: ClearHiCacheReqInput):
        if self.enable_hierarchical_cache:
            self.tree_cache.clear_storage_backend()
            logger.info("Hierarchical cache cleared successfully!")
            if_success = True
        else:
            logging.warning("Hierarchical cache is not enabled.")
            if_success = False
        return ClearHiCacheReqOutput(success=if_success)

    def on_idle(self):
        """Idle housekeeping: guard, check, metrics, reset, sleep."""
        if not self.is_fully_idle():
            return

        # memory leak check (skipped for hisparse — pool counters intentionally
        # diverge during host-backup, see _get_swa_token_info clamp).
        if not self.enable_hisparse:
            has_leak, messages = self.invariant_checker._check_all_pools(
                self.pool_stats_observer.get_pool_stats(),
            )
            if has_leak:
                self.invariant_checker._report_leak("pool", "\n".join(messages))
            self.invariant_checker._check_req_pool()

        # tree cache sanity check
        self.invariant_checker._check_tree_cache()

        # metrics every 30s
        self.metrics_reporter._maybe_log_idle_metrics()

        # kv event publishing
        self.kv_events_publisher.publish_kv_events()

        # reset token ratio
        self.new_token_ratio_tracker.reset()

        # reset device timer window so idle time isn't counted
        self.metrics_reporter.reset_device_timer_window()

        # Publish the idle state so /get_loads and DP balancing do not see stale load.
        self.publish_load_snapshot(force=True)

        # sleep until next event
        self.maybe_sleep_on_idle()

    def is_fully_idle(self, for_health_check=False) -> bool:
        # Health check piggybacks on running requests in process_output.
        # Only running_batch + waiting_queue guarantee active GPU processing;
        # disagg queues (bootstrap/prealloc/transfer) may have items without
        # any request actually running on GPU — e.g. stuck handshake, full
        # KV cache, or stalled transfer — so they can't carry health info.
        # Batch running status
        idle = (
            self.running_batch.is_empty()
            and self.chunked_req is None
            and not self.dllm_manager.any_staging_reqs()
            and (self.last_batch is None or self.last_batch.is_empty())
            and (self.cur_batch is None or self.cur_batch.is_empty())
            and (not self.enable_overlap or len(self.result_queue) == 0)
            and self._pp_microbatches_drained()
        )

        # Waiting queues: waiting + bootstrapping + preallocation + kv transfer (decode)
        idle &= len(self.waiting_queue) == 0
        indexed_waiters = getattr(self, "agentic_kv_waiting_by_rid", None)
        idle &= (
            len(
                getattr(self, "agentic_kv_waiting_queue", ())
                if indexed_waiters is None or getattr(self, "tp_size", 1) > 1
                else indexed_waiters
            )
            == 0
        )
        idle &= len(getattr(self, "agentic_early_direct_receives", ())) == 0

        if not for_health_check:
            # Grammar queue and prefill inflight queue may not produce batch
            # results instantly, but they still indicate the server is not idle.
            idle &= len(self.grammar_manager.grammar_queue) == 0
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                idle &= len(self.disagg_prefill_inflight_queue) == 0
                idle &= len(self.disagg_prefill_bootstrap_queue.queue) == 0
                # Direct/Slow workset pages remain allocator-owned until the
                # TP retirement command frees every rank at one scheduler
                # boundary.  Do not run strict idle leak checks in that
                # legitimate hand-off window.
                broker = getattr(self, "agentic_p_workset_broker", None)
                if broker is not None:
                    idle &= int(broker.leased_tokens) == 0

            if self.disaggregation_mode == DisaggregationMode.DECODE:
                idle &= len(self.disagg_decode_prealloc_queue.queue) == 0
                idle &= len(self.disagg_decode_prealloc_queue.retracted_queue) == 0
                idle &= len(self.disagg_decode_transfer_queue.queue) == 0
                if self.decode_offload_manager is not None:
                    idle &= len(self.decode_offload_manager.ongoing_offload) == 0
                    # Agentic Direct/Shared-Arena candidates intentionally keep
                    # a finished request's complete KV and Mamba state alive
                    # until P receives it or Host staging commits.  They are
                    # allocator-owned in-flight work, not a pool leak.
                    idle &= (
                        int(
                            getattr(
                                self.decode_offload_manager,
                                "agentic_inflight_snapshot_count",
                                0,
                            )
                        )
                        == 0
                    )

            # HiSparse: staging requests transitioning prefill -> decode
            if self.enable_hisparse:
                idle &= not self.hisparse_coordinator.has_ongoing_staging()

            # HiCache: in-flight async ops (GPU↔Host↔L3) must drain before
            # destructive operations like attach/detach/flush_cache.
            if self.enable_hierarchical_cache:
                tc = self.tree_cache
                idle &= len(tc.ongoing_write_through) == 0
                idle &= len(tc.ongoing_load_back) == 0
                if tc.enable_storage:
                    idle &= len(tc.ongoing_prefetch) == 0
                    idle &= len(tc.ongoing_backup) == 0

        return idle

    def _pp_microbatches_drained(self) -> bool:
        if self.ps.pp_size == 1:
            return True
        return all(x.is_empty() for x in self.running_mbs) and all(
            mb is None or mb.is_empty() for mb in self.mbs
        )

    def attach_hicache_storage_wrapped(
        self, recv_req: AttachHiCacheStorageReqInput
    ) -> AttachHiCacheStorageReqOutput:
        if not self.enable_hierarchical_cache:
            return AttachHiCacheStorageReqOutput(
                success=False, message="Hierarchical cache is not enabled."
            )

        if not self.is_fully_idle():
            return AttachHiCacheStorageReqOutput(
                success=False,
                message=(
                    "Reject attach: scheduler is not idle. "
                    f"#queue-req={len(self.waiting_queue)} "
                    f"#running-req={len(self.running_batch.reqs)}"
                ),
            )

        if not hasattr(self.tree_cache, "attach_storage_backend"):
            return AttachHiCacheStorageReqOutput(
                success=False,
                message="Current tree_cache implementation does not support dynamic attach.",
            )

        try:
            ok, msg = self.tree_cache.attach_storage_backend(
                storage_backend=recv_req.hicache_storage_backend,
                storage_backend_extra_config_json=recv_req.hicache_storage_backend_extra_config_json,
                served_model_name=self.server_args.served_model_name,
                hicache_storage_prefetch_policy=recv_req.hicache_storage_prefetch_policy,
                hicache_write_policy=recv_req.hicache_write_policy,
            )
        except Exception as e:
            logger.exception("Attach HiCache storage backend failed with exception.")
            return AttachHiCacheStorageReqOutput(success=False, message=str(e))
        if ok:
            self.enable_hicache_storage = True
            self.server_args.hicache_storage_backend = recv_req.hicache_storage_backend
            if recv_req.hicache_storage_backend_extra_config_json is not None:
                self.server_args.hicache_storage_backend_extra_config = (
                    recv_req.hicache_storage_backend_extra_config_json
                )
            if recv_req.hicache_storage_prefetch_policy is not None:
                self.server_args.hicache_storage_prefetch_policy = (
                    recv_req.hicache_storage_prefetch_policy
                )
            if recv_req.hicache_write_policy is not None:
                self.server_args.hicache_write_policy = recv_req.hicache_write_policy
            logger.info(
                f"Attached HiCache storage backend: {recv_req.hicache_storage_backend}"
            )
        return AttachHiCacheStorageReqOutput(success=ok, message=msg)

    def detach_hicache_storage_wrapped(
        self, recv_req: DetachHiCacheStorageReqInput
    ) -> DetachHiCacheStorageReqOutput:
        if not self.enable_hierarchical_cache:
            return DetachHiCacheStorageReqOutput(
                success=False, message="Hierarchical cache is not enabled."
            )

        if not self.is_fully_idle():
            return DetachHiCacheStorageReqOutput(
                success=False,
                message=(
                    "Reject detach: scheduler is not idle. "
                    f"#queue-req={len(self.waiting_queue)} "
                    f"#running-req={len(self.running_batch.reqs)}"
                ),
            )

        if not hasattr(self.tree_cache, "detach_storage_backend"):
            return DetachHiCacheStorageReqOutput(
                success=False,
                message="Current tree_cache implementation does not support dynamic detach.",
            )

        # Idempotent detach: even if scheduler thinks storage is disabled, we still
        # attempt best-effort cleanup in tree_cache (it may have leftover state).
        try:
            ok, msg = self.tree_cache.detach_storage_backend()
        except Exception as e:
            logger.exception("Detach HiCache storage backend failed with exception.")
            return DetachHiCacheStorageReqOutput(success=False, message=str(e))

        if ok or (not self.enable_hicache_storage):
            # Treat "already disabled / nothing to do" as success for idempotence.
            self.enable_hicache_storage = False
            self.server_args.hicache_storage_backend = None
            self.server_args.hicache_storage_backend_extra_config = None
            logger.info("Detached HiCache storage backend.")
            return DetachHiCacheStorageReqOutput(
                success=True, message=msg or "HiCache storage backend is detached."
            )

        return DetachHiCacheStorageReqOutput(success=False, message=msg)

    def flush_cache(self, empty_cache: bool = True):
        """Flush memory pools (e.g., KV cache, Mamba cache) and optionally empty device allocator cache."""
        if self.is_fully_idle():
            self.cur_batch = None
            self.last_batch = None
            self.tree_cache.reset()
            self.req_to_token_pool.clear()
            self.token_to_kv_pool_allocator.clear()
            self.grammar_manager.clear()
            self.metrics_reporter.reset_metrics()

            if self.draft_worker:
                self.draft_worker.clear_cache_pool()

            if empty_cache:
                current_platform.empty_cache()
            # Per-DP-group leader logs once: ranks within a DP group are
            # state-synchronous, but DP groups may diverge.
            if self.metrics_reporter.is_stats_logging_rank:
                logger.info("Cache flushed successfully!")
            success = True
        else:
            logging.warning(
                f"Cache not flushed because there are pending requests. "
                f"#queue-req: {len(self.waiting_queue)}, "
                f"#running-req: {len(self.running_batch.reqs)}"
            )
            success = False
        return success

    def get_internal_state(self, recv_req: GetInternalStateReq):
        ret = dict(vars(get_global_server_args()))  # vars returns a ref to obj.__dict__
        ret["last_gen_throughput"] = self.metrics_reporter.last_gen_throughput
        ret["memory_usage"] = {
            "weight": round(self.tp_worker.model_runner.weight_load_mem_usage, 2),
            "kvcache": round(
                self.token_to_kv_pool_allocator.get_kvcache().mem_usage, 2
            ),
            "token_capacity": int(self.max_total_num_tokens),
            "graph": round(self.tp_worker.model_runner.graph_mem_usage, 2),
        }
        ret["effective_max_running_requests_per_dp"] = self.max_running_requests

        if (
            not self.spec_algorithm.is_none()
            and self.metrics_reporter.spec_total_num_forward_ct > 0
        ):
            ret["avg_spec_accept_length"] = (
                self.metrics_reporter.spec_total_num_accept_tokens
                / self.metrics_reporter.spec_total_num_forward_ct
            )

        if RECORD_STEP_TIME:
            ret["step_time_dict"] = self.metrics_reporter.step_time_dict

        # This field is not serializable.
        ret.pop("model_config", None)

        return GetInternalStateReqOutput(internal_state=ret)

    def set_internal_state(self, recv_req: SetInternalStateReq):
        server_args_dict = recv_req.server_args
        args_allow_update = set(
            [
                "pp_max_micro_batch_size",
                "speculative_accept_threshold_single",
                "speculative_accept_threshold_acc",
            ]
        )

        if_success = True
        for k, v in server_args_dict.items():
            if k not in args_allow_update:
                logging.warning(f"Updating {k} is not supported.")
                if_success = False
                break
            elif k == "pp_max_micro_batch_size" and (
                v > self.max_running_requests // self.ps.pp_size or v < 1
            ):
                logging.warning(
                    f"Updating {k} to {v} is rejected because it is out of the valid range [1, {self.max_running_requests // self.ps.pp_size}]."
                )
                if_success = False
                break

        if if_success:
            if (
                not self.spec_algorithm.is_none()
                and self.metrics_reporter.spec_total_num_forward_ct > 0
            ):
                avg_spec_accept_length = (
                    self.metrics_reporter.spec_total_num_accept_tokens
                    / self.metrics_reporter.spec_total_num_forward_ct
                )
                logger.info(f"{avg_spec_accept_length=}")
            self.metrics_reporter.spec_total_num_accept_tokens = (
                self.metrics_reporter.spec_total_num_forward_ct
            ) = 0
            for k, v in server_args_dict.items():
                setattr(get_global_server_args(), k, v)
            logger.info(f"Global server args updated! {get_global_server_args()=}")
        return SetInternalStateReqOutput(
            updated=True,
            server_args=vars(get_global_server_args()),
        )

    def save_remote_model(self, **kwargs):
        self.weight_updater.save_remote_model(kwargs)

    def save_sharded_model(self, **kwargs):
        self.weight_updater.save_sharded_model(kwargs)

    def handle_rpc_request(self, recv_req: RpcReqInput):
        # Handle RPC requests
        logger.info(
            f"handle_rpc_request: {recv_req.method}, param: {recv_req.parameters}"
        )

        success = True
        exec = None
        try:
            func = getattr(self, recv_req.method)
            if recv_req.parameters is not None:
                func(**recv_req.parameters)
            else:
                func()
        except Exception as e:
            success = False
            exec = e
            logger.error(f"Failed to call rpc {recv_req.method}: {str(e)}")

        barrier()
        return RpcReqOutput(success, "" if not exec else str(exec))

    def abort_request(self, recv_req: AbortReq):
        if (chunked_req := self.chunked_req) is not None:
            if recv_req.abort_all or chunked_req.rid.startswith(recv_req.rid):
                self._pending_chunked_abort_req = chunked_req

        # todo hisparse, release resources for abort requests in hisparse coordinator
        # Requests waiting for their parent generation have not entered any of
        # SGLang's ordinary queues yet, so abort them explicitly.
        remaining_agentic_waiters = []
        for req, started_at in self.agentic_kv_waiting_queue:
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                if getattr(self, "tp_size", 1) == 1:
                    self._agentic_forget_waiter(req)
                self._agentic_abort_cleanup(req)
                self.ipc_channels.send_to_tokenizer.send_output(
                    AbortReq(rid=req.rid), req
                )
            else:
                remaining_agentic_waiters.append((req, started_at))
        self.agentic_kv_waiting_queue = remaining_agentic_waiters

        # Delete requests in the waiting queue
        to_del = []
        for i, req in enumerate(self.waiting_queue):
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                to_del.append(i)

        # Sort in reverse order to avoid index issues when deleting
        for i in reversed(to_del):
            # Abort method 1: directly pop from the queue
            # This only works for requests that have not started anything.
            # We still need to send something back to TokenizerManager to clean up the state.
            req = self.waiting_queue.pop(i)
            self._agentic_abort_cleanup(req)
            if self.enable_hicache_storage:
                # to release prefetch events associated with the request
                self.tree_cache.release_aborted_request(req.rid)
            self.ipc_channels.send_to_tokenizer.send_output(AbortReq(rid=req.rid), req)
            # For disaggregation decode mode, the request in the waiting queue has KV cache allocated.
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                release_kv_cache(req, self.tree_cache)
            # For disaggregation prefill mode, free the metadata buffer index
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                bootstrap_pending = req.pending_bootstrap
                maybe_release_metadata_buffer(
                    req, self.req_to_metadata_buffer_idx_allocator
                )
                if (
                    bootstrap_pending
                    and hasattr(req, "disagg_kv_sender")
                    and req.disagg_kv_sender is not None
                ):
                    if hasattr(req.disagg_kv_sender, "abort"):
                        req.disagg_kv_sender.abort()

            # For mamba radix cache
            if (
                req.mamba_pool_idx is not None
                and self.disaggregation_mode != DisaggregationMode.DECODE
            ):
                release_kv_cache(req, self.tree_cache, is_insert=False)
            logger.debug(f"Abort queued request. {req.rid=}")

        # Delete the requests in the grammar queue
        # Abort method 2: call `set_finish_with_abort`
        # The request will still run one prefill forward pass.
        # In this case, we change the input_ids to be only one token to make this prefill cheap.
        self.grammar_manager.abort_requests(recv_req)

        # Delete requests not in the waiting queue when PD disaggregation is enabled
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # Abort requests that have not yet been bootstrapped
            for req in self.disagg_prefill_bootstrap_queue.queue:
                if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort bootstrap queue request. {req.rid=}")
                    self._agentic_abort_cleanup(req)
                    if hasattr(req.disagg_kv_sender, "abort"):
                        req.disagg_kv_sender.abort()

            # Abort in-flight requests
            for req in self.disagg_prefill_inflight_queue:
                if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort inflight queue request. {req.rid=}")
                    self._agentic_abort_cleanup(req)
                    if hasattr(req.disagg_kv_sender, "abort"):
                        req.disagg_kv_sender.abort()

        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            # Abort requests that have not yet finished preallocation
            for decode_req in self.disagg_decode_prealloc_queue.queue:
                if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort prealloc queue request. {decode_req.req.rid=}")
                    decode_req.kv_receiver.abort()

            # Abort requests waiting for kvcache to release tree cache
            for decode_req in self.disagg_decode_transfer_queue.queue:
                if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort transfer queue request. {decode_req.req.rid=}")
                    decode_req.kv_receiver.abort()

            # Abort requests already retracted to CPU cache
            if self.disagg_decode_prealloc_queue.retracted_queue:
                remaining_retracted = []
                for decode_req in self.disagg_decode_prealloc_queue.retracted_queue:
                    if recv_req.abort_all or decode_req.rid.startswith(recv_req.rid):
                        assert hasattr(decode_req, "kv_cache_cpu")
                        del decode_req.kv_cache_cpu
                        self.ipc_channels.send_to_tokenizer.send_output(
                            AbortReq(rid=decode_req.rid), decode_req
                        )
                    else:
                        remaining_retracted.append(decode_req)
                self.disagg_decode_prealloc_queue.retracted_queue = remaining_retracted

        # Delete requests in the running batch
        if self.cur_batch is self.running_batch or self.cur_batch is None:
            reqs = self.running_batch.reqs
        else:
            reqs = self.running_batch.reqs + self.cur_batch.reqs

        for req in reqs:
            if not req.finished() and (
                recv_req.abort_all or req.rid.startswith(recv_req.rid)
            ):
                # Abort method 3: set `to_finish`
                # The request will still run one decode forward pass.
                # Then we reuse all existing code to clean up the KV cache allocation.
                logger.debug(f"Abort running request. {req.rid=}")
                self._agentic_abort_cleanup(req)
                req.to_finish = FINISH_ABORT()

    def _pause_engine(self) -> Tuple[List[Req], int]:
        raise NotImplementedError()

    def pause_generation(self, recv_req: PauseGenerationReqInput):
        self._engine_paused = True

        if recv_req.mode == "in_place":
            # In-place pause: just set the flag and return immediately.
            # All scheduler state (running_batch, last_batch, chunked_req,
            # result_queue) is left untouched. On resume, the normal event
            # loop (get_next_batch_to_run) handles last_batch merge,
            # chunked_req cleanup, and overlap result processing through
            # the standard code paths. This avoids duplicating batch
            # manipulation logic and the accounting bugs that come with it.
            return

        if self.enable_overlap and self.last_batch:
            # Process the results of the last batch
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        if self.last_batch and self.last_batch.forward_mode.is_extend():
            chunked_req_to_exclude = set()
            self.last_batch.filter_batch(
                chunked_req_to_exclude=list(chunked_req_to_exclude)
            )
            # Skip merge for disagg prefill: completed prefill requests are
            # already in disagg_prefill_inflight_queue. Merging them into
            # running_batch leaks them, since the prefill event loop never
            # calls update_running_batch to clean them up.
            if (
                not self.last_batch.is_empty()
                and self.disaggregation_mode != DisaggregationMode.PREFILL
            ):
                if self.running_batch.is_empty():
                    self.running_batch = self.last_batch
                else:
                    self.running_batch.merge_batch(self.last_batch)

        self.last_batch = None
        self.cur_batch = None

        if recv_req.mode == "retract" and not self.running_batch.is_empty():
            self.running_batch.filter_batch()
            if len(self.running_batch.reqs) != 0:
                retracted_reqs = self.running_batch.retract_all(self.server_args)
                for req in retracted_reqs:
                    self._add_request_to_queue(req)

            self.running_batch.batch_is_full = False
            self.chunked_req = None

        # Surface the paused state to dashboards immediately. The scheduler
        # event loop short-circuits before reaching ``on_idle`` while paused,
        # so without this hop ``gen_throughput`` retains its last non-zero
        # value and KV events are not flushed for the entire pause window
        # (e.g. across a weight update). Zero the gauge, force a one-shot
        # idle log by resetting the rate-limit timestamp, and flush pending
        # KV events.
        self.metrics_reporter.last_gen_throughput = 0.0
        if self.metrics_reporter.current_scheduler_metrics_enabled:
            self.metrics_reporter.metrics_collector.last_log_time = 0.0
            self.metrics_reporter._maybe_log_idle_metrics()
        self.kv_events_publisher.publish_kv_events()

    def continue_generation(self, recv_req: ContinueGenerationReqInput):
        if recv_req.torch_empty_cache:
            before_mb = torch.cuda.memory_reserved() / (1024 * 1024)
            torch.cuda.empty_cache()
            after_mb = torch.cuda.memory_reserved() / (1024 * 1024)
            logger.info(
                f"[continue_generation] torch.cuda.empty_cache() called: "
                f"reserved {before_mb:.1f} MB -> {after_mb:.1f} MB "
                f"(freed {before_mb - after_mb:.1f} MB)"
            )
        self._engine_paused = False

    def load_lora_adapter(
        self, recv_req: LoadLoRAAdapterReqInput
    ) -> LoadLoRAAdapterReqOutput:
        """In-place loading a new lora adapter from disk or huggingface."""

        result = self.tp_worker.load_lora_adapter(recv_req)
        return result

    def load_lora_adapter_from_tensors(
        self, recv_req: LoadLoRAAdapterFromTensorsReqInput
    ) -> LoadLoRAAdapterFromTensorsReqOutput:
        """In-place loading a new lora adapter from serialized tensors."""

        result = self.tp_worker.load_lora_adapter_from_tensors(recv_req)
        return result

    def unload_lora_adapter(
        self, recv_req: UnloadLoRAAdapterReqInput
    ) -> UnloadLoRAAdapterReqOutput:
        """Unload the lora adapter."""

        result = self.tp_worker.unload_lora_adapter(recv_req)
        return result

    def init_weights_send_group_for_remote_instance(
        self, recv_req: InitWeightsSendGroupForRemoteInstanceReqInput
    ):
        """Init the seed and client instance communication group."""
        success, message = self.tp_worker.init_weights_send_group_for_remote_instance(
            recv_req
        )
        return InitWeightsSendGroupForRemoteInstanceReqOutput(success, message)

    def send_weights_to_remote_instance(
        self, recv_req: SendWeightsToRemoteInstanceReqInput
    ):
        """Send the seed instance weights to the destination instance."""
        success, message = self.tp_worker.send_weights_to_remote_instance(recv_req)
        return SendWeightsToRemoteInstanceReqOutput(success, message)

    def slow_down(self, recv_req: SlowDownReqInput):
        t = recv_req.forward_sleep_time
        if t is not None and t <= 0:
            t = None
        self.forward_sleep_time = t
        return SlowDownReqOutput()

    def expert_distribution_handle(self, recv_req: ExpertDistributionReq):
        action = recv_req.action
        if action == ExpertDistributionReqType.START_RECORD:
            get_global_expert_distribution_recorder().start_record()
        elif action == ExpertDistributionReqType.STOP_RECORD:
            get_global_expert_distribution_recorder().stop_record()
        elif action == ExpertDistributionReqType.DUMP_RECORD:
            get_global_expert_distribution_recorder().dump_record()
        else:
            raise ValueError(f"Unrecognized ExpertDistributionReq value: {recv_req=}")
        return ExpertDistributionReqOutput()

    def open_session(self, recv_req: OpenSessionReqInput):
        output = self.session_controller.open(recv_req)
        if self.ps.pp_rank == 0 and self.ps.tp_rank == 0 and self.ps.attn_cp_rank == 0:
            return output
        return None

    def close_session(self, recv_req: CloseSessionReqInput):
        self.session_controller.close(recv_req)

    def maybe_sleep_on_idle(self):
        if self.idle_sleeper is not None:
            self.idle_sleeper.maybe_sleep()

    def handle_freeze_gc(self, recv_req: FreezeGCReq):
        """Handle freeze_gc request: freeze scheduler's GC and forward to detokenizer."""
        freeze_gc("Scheduler")
        self.ipc_channels.send_to_detokenizer.send_output(recv_req, recv_req)
        return None

    def handle_shutdown(self, recv_req: ShutdownReq):
        # Break the event loop; the finally in run_scheduler_process releases resources.
        self.gracefully_exit = True
        return None

    def configure_logging(self, recv_req: ConfigureLoggingReq):
        if recv_req.log_level is not None:
            logging.getLogger().setLevel(recv_req.log_level.upper())
        self.ipc_channels.send_to_detokenizer.send_output(recv_req, recv_req)

    def handle_dumper_control(self, recv_req: DumperControlReqInput):
        from sglang.srt.debug_utils.dumper import dumper

        try:
            response: list = []
            if (
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ):
                response = dumper._http_manager.handle_request(
                    method=recv_req.method, body=recv_req.body
                )
            self.ipc_channels.send_to_tokenizer.send_output(
                DumperControlReqOutput(success=True, response=response), recv_req
            )
        except Exception as e:
            print(f"[Scheduler] handle_dumper_control error: {e}", flush=True)
            self.ipc_channels.send_to_tokenizer.send_output(
                DumperControlReqOutput(success=False, response=[], error=str(e)),
                recv_req,
            )

    # placeholder for override
    def update_cache_from_scheduler(
        self, schedule_batch: ScheduleBatch, batch_result: GenerationBatchResult
    ):
        pass


def dispatch_event_loop(scheduler: Scheduler):
    # Dispatch to the appropriate event loop based on the disaggregation mode
    server_args = scheduler.server_args
    disaggregation_mode: DisaggregationMode = scheduler.disaggregation_mode
    if disaggregation_mode == DisaggregationMode.NULL:
        if scheduler.enable_pdmux:
            scheduler.event_loop_pdmux()
        elif server_args.pp_size > 1:
            scheduler.event_loop_pp()
        elif scheduler.enable_overlap_mlx:
            scheduler.event_loop_overlap_mlx()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap()
        else:
            scheduler.event_loop_normal()
    elif disaggregation_mode == DisaggregationMode.PREFILL:
        if server_args.pp_size > 1:
            scheduler.event_loop_pp_disagg_prefill()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap_disagg_prefill()
        else:
            scheduler.event_loop_normal_disagg_prefill()
    elif disaggregation_mode == DisaggregationMode.DECODE:
        if server_args.pp_size > 1:
            scheduler.event_loop_pp_disagg_decode()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap_disagg_decode()
        else:
            scheduler.event_loop_normal_disagg_decode()


def configure_scheduler_process(
    server_args: ServerArgs,
    gpu_id: int,
    tp_rank: int,
    attn_cp_rank: int,
    moe_dp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
) -> Optional[int]:
    """Configure scheduler worker: logging, process title, etc.

    Returns:
        dp_rank
    """
    kill_itself_when_parent_died()

    # Generate the logger prefix
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
        dp_rank = int(os.environ["SGLANG_DP_RANK"])

    prefix = ""
    if dp_rank is not None:
        prefix += f" DP{dp_rank}"
    if server_args.pp_size > 1:
        prefix += f" PP{pp_rank}"
    if server_args.attn_cp_size > 1:
        prefix += f" ATTN_CP{attn_cp_rank}"
    if server_args.moe_dp_size > 1:
        prefix += f" MOE_DP{moe_dp_rank}"
    if server_args.tp_size > 1:
        prefix += f" TP{tp_rank}"
    if server_args.ep_size > 1:
        prefix += f" EP{moe_ep_rank}"

    # Config the process
    setproctitle.setproctitle(f"sglang::scheduler{prefix.replace(' ', '_')}")
    faulthandler.enable()
    if os.getenv("SGLANG_AGENTIC_DEBUG_STACK_SIGNAL", "0") == "1":
        faulthandler.register(signal.SIGUSR2, all_threads=True)

    # Configure the logger
    configure_logger(server_args, prefix=prefix)
    suppress_other_loggers()

    # Set cpu affinity to this gpu process
    if envs.SGLANG_SET_CPU_AFFINITY.get():
        set_gpu_proc_affinity(
            server_args.pp_size, server_args.tp_size, server_args.nnodes, gpu_id
        )
    if not envs.SGLANG_NUMA_BIND_V2.get():
        numa_node = get_numa_node_if_available(server_args, gpu_id)
        if numa_node is not None:
            numa_bind_to_node(numa_node)

    return dp_rank


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    attn_cp_rank: int,
    moe_dp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
):
    # Load plugins so hooks can override Scheduler and its dependencies.
    load_plugins()
    dp_rank = configure_scheduler_process(
        server_args,
        gpu_id,
        tp_rank,
        attn_cp_rank,
        moe_dp_rank,
        moe_ep_rank,
        pp_rank,
        dp_rank,
    )
    parent_process = psutil.Process().parent()

    # Set up tracing
    if server_args.enable_trace:
        process_tracing_init(
            server_args.otlp_traces_endpoint,
            "sglang",
            trace_modules=server_args.trace_modules,
        )
        thread_label = "Scheduler"
        if server_args.disaggregation_mode == "prefill":
            thread_label = "Prefill Scheduler"
        elif server_args.disaggregation_mode == "decode":
            thread_label = "Decode Scheduler"
        trace_set_thread_info(thread_label, tp_rank, dp_rank, pp_rank)

    # Create a scheduler and run the event loop
    scheduler = None
    try:
        scheduler = Scheduler(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            attn_cp_rank,
            moe_dp_rank,
            dp_rank,
        )

        # Send initialization info back to the parent process
        pipe_writer.send(scheduler.get_init_info())

        # Run the event loop (blocks until a ShutdownReq sets gracefully_exit)
        scheduler.run_event_loop()

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
        # Opt-in: SIGKILL the pgroup so sibling ranks don't spew thousands
        # of NCCL/TCPStore tracebacks before they finally die.
        if envs.SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION.get():
            try:
                os.killpg(os.getpgrp(), signal.SIGKILL)
            except Exception:
                pass
    finally:
        if scheduler is not None:
            # FPM has a background ZMQ publisher thread that needs explicit
            # teardown to flush queued metrics and close the socket cleanly.
            scheduler.metrics_reporter._shutdown_fpm()
            # Graceful path only: on the exception path the GPU may be wedged
            # and the synchronize() in destroy() could itself hang.
            if scheduler.gracefully_exit:
                scheduler.release_host_resources()
