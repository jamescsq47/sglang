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
import faulthandler
import json
import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

from sglang.srt.utils.common import suppress_noisy_warnings

suppress_noisy_warnings()

import psutil
import numpy as np
import setproctitle
import torch
import torch.distributed
import zmq
from torch.cuda import Stream as CudaStream
from torch.distributed import barrier

from sglang.jit_kernel.ngram_embedding import update_token_table
from sglang.srt.configs.model_config import ModelConfig, ModelImpl
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.constrained.grammar_manager import GrammarManager
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
    release_req_to_metadata_buffer,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    FAKE_BOOTSTRAP_HOST,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    kv_to_page_indices,
    prepare_abort,
)
from sglang.srt.distributed import get_pp_group, get_world_group
from sglang.srt.distributed.parallel_state import get_tp_group
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
from sglang.srt.lora.lora_overlap_loader import LoRAOverlapLoader
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.managers.io_struct import (
    AbortReq,
    ActiveRanksOutput,
    AttachHiCacheStorageReqInput,
    AttachHiCacheStorageReqOutput,
    BaseBatchReq,
    BaseReq,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    CheckWeightsReqInput,
    ClearHiCacheReqInput,
    ClearHiCacheReqOutput,
    CloseSessionReqInput,
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
    FlushCacheReqOutput,
    FreezeGCReq,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    GetLoadReqInput,
    GetLoadsReqInput,
    GetWeightsByNameReqInput,
    HealthCheckOutput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsSendGroupForRemoteInstanceReqOutput,
    InitWeightsUpdateGroupReqInput,
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterFromTensorsReqOutput,
    LoadLoRAAdapterReqInput,
    LoadLoRAAdapterReqOutput,
    OpenSessionReqInput,
    PauseGenerationReqInput,
    ProfileReq,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    RpcReqInput,
    RpcReqOutput,
    SendWeightsToRemoteInstanceReqInput,
    SendWeightsToRemoteInstanceReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
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
from sglang.srt.managers.mm_utils import (
    has_shm_features,
    init_mm_embedding_cache,
    unwrap_shm_features,
)
from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.managers.prefill_delayer import (
    PrefillDelayer,
    PrefillDelayerSinglePassExecutor,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    ModelWorkerBatch,
    MultimodalInputs,
    Req,
    ScheduleBatch,
)
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
    SchedulePolicy,
)
from sglang.srt.managers.scheduler_dp_attn_mixin import SchedulerDPAttnMixin
from sglang.srt.managers.scheduler_input_blocker import SchedulerInputBlocker
from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.managers.scheduler_profiler_mixin import SchedulerProfilerMixin
from sglang.srt.managers.scheduler_recv_skipper import SchedulerRecvSkipper
from sglang.srt.managers.scheduler_runtime_checker_mixin import (
    SchedulerRuntimeCheckerMixin,
    create_scheduler_watchdog,
)
from sglang.srt.managers.scheduler_update_weights_mixin import (
    SchedulerUpdateWeightsMixin,
)
from sglang.srt.managers.session_controller import SessionController
from sglang.srt.managers.utils import GenerationBatchResult, validate_input_length
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.session_aware_cache import SessionAwareCache
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.model_loader.utils import get_resolved_model_impl
from sglang.srt.multiplex.multiplexing_mixin import SchedulerMultiplexMixin
from sglang.srt.observability.req_time_stats import (
    real_time,
    set_schedule_time_batch,
    set_time_batch,
)
from sglang.srt.observability.scheduler_metrics_mixin import (
    RECORD_STEP_TIME,
    PrefillStats,
    SchedulerMetricsMixin,
)
from sglang.srt.observability.trace import process_tracing_init, trace_set_thread_info
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import PortArgs, ServerArgs, get_global_server_args
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    DynamicGradMode,
    broadcast_pyobj,
    configure_gc_logger,
    configure_logger,
    freeze_gc,
    get_available_gpu_memory,
    get_bool_env_var,
    get_int_env_var,
    is_mps,
    kill_itself_when_parent_died,
    point_to_point_pyobj,
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
from sglang.srt.utils.network import get_zmq_socket
from sglang.srt.utils.numa_utils import get_numa_node_if_available, numa_bind_to_node
from sglang.srt.utils.tensor_bridge import use_mlx
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

if is_mps():
    CudaStreamContext = nullcontext
else:
    from torch.cuda import StreamContext as CudaStreamContext

logger = logging.getLogger(__name__)


@dataclass
class AgenticPWorksetLease:
    """Physical P-HBM ownership for one complete next-turn prompt.

    The parent slice is the destination of Direct or Slow restore.  The suffix
    slice stays unavailable to every other request until this exact request is
    admitted for incremental Prefill.  All allocator mutations are performed
    by the model scheduler; I/O workers consume only immutable page ids.
    """

    snapshot_id: str
    lease_id: int
    owner: str
    parent_tokens: int
    parent_allocated_tokens: int
    prompt_tokens: int
    allocated_tokens: int
    device_indices: torch.Tensor
    parent_page_indices: np.ndarray
    parent_bound: bool = False
    state: str = "active"
    suffix_cursor: int = 0
    io_attempt: Optional[str] = None

    @property
    def parent_indices(self) -> torch.Tensor:
        return self.device_indices[: self.parent_allocated_tokens]

    @property
    def suffix_indices(self) -> torch.Tensor:
        return self.device_indices[self.parent_allocated_tokens :]

    @property
    def suffix_allocated_tokens(self) -> int:
        return self.allocated_tokens - self.parent_allocated_tokens

    @property
    def remaining_suffix_indices(self) -> torch.Tensor:
        if self.state == "consumed":
            return self.suffix_indices[:0]
        return self.suffix_indices[self.suffix_cursor :]


class AgenticPWorksetLeaseBroker:
    """Thread-safe intent queue with scheduler-owned physical allocation."""

    def __init__(self, page_size: int):
        self.page_size = int(page_size)
        self._intents: Dict[str, Tuple[str, int, int]] = {}
        self._leases: Dict[str, AgenticPWorksetLease] = {}
        self._release_requested: Dict[str, int] = {}
        self._next_lease_id = 1
        self._grants = 0
        self._allocation_failures = 0
        self._lock = threading.RLock()

    @staticmethod
    def direct_owner(snapshot_id: str) -> str:
        return f"direct:{snapshot_id}"

    @staticmethod
    def slow_owner(snapshot_id: str, rid: str) -> str:
        return f"slow:{snapshot_id}:{rid}"

    def request(
        self,
        snapshot_id: str,
        parent_tokens: int,
        prompt_tokens: int,
        *,
        owner: str = "legacy",
    ) -> bool:
        parent_tokens = int(parent_tokens)
        prompt_tokens = int(prompt_tokens)
        if parent_tokens <= 0 or prompt_tokens < parent_tokens:
            raise ValueError(
                f"invalid workset shape parent={parent_tokens} prompt={prompt_tokens}"
            )
        with self._lock:
            current = self._leases.get(snapshot_id)
            if current is not None:
                return (
                    current.owner == owner
                    and current.parent_tokens == parent_tokens
                    and current.prompt_tokens == prompt_tokens
                    and current.state not in {"releasing", "consumed"}
                )
            pending = self._intents.get(snapshot_id)
            if pending is not None:
                return pending == (owner, parent_tokens, prompt_tokens)
            if snapshot_id in self._release_requested:
                return False
            self._intents[snapshot_id] = (owner, parent_tokens, prompt_tokens)
            return True

    def get(
        self, snapshot_id: str, *, owner: Optional[str] = None
    ) -> Optional[AgenticPWorksetLease]:
        with self._lock:
            lease = self._leases.get(snapshot_id)
            if lease is not None and (owner is None or lease.owner == owner):
                return lease
            return None

    def request_release(
        self,
        snapshot_id: str,
        lease: Optional[AgenticPWorksetLease] = None,
        *,
        owner: Optional[str] = None,
        io_attempt: Optional[str] = None,
    ) -> bool:
        """Schedule release of the exact lease owned by the caller.

        A caller without a lease identity cannot release anything; intent
        cancellation is a separate owner-scoped operation.  This prevents a
        delayed Direct callback from cancelling a newer Slow owner for the
        same request-generation.
        """

        with self._lock:
            if lease is None:
                return False
            current = self._leases.get(snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                return False
            if owner is not None and current.owner != owner:
                return False
            pending = self._intents.get(snapshot_id)
            if pending is not None and pending[0] == current.owner:
                self._intents.pop(snapshot_id, None)
            if current.state in {"io_reserved", "io_inflight"}:
                if current.io_attempt != io_attempt:
                    return False
                current.state = "release_pending"
                return False
            if current.state in {
                "release_pending",
                "binding",
                "handed",
                "consumed",
                "releasing",
            }:
                return False
            current.state = "releasing"
            self._release_requested[snapshot_id] = current.lease_id
            return True

    def release_handed(
        self,
        snapshot_id: str,
        lease: AgenticPWorksetLease,
        *,
        req,
    ) -> bool:
        """Release a workset after scheduler ownership was handed to ``req``.

        Transport callbacks deliberately cannot release a handed lease.  Only
        the live request that owns the suffix may return it on cancellation.
        This separates request lifetime from stale Direct/Slow completions.
        """

        with self._lock:
            current = self._leases.get(snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                return False
            if current.state != "handed":
                return False
            if getattr(req, "_agentic_p_workset_lease", None) is not current:
                return False
            current.state = "releasing"
            self._release_requested[snapshot_id] = current.lease_id
            return True

    def service(self, allocator, *, reserve_tokens: int = 0) -> None:
        """Allocate/free only at a scheduler-safe boundary.

        ``reserve_tokens`` protects the unfinished suffix of the one native
        chunked-Prefill request.  That request was admitted only after the
        scheduler proved that its complete prompt fitted, but its later
        chunks are allocated lazily.  Background Direct/Slow intents must not
        consume that already-promised capacity between chunks.
        """

        reserve_tokens = max(0, int(reserve_tokens))

        with self._lock:
            releases = tuple(self._release_requested.items())
            self._release_requested.clear()
            for snapshot_id, expected_id in releases:
                lease = self._leases.get(snapshot_id)
                if (
                    lease is not None
                    and lease.lease_id == expected_id
                    and lease.state == "releasing"
                ):
                    self._leases.pop(snapshot_id, None)
                    allocator.free(
                        lease.remaining_suffix_indices
                        if lease.parent_bound
                        else lease.device_indices
                    )

            for snapshot_id, (owner, parent_tokens, prompt_tokens) in tuple(
                self._intents.items()
            ):
                # Parent and suffix have distinct ownership transitions: the
                # parent is filled by Direct/Slow I/O, while the suffix is
                # filled by incremental Prefill.  Round each slice
                # independently so an unaligned parent can never steal the
                # first page required by the suffix.
                parent_allocated = (
                    (parent_tokens + self.page_size - 1) // self.page_size
                ) * self.page_size
                suffix_tokens = prompt_tokens - parent_tokens
                suffix_allocated = (
                    (suffix_tokens + self.page_size - 1) // self.page_size
                ) * self.page_size
                allocated_tokens = parent_allocated + suffix_allocated
                if reserve_tokens and (
                    allocator.available_size() - reserve_tokens < allocated_tokens
                ):
                    self._allocation_failures += 1
                    continue
                device_indices = allocator.alloc(allocated_tokens)
                if device_indices is None:
                    self._allocation_failures += 1
                    continue
                parent_indices = device_indices[:parent_allocated]
                page_indices = kv_to_page_indices(
                    parent_indices.cpu().numpy(), self.page_size
                )
                self._leases[snapshot_id] = AgenticPWorksetLease(
                    snapshot_id=snapshot_id,
                    lease_id=self._next_lease_id,
                    owner=owner,
                    parent_tokens=parent_tokens,
                    parent_allocated_tokens=parent_allocated,
                    prompt_tokens=prompt_tokens,
                    allocated_tokens=allocated_tokens,
                    device_indices=device_indices,
                    parent_page_indices=page_indices,
                )
                self._next_lease_id += 1
                self._grants += 1
                self._intents.pop(snapshot_id, None)

    def handoff_to_req(
        self, snapshot_id: str, req, lease: AgenticPWorksetLease
    ) -> None:
        """Move the complete lease from broker ownership to one live Req."""

        with self._lock:
            current = self._leases.get(snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                raise RuntimeError(f"workset lease disappeared for {snapshot_id}")
            if current.state != "binding":
                raise RuntimeError(
                    f"workset lease is {current.state} for {snapshot_id}"
                )
            if not current.parent_bound:
                raise RuntimeError(f"workset parent is not bound for {snapshot_id}")
            actual_prompt_tokens = len(req.origin_input_ids)
            if actual_prompt_tokens != current.prompt_tokens:
                raise RuntimeError(
                    f"workset prompt changed for {snapshot_id}: "
                    f"reserved={current.prompt_tokens} actual={actual_prompt_tokens}"
                )
            current.state = "handed"
            self._intents.pop(snapshot_id, None)
            req._agentic_p_workset_lease = current
            req._agentic_p_workset_broker = self
            req._agentic_workset_suffix_indices = current.remaining_suffix_indices

    def consume_suffix(
        self,
        lease: AgenticPWorksetLease,
        extend_tokens: int,
        *,
        final_prompt_chunk: bool,
    ) -> torch.Tensor:
        """Transfer suffix slots to one Prefill chunk.

        The returned tensor has the length expected by SGLang's extend
        batch.  On the final logical prompt chunk, ownership of the whole
        final KV page (including unused padding slots) moves to the request,
        so the broker drops the lease without freeing that padding.
        """

        extend_tokens = int(extend_tokens)
        if extend_tokens < 0:
            raise ValueError("extend_tokens must be non-negative")
        with self._lock:
            current = self._leases.get(lease.snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                raise RuntimeError(f"workset lease disappeared for {lease.snapshot_id}")
            if current.state != "handed":
                raise RuntimeError(
                    f"workset lease is {current.state} for {lease.snapshot_id}"
                )
            start = current.suffix_cursor
            end = start + extend_tokens
            logical_suffix_tokens = current.prompt_tokens - current.parent_tokens
            physical_suffix_tokens = current.suffix_allocated_tokens
            if end > physical_suffix_tokens:
                raise RuntimeError(
                    f"workset suffix over-consumed for {lease.snapshot_id}: "
                    f"end={end} physical={physical_suffix_tokens}"
                )
            if final_prompt_chunk and end < logical_suffix_tokens:
                raise RuntimeError(
                    f"final workset chunk is incomplete for {lease.snapshot_id}: "
                    f"end={end} logical={logical_suffix_tokens}"
                )
            if (
                not final_prompt_chunk
                and end < physical_suffix_tokens
                and end % self.page_size
            ):
                raise RuntimeError(
                    "non-final chunked Prefill must end on a KV page boundary"
                )
            indices = current.suffix_indices[start:end]
            current.suffix_cursor = end
            if final_prompt_chunk or end == physical_suffix_tokens:
                current.state = "consumed"
                self._leases.pop(lease.snapshot_id, None)
            return indices

    def begin_bind(self, snapshot_id: str, lease: AgenticPWorksetLease) -> bool:
        """Atomically transfer a completed I/O lease to scheduler binding."""

        with self._lock:
            current = self._leases.get(snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                return False
            if current.state != "active":
                return current.state == "binding"
            current.state = "binding"
            return True

    def begin_io_attempt(
        self,
        snapshot_id: str,
        lease: AgenticPWorksetLease,
        attempt: str,
    ) -> bool:
        """Exclusively reserve one lease for one concrete Direct session."""

        with self._lock:
            current = self._leases.get(snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                return False
            if current.state != "active":
                return False
            current.state = "io_reserved"
            current.io_attempt = str(attempt)
            return True

    def mark_io_inflight(
        self,
        snapshot_id: str,
        lease: AgenticPWorksetLease,
        attempt: str,
    ) -> None:
        """Fence allocator reuse after this attempt publishes destinations."""

        with self._lock:
            current = self._leases.get(snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                raise RuntimeError(f"workset lease disappeared for {snapshot_id}")
            if current.state != "io_reserved" or current.io_attempt != attempt:
                raise RuntimeError(
                    f"cannot start attempt={attempt} on {current.state} "
                    f"workset {snapshot_id} owned_by={current.io_attempt}"
                )
            current.state = "io_inflight"

    def cancel_io_attempt(
        self,
        snapshot_id: str,
        lease: AgenticPWorksetLease,
        attempt: str,
    ) -> bool:
        """Drop an exclusive attempt before any remote write can begin."""

        with self._lock:
            current = self._leases.get(snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                return False
            if current.state != "io_reserved" or current.io_attempt != attempt:
                return False
            current.state = "active"
            current.io_attempt = None
            return True

    def mark_io_quiesced(
        self,
        snapshot_id: str,
        lease: AgenticPWorksetLease,
        attempt: str,
    ) -> bool:
        """Publish a definitive transport terminal state to the allocator."""

        with self._lock:
            current = self._leases.get(snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                return False
            if current.io_attempt != attempt:
                return False
            if current.state == "io_inflight":
                current.state = "active"
                current.io_attempt = None
                return True
            if current.state == "release_pending":
                current.state = "releasing"
                current.io_attempt = None
                self._release_requested[snapshot_id] = current.lease_id
                return True
            return False

    def commit_parent_bound(
        self, snapshot_id: str, lease: AgenticPWorksetLease
    ) -> None:
        with self._lock:
            current = self._leases.get(snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                raise RuntimeError(f"workset lease disappeared for {snapshot_id}")
            if current.state != "binding":
                raise RuntimeError(
                    f"cannot bind {current.state} workset lease for {snapshot_id}"
                )
            current.parent_bound = True

    def abort_bind(
        self,
        snapshot_id: str,
        lease: AgenticPWorksetLease,
        *,
        parent_bound: bool,
    ) -> bool:
        """Release a scheduler-owned bind after its Radix mutation is undone."""

        with self._lock:
            current = self._leases.get(snapshot_id)
            if current is None or current.lease_id != lease.lease_id:
                return False
            if current.state != "binding":
                return False
            current.parent_bound = bool(parent_bound)
            current.state = "releasing"
            self._release_requested[snapshot_id] = current.lease_id
            return True

    def cancel_unstarted(
        self, snapshot_id: str, *, owner: Optional[str] = None
    ) -> bool:
        """Atomically cancel work that has not started physical I/O.

        The scheduler can grant an intent between two progress-worker passes.
        Cancellation must therefore cover both representations: a pending
        intent and an ``active`` lease.  Once I/O, binding, or request handoff
        begins, the corresponding owner-specific terminal path is solely
        responsible for release.
        """

        with self._lock:
            cancelled = False
            pending = self._intents.get(snapshot_id)
            if pending is not None and (owner is None or pending[0] == owner):
                self._intents.pop(snapshot_id, None)
                cancelled = True
            lease = self._leases.get(snapshot_id)
            if (
                lease is not None
                and lease.state == "active"
                and (owner is None or lease.owner == owner)
            ):
                lease.state = "releasing"
                self._release_requested[snapshot_id] = lease.lease_id
                cancelled = True
            return cancelled

    @property
    def leased_tokens(self) -> int:
        with self._lock:
            return sum(lease.allocated_tokens for lease in self._leases.values())

    @property
    def stats(self) -> Tuple[int, int, int]:
        """Return pending intents, grants, and allocation misses."""

        with self._lock:
            return len(self._intents), self._grants, self._allocation_failures

    @property
    def lease_state_summary(self) -> str:
        """Compact count/token ownership summary for progress diagnostics."""

        with self._lock:
            counts: Dict[str, int] = {}
            tokens: Dict[str, int] = {}
            for lease in self._leases.values():
                counts[lease.state] = counts.get(lease.state, 0) + 1
                tokens[lease.state] = tokens.get(lease.state, 0) + int(
                    lease.allocated_tokens
                )
            return ",".join(
                f"{state}:{counts[state]}/{tokens[state]}"
                for state in sorted(counts)
            ) or "empty"

    @property
    def unaccounted_tokens(self) -> int:
        """Lease pages not already represented by a bound Radix parent."""

        with self._lock:
            return sum(
                (
                    lease.remaining_suffix_indices.numel()
                    if lease.parent_bound
                    else lease.allocated_tokens
                )
                for lease in self._leases.values()
            )


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


# Test retract decode for debugging purposes
TEST_RETRACT = envs.SGLANG_TEST_RETRACT.get()
TEST_RETRACT_INTERVAL = envs.SGLANG_TEST_RETRACT_INTERVAL.get()
TEST_RETRACT_NO_PREFILL_BS = envs.SGLANG_TEST_RETRACT_NO_PREFILL_BS.get()

_is_npu = is_npu()


@dataclass
class EmbeddingBatchResult:
    embeddings: torch.Tensor
    copy_done: Optional[torch.cuda.Event] = None

    def copy_to_cpu(self):
        """Copy embeddings tensor to CPU in overlap scheduling."""

        if isinstance(self.embeddings, torch.Tensor):
            self.copy_done = torch.get_device_module(self.embeddings.device).Event()
            self.embeddings = self.embeddings.to("cpu", non_blocking=True)
        else:
            assert isinstance(self.embeddings, list)
            if len(self.embeddings) == 0:
                return

            self.copy_done = torch.get_device_module(self.embeddings[0].device).Event()
            self.embeddings = [
                emb.to("cpu", non_blocking=True) for emb in self.embeddings
            ]

        self.copy_done.record()


class Scheduler(
    SchedulerOutputProcessorMixin,
    SchedulerUpdateWeightsMixin,
    SchedulerProfilerMixin,
    SchedulerMetricsMixin,
    SchedulerDisaggregationDecodeMixin,
    SchedulerDisaggregationPrefillMixin,
    SchedulerMultiplexMixin,
    SchedulerRuntimeCheckerMixin,
    SchedulerPPMixin,
    SchedulerDPAttnMixin,
    SchedulerDllmMixin,
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
        self.init_soft_watchdog(server_args)

        # Parse args
        self.server_args = server_args
        self.tp_rank = tp_rank
        self.moe_ep_rank = moe_ep_rank
        self.pp_rank = pp_rank
        self.attn_cp_rank = attn_cp_rank
        self.attn_cp_size = server_args.attn_cp_size
        self.moe_dp_rank = moe_dp_rank
        self.moe_dp_size = server_args.moe_dp_size
        self.dp_rank = dp_rank
        self.tp_size = server_args.tp_size
        self.moe_ep_size = server_args.ep_size
        self.pp_size = server_args.pp_size
        self.dp_size = server_args.dp_size
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
        self.enable_overlap = not server_args.disable_overlap_schedule
        self.enable_pdmux = server_args.enable_pdmux
        self.skip_tokenizer_init = server_args.skip_tokenizer_init
        self.stream_interval = server_args.stream_interval
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.gpu_id = gpu_id
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
        self.enable_hicache_storage = (
            server_args.hicache_storage_backend is not None and not custom_storage_only
        )
        self.max_recv_per_poll = envs.SGLANG_SCHEDULER_MAX_RECV_PER_POLL.get()
        self.enable_hisparse = server_args.enable_hisparse
        self.hisparse_coordinator: Optional[HiSparseCoordinator] = None

        # Distributed rank info
        self.attn_tp_rank, self.attn_tp_size, self.attn_dp_rank = (
            compute_dp_attention_world_info(
                server_args.enable_dp_attention,
                self.tp_rank,
                self.tp_size,
                self.dp_size,
                self.attn_cp_size,
            )
        )

        self.enable_kv_cache_events = bool(
            server_args.kv_events_config and self.attn_tp_rank == 0
        )

        # Init model configs
        self.init_model_config()

        # Init metrics stats
        self.init_metrics(tp_rank, pp_rank, dp_rank)

        # Init inter-process communication
        self.init_ipc_channels(port_args)

        # Init PD-multiplexing context
        if self.enable_pdmux:
            self.init_pdmux()

        # Init tokenizer
        self.init_tokenizer()

        # Init moe config and GEMM config (FP8 GEMM, etc.)
        self.init_moe_gemm_config()

        # Init mamba backend
        self.init_mamba_backend()

        # Launch a model worker and draft model worker if using speculative decoding
        self.init_model_worker()

        if (t := envs.SGLANG_TEST_STUCK_SCHEDULER_INIT.get()) > 0:
            time.sleep(t)

        # Init cache and memory pool
        self.init_cache_with_memory_pool()

        # Init running status
        self.init_running_status()

        # Init chunked prefill
        self.init_chunked_prefill()

        # Init diffusion LLM
        self.init_diffusion_llm()

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

        # Init request dispatcher
        self.init_request_dispatcher()

        # Init LoRA overlap loader
        if self.enable_lora_overlap_loading:
            self.lora_overlap_loader = LoRAOverlapLoader(
                self.tp_worker.model_runner.lora_manager
            )

        # Init the grammar backend for constrained generation
        self.grammar_manager = GrammarManager(self)

        self.is_initializing = False

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
        context = zmq.Context(2)
        self.idle_sleeper = None

        if self.pp_rank == 0 and self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )
            self.recv_from_rpc = get_zmq_socket(
                context, zmq.DEALER, port_args.rpc_ipc_name, False
            )

            send_to_tokenizer = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_ipc_name, False
            )
            if self.server_args.skip_tokenizer_init:
                # Directly send to the TokenizerManager
                send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.tokenizer_ipc_name, False
                )
            else:
                # Send to the DetokenizerManager
                send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.detokenizer_ipc_name, False
                )

            self.send_to_tokenizer = SenderWrapper(send_to_tokenizer)
            self.send_to_detokenizer = SenderWrapper(send_to_detokenizer)

            if self.server_args.sleep_on_idle:
                self.idle_sleeper = IdleSleeper(
                    [
                        self.recv_from_tokenizer,
                        self.recv_from_rpc,
                    ]
                )
        else:
            self.recv_from_tokenizer = None
            self.recv_from_rpc = None
            self.send_to_tokenizer = SenderWrapper(None)
            self.send_to_detokenizer = SenderWrapper(None)

        if self.current_scheduler_metrics_enabled:
            self.send_metrics_from_scheduler = get_zmq_socket(
                context, zmq.PUSH, port_args.metrics_ipc_name, False
            )

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
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
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
            self.tokenizer.think_end_id = self.tokenizer.encode(
                reasoning_parser.detector.think_end_token, add_special_tokens=False
            )[0]
            self._think_end_id = self.tokenizer.think_end_id
        else:
            self._think_end_id = None

    def init_mamba_backend(self) -> None:
        initialize_mamba_selective_state_update_backend(self.server_args)

    def init_moe_gemm_config(self):
        # For the MM models, check the text_config for MoE settings
        config_to_check = getattr(
            self.model_config.hf_config, "text_config", self.model_config.hf_config
        )

        if hasattr(config_to_check, "num_experts_per_tok"):
            initialize_moe_config(self.server_args)

        # Initialize GEMM-related configuration for FP8 and FP4 backends.
        initialize_fp8_gemm_config(self.server_args)
        initialize_fp4_gemm_config(self.server_args)

        # This must be called after initialize_moe_config
        self.require_mlp_sync = require_mlp_sync(self.server_args)

    def init_tp_model_worker(self):

        worker_kwargs = dict(
            server_args=self.server_args,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            moe_ep_rank=self.moe_ep_rank,
            pp_rank=self.pp_rank,
            attn_cp_rank=self.attn_cp_rank,
            moe_dp_rank=self.moe_dp_rank,
            dp_rank=self.dp_rank,
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
            return

        # Launch a draft worker for speculative decoding
        draft_worker_kwargs = dict(
            server_args=self.server_args,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            moe_ep_rank=self.moe_ep_rank,
            nccl_port=self.nccl_port,
            target_worker=self.tp_worker,
            dp_rank=self.dp_rank,
            attn_cp_rank=self.attn_cp_rank,
            moe_dp_rank=self.moe_dp_rank,
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

    def init_model_worker(self):
        self.init_tp_model_worker()
        self.maybe_init_draft_worker()

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
        if get_global_server_args().pp_max_micro_batch_size is None:
            get_global_server_args().pp_max_micro_batch_size = max(
                self.max_running_requests // self.pp_size, 1
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

        self.pad_input_ids_func = self.tp_worker.get_pad_input_ids_func()
        set_random_seed(self.random_seed)

        # Print debug info
        if self.tp_rank == 0:
            avail_mem = get_available_gpu_memory(
                self.device, self.gpu_id, empty_cache=False
            )
            logger.info(
                f"max_total_num_tokens={self.max_total_num_tokens}, "
                f"chunked_prefill_size={self.server_args.chunked_prefill_size}, "
                f"max_prefill_tokens={self.max_prefill_tokens}, "
                f"max_running_requests={self.max_running_requests}, "
                f"context_len={self.model_config.context_len}, "
                f"{'available_cpu_mem' if self.device == 'cpu' else 'available_gpu_mem'}={avail_mem:.2f} GB"
            )

        if self.enable_metrics and hasattr(self, "metrics_collector"):
            self.metrics_collector.emit_cache_config_info(
                self.page_size, self.max_total_num_tokens // self.page_size
            )

    def init_cache_with_memory_pool(self):
        server_args = self.server_args
        uses_transformers_backend = (
            get_resolved_model_impl(self.model_config) == ModelImpl.TRANSFORMERS
        )

        # Hybrid memory pool
        self.is_hybrid_swa = self.tp_worker.is_hybrid_swa
        self.is_hybrid_ssm = (
            self.tp_worker.model_runner.hybrid_gdn_config is not None
            or self.tp_worker.model_runner.mamba2_config is not None
        )

        self.sliding_window_size = None
        if self.is_hybrid_swa:
            self.sliding_window_size = self.tp_worker.sliding_window_size
            self.full_tokens_per_layer, self.swa_tokens_per_layer = (
                self.tp_worker.get_tokens_per_layer_info()
            )

        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            self.tp_worker.get_memory_pool()
        )

        self.disable_radix_cache = server_args.disable_radix_cache or (
            self.model_config.is_multimodal and uses_transformers_backend
        )
        if self.disable_radix_cache and not server_args.disable_radix_cache:
            logger.warning(
                "Radix cache is disabled for multimodal models with the "
                "Transformers backend to avoid multimodal prefix-cache mismatches."
            )

        effective_chunked_prefill_size = server_args.chunked_prefill_size
        if self.model_config.is_multimodal and uses_transformers_backend:
            effective_chunked_prefill_size = None

        params = CacheInitParams(
            disable=self.disable_radix_cache,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            page_size=self.page_size,
            is_eagle=self.spec_algorithm.is_eagle(),
            tp_cache_group=(
                self.attn_tp_cpu_group
                if self.server_args.enable_dp_attention
                else self.tp_cpu_group
            ),
            eviction_policy=server_args.radix_eviction_policy,
            enable_metrics=self.enable_metrics,
            enable_kv_cache_events=self.enable_kv_cache_events,
            enable_mamba_extra_buffer=server_args.enable_mamba_extra_buffer(),
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            chunked_prefill_size=effective_chunked_prefill_size,
            sliding_window_size=self.sliding_window_size,
        )

        if effective_chunked_prefill_size is not None and self.disable_radix_cache:
            if not self.is_hybrid_swa:
                from sglang.srt.mem_cache.chunk_cache import ChunkCache

                self.tree_cache = ChunkCache(params)
            else:
                from sglang.srt.mem_cache.chunk_cache import SWAChunkCache

                self.tree_cache = SWAChunkCache(params)
        else:

            if envs.SGLANG_EXPERIMENTAL_CPP_RADIX_TREE.get():
                # lazy import to avoid JIT overhead
                from sglang.srt.mem_cache.radix_cache_cpp import RadixCacheCpp

                logger.info("Using experimental C++ radix tree implementation.")
                self.tree_cache = RadixCacheCpp(params=params, server_args=server_args)
            elif self.enable_hierarchical_cache:
                if self.is_hybrid_ssm:
                    from sglang.srt.mem_cache.hi_mamba_radix_cache import (
                        HiMambaRadixCache,
                    )

                    self.tree_cache = HiMambaRadixCache(
                        params=params, server_args=server_args
                    )
                else:
                    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

                    self.tree_cache = HiRadixCache(
                        params=params, server_args=server_args
                    )
                self.tp_worker.register_hicache_layer_transfer_counter(
                    self.tree_cache.cache_controller.layer_done_counter
                )
            elif self.is_hybrid_swa:
                from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache

                self.tree_cache = SWARadixCache(params=params)
            elif self.is_hybrid_ssm:
                from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache

                self.tree_cache = MambaRadixCache(params)
            elif server_args.enable_lmcache:
                from sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache import (
                    LMCRadixCache,
                )

                self.tree_cache = LMCRadixCache(
                    params=params,
                    model_config=self.model_config,
                    tp_size=self.tp_size,
                    rank=self.tp_rank,
                    tp_group=self.tp_group,
                )
            else:
                self.tree_cache = RadixCache(params)

        if server_args.enable_streaming_session:
            self.tree_cache = SessionAwareCache(self.tree_cache)

        if self.enable_hisparse:
            # Coordinator was created inside ModelRunner.initialize() before CUDA graph capture
            self.hisparse_coordinator = self.tp_worker.model_runner.hisparse_coordinator
            self.hisparse_coordinator.set_decode_producer_stream(self.forward_stream)

        custom_agentic_decode = bool(
            envs.SGLANG_AGENTIC_KV_LIFECYCLE.get()
            and envs.SGLANG_AGENTIC_KV_CUSTOM_STORAGE_ONLY.get()
            and envs.SGLANG_AGENTIC_KV_HOST_STAGING.get()
        )
        if server_args.disaggregation_mode == "decode" and (
            server_args.disaggregation_decode_enable_offload_kvcache
            or custom_agentic_decode
        ):
            self.decode_offload_manager = DecodeKVCacheOffloadManager(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                tp_group=params.tp_cache_group,
                tree_cache=self.tree_cache,
                server_args=self.server_args,
            )
        else:
            self.decode_offload_manager = None

        embedding_cache_size = envs.SGLANG_VLM_CACHE_SIZE_MB.get()
        init_mm_embedding_cache(embedding_cache_size * 1024 * 1024)

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
        self.agentic_p_workset_broker = AgenticPWorksetLeaseBroker(
            self.server_args.page_size
        )
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
        else:
            self.agentic_tp_direct_mailbox = None
            self.agentic_tp_host_mailbox = None
            self.agentic_tp_p2d_sender_mailbox = None
            self.agentic_tp_p2d_receiver_mailbox = None
            self.agentic_tp_p2d_admission_mailbox = None
            self.agentic_tp_p2d_cleanup_mailbox = None
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
        self._pending_flush: Optional[Tuple[FlushCacheReqInput, float]] = None
        self.num_retracted_reqs: int = 0
        self.num_paused_reqs: int = 0
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
        self.is_mixed_chunk = (
            self.chunked_prefill_size is not None
            and self.server_args.enable_mixed_chunk
        )

        # Init the dynamic chunking predictor for PP
        self.enable_dynamic_chunking = (
            self.server_args.enable_dynamic_chunking and self.pp_size > 1
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
            self.prefill_delayer = PrefillDelayer(
                dp_size=self.dp_size,
                attn_tp_size=self.attn_tp_size,
                cpu_group=self.tp_cpu_group,
                server_args=self.server_args,
                metrics_collector=(
                    self.metrics_collector if self.enable_metrics else None
                ),
                max_delay_passes=self.server_args.prefill_delayer_max_delay_passes,
                token_usage_low_watermark=self.server_args.prefill_delayer_token_usage_low_watermark,
                device=(
                    self.tp_group.device
                    if self.server_args.disable_overlap_schedule
                    else "cpu"
                ),
            )

        # NOTE: preemption is enabled by default for priority scheduling.
        self.enable_priority_preemption = (
            self.enable_priority_scheduling
            and not self.server_args.disable_priority_preemption
        )

        self.init_new_token_ratio = min(
            envs.SGLANG_INIT_NEW_TOKEN_RATIO.get()
            * self.server_args.schedule_conservativeness,
            1.0,
        )
        self.min_new_token_ratio = min(
            self.init_new_token_ratio * envs.SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR.get(),
            1.0,
        )
        self.new_token_ratio_decay = (
            self.init_new_token_ratio - self.min_new_token_ratio
        ) / envs.SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS.get()
        self.new_token_ratio = self.init_new_token_ratio

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
        self.offload_tags = set()

        # Init recv skipper and input blocker
        self.recv_skipper = SchedulerRecvSkipper.maybe_create(self.server_args)
        self.input_blocker = (
            SchedulerInputBlocker(noop=self.attn_tp_rank != 0)
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

        if self.draft_worker is None or self.spec_algorithm.is_ngram():
            draft_token_to_kv_pool = None
        elif self.spec_algorithm.supports_spec_v2() and self.enable_overlap:
            if self.server_args.enable_multi_layer_eagle:
                draft_runner = self.draft_worker.draft_worker.draft_runner_list[0]
            else:
                draft_runner = self.draft_worker.draft_worker.draft_runner
            draft_token_to_kv_pool = draft_runner.token_to_kv_pool
            model_config = draft_runner.model_config
        else:
            # todo: should we fix this when enabling mtp or it doesn't matter since we only enable mtp in decode node thus we don't transfer draft kvs between P and D?
            draft_token_to_kv_pool = self.draft_worker.model_runner.token_to_kv_pool
            model_config = self.draft_worker.model_config

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
                    model_config.hidden_size
                    if self.spec_algorithm.is_eagle()
                    else 16  # minimal padding size for RDMA
                ),
                hidden_states_dtype=(
                    model_config.dtype
                    if self.spec_algorithm.is_eagle()
                    else torch.float32
                ),
                custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
            )

            # The decode requests polling kv cache
            self.disagg_decode_transfer_queue = DecodeTransferQueue(
                gloo_group=self.attn_tp_cpu_group,
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                tp_rank=self.tp_rank,
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
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                dp_size=self.server_args.dp_size,
                gpu_id=self.gpu_id,
                bootstrap_port=self.server_args.disaggregation_bootstrap_port,
                max_total_num_tokens=self.max_total_num_tokens,
                pp_rank=self.pp_rank,
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
                    model_config.hidden_size
                    if self.spec_algorithm.is_eagle()
                    or self.spec_algorithm.is_standalone()
                    else 16  # minimal padding size for RDMA
                ),
                hidden_states_dtype=(
                    model_config.dtype
                    if self.spec_algorithm.is_eagle()
                    or self.spec_algorithm.is_standalone()
                    else torch.float32
                ),
                custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
            )

            self.disagg_prefill_bootstrap_queue = PrefillBootstrapQueue(
                token_to_kv_pool=self.token_to_kv_pool_allocator.get_kvcache(),
                draft_token_to_kv_pool=draft_token_to_kv_pool,
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                metadata_buffers=self.disagg_metadata_buffers,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                gpu_id=self.gpu_id,
                bootstrap_port=self.server_args.disaggregation_bootstrap_port,
                gloo_group=self.attn_tp_cpu_group,
                max_total_num_tokens=self.max_total_num_tokens,
                scheduler=self,
                pp_rank=self.pp_rank,
                pp_size=self.pp_size,
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
                if self.agentic_early_claim_store is not None:
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
                    if envs.SGLANG_AGENTIC_KV_CUSTOM_STORAGE_ONLY.get():
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
                    else:
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
                if os.getenv("SGLANG_AGENTIC_KV_P2D_HOST_STAGING", "0").lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }:
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
            and self.server_args.encoder_transfer_backend == "zmq_to_scheduler"
        ):
            self.mm_receiver = create_mm_receiver(
                self.server_args,
                hf_config=self.model_config.hf_config,
                pp_rank=self.pp_rank,
                tp_rank=self.tp_rank,
                tp_group=self.tp_group,
                scheduler=self,
            )

    def init_overlap(self):
        self.device_module = torch.get_device_module(self.device)

        self.forward_stream_ctx: CudaStreamContext = self.device_module.stream(
            self.forward_stream
        )
        self.copy_stream: CudaStream = self.device_module.Stream()
        self.copy_stream_ctx: CudaStreamContext = self.device_module.stream(
            self.copy_stream
        )

        if not self.enable_overlap:
            self.future_map = None
            return

        self.future_map = FutureMap(
            self.max_running_requests,
            self.chunked_prefill_size,
            self.model_config.context_len,
            self.device,
            self.spec_algorithm,
        )
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
                (FlushCacheReqInput, self.flush_cache_wrapped),
                (ClearHiCacheReqInput, self.clear_hicache_storage_wrapped),
                (AttachHiCacheStorageReqInput, self.attach_hicache_storage_wrapped),
                (DetachHiCacheStorageReqInput, self.detach_hicache_storage_wrapped),
                (AbortReq, self.abort_request),
                (OpenSessionReqInput, self.open_session),
                (CloseSessionReqInput, self.close_session),
                (UpdateWeightFromDiskReqInput, self.update_weights_from_disk),
                (InitWeightsUpdateGroupReqInput, self.init_weights_update_group),
                (DestroyWeightsUpdateGroupReqInput, self.destroy_weights_update_group),
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
                    self.update_weights_from_distributed,
                ),
                (UpdateWeightsFromTensorReqInput, self.update_weights_from_tensor),
                (UpdateWeightsFromIPCReqInput, self.update_weights_from_ipc),
                (GetWeightsByNameReqInput, self.get_weights_by_name),
                (ReleaseMemoryOccupationReqInput, self.release_memory_occupation),
                (ResumeMemoryOccupationReqInput, self.resume_memory_occupation),
                (CheckWeightsReqInput, self.check_weights),
                (SlowDownReqInput, self.slow_down),
                (ProfileReq, self.profile),
                (FreezeGCReq, self.handle_freeze_gc),
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
                (GetLoadReqInput, self.get_load),
                (GetLoadsReqInput, self.get_loads),
                (PauseGenerationReqInput, self.pause_generation),
                (ContinueGenerationReqInput, self.continue_generation),
                (DumperControlReqInput, self.handle_dumper_control),
            ]
        )

    def _abort_on_running_timeout(self):
        # NOTE: this should be called before a batch is launched,
        # as current spec-v1 still filters batch inside verify stage.
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

    def run_event_loop(self) -> None:
        """Run the scheduler's event loop.

        Sets up the schedule stream and dispatches to the appropriate event loop.
        The event loop blocks until shutdown.
        """
        self.schedule_stream = self.device_module.Stream(priority=0)
        if self.device == "cpu":
            self.schedule_stream.synchronize = lambda: None  # No-op for CPU
        with self.device_module.StreamContext(self.schedule_stream):
            dispatch_event_loop(self)

    @DynamicGradMode()
    def event_loop_normal(self):
        """A normal scheduler loop."""
        while True:
            # Receive requests
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                self.cancel_bubble_timer()
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
                self.self_check_during_idle()

            # Update last_batch
            self.last_batch = batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.self_check_during_busy()

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
            # Receive requests
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue

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
                self.cancel_bubble_timer()

            # Process the last batch
            if self.last_batch:
                if not disable_overlap_for_batch:
                    pop_and_process()
            elif batch is None:
                # When the server is idle, do self-check and re-init some states
                self.self_check_during_idle()

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            if self.is_generation:
                self.launch_batch_sample_if_needed(batch_result)

            # Update last_batch
            self.last_batch = batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.self_check_during_busy()

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
            and batch.is_spec_v2
            and batch.has_grammar
            and batch.forward_mode.is_decode()
            and len(self.result_queue) > 0
        )

        return disable_overlap_for_batch or need_grammar_sync

    def recv_limit_reached(self, num_recv_reqs: int) -> bool:
        if self.max_recv_per_poll < 0:
            return False
        return num_recv_reqs >= self.max_recv_per_poll

    def recv_requests(
        self,
    ) -> List[Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput, Any]]:
        """Receive results at tp_rank = 0 and broadcast it to all other TP ranks."""

        if self.recv_skipper is not None:
            last_forward_mode = (
                self.last_batch.forward_mode if self.last_batch is not None else None
            )
            if not self.recv_skipper.handle(last_forward_mode):
                return []

        if self.pp_rank == 0:
            if self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
                recv_reqs = []

                while True:
                    try:
                        if self.recv_limit_reached(len(recv_reqs)):
                            break
                        recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    recv_reqs.append(recv_req)

                while True:
                    try:
                        if self.recv_limit_reached(len(recv_reqs)):
                            break
                        recv_rpc = self.recv_from_rpc.recv_pyobj(zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    recv_reqs.append(recv_rpc)
            else:
                recv_reqs = None
        else:
            if self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
                dp_offset = self.attn_dp_rank * self.attn_tp_size
                recv_reqs = point_to_point_pyobj(
                    [],
                    self.pp_rank * self.tp_size + dp_offset,
                    self.world_group.cpu_group,
                    (self.pp_rank - 1) * self.tp_size + dp_offset,
                    self.pp_rank * self.tp_size + dp_offset,
                )
            else:
                recv_reqs = None

        if self.input_blocker is not None:
            recv_reqs = self.input_blocker.handle(recv_reqs)

        if self.tp_size > 1 and envs.SGLANG_AGENTIC_KV_LIFECYCLE.get():
            self._agentic_tp_reduce_direct_status()
            self._agentic_tp_reduce_host_status()

        # Agentic TP admission follows the same control plane as ordinary
        # request ingress.  Do not introduce a second TP collective in the
        # scheduler loop: it can be reached in a different order from model
        # collectives when a rank-local Direct/Host DMA completes.  TP0
        # appends one tiny control record to the native recv broadcast and all
        # ranks consume it below after that broadcast has completed.
        if (
            self.tp_size > 1
            and self.pp_rank == 0
            and self.attn_tp_rank == 0
            and self.attn_cp_rank == 0
        ):
            if recv_reqs is None:
                recv_reqs = []
            control = self._agentic_tp_prepare_admission_control()
            if control is not None:
                recv_reqs.append(control)

        if self.server_args.enable_dp_attention:
            if self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
                work_reqs, control_reqs = self._split_work_and_control_reqs(recv_reqs)
            else:
                work_reqs = None
                control_reqs = None

            if self.attn_tp_size != 1:
                work_reqs = broadcast_pyobj(
                    work_reqs,
                    self.attn_tp_group.rank,
                    self.attn_tp_cpu_group,
                    src=self.attn_tp_group.ranks[0],
                )

            if self.attn_cp_size != 1:
                work_reqs = broadcast_pyobj(
                    work_reqs,
                    self.attn_cp_group.rank,
                    self.attn_cp_cpu_group,
                    src=self.attn_cp_group.ranks[0],
                )

            if self.tp_size != 1:
                control_reqs = broadcast_pyobj(
                    control_reqs,
                    self.tp_group.rank,
                    self.tp_cpu_group,
                    src=self.tp_group.ranks[0],
                )
            recv_reqs = work_reqs + control_reqs
        elif self.tp_size != 1:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.tp_group.rank,
                self.tp_cpu_group,
                src=self.tp_group.ranks[0],
            )

        recv_reqs = self._agentic_tp_consume_admission_control(recv_reqs)

        # Process MM requests under EPD-disaggregation mode
        if (
            self.pp_rank == 0
            and self.server_args.language_only
            and self.server_args.encoder_transfer_backend == "zmq_to_scheduler"
        ):
            recv_reqs, abort_reqs = self.mm_receiver.process_waiting_requests(recv_reqs)
            for req, error_msg, error_code in abort_reqs:

                status_code = (
                    HTTPStatus.BAD_REQUEST
                    if error_code == 400
                    else HTTPStatus.INTERNAL_SERVER_ERROR
                )
                prepare_abort(req, error_msg, status_code=status_code)
                self.stream_output([req], req.return_logprob)

        # Unwrap shared memory features AFTER all broadcasts complete,
        # so that ShmPointerMMData metadata (not full tensor data) is what
        # gets serialized during broadcast_pyobj.
        if recv_reqs:
            # Barrier for the non-DP-attention path only: there is a single
            # broadcast_pyobj on tp_cpu_group where the source rank returns
            # the original objects immediately while other ranks are still in
            # pickle.loads (-> __setstate__ -> shm_open).  Without a barrier
            # the source can call materialize() / shm_unlink before others
            # open the segment.  recv_reqs is consistent across all ranks
            # here (same broadcast), so the guard is deadlock-free.
            #
            # Under DP-attention no barrier is needed: the control_reqs
            # broadcast on tp_cpu_group (step 3) is a collective that forces
            # every rank to complete the earlier attn_tp / attn_cp work_reqs
            # deserializations (steps 1-2, which call shm_open) before any
            # rank returns from step 3.  POSIX guarantees shm_unlink only
            # removes the name; already-open handles stay valid.
            if (
                not self.server_args.enable_dp_attention
                and self.tp_size > 1
                and self.model_config.is_multimodal
                and has_shm_features(recv_reqs)
            ):
                barrier(group=self.tp_cpu_group)
            for req in recv_reqs:
                unwrap_shm_features(req)

        return recv_reqs

    def _split_work_and_control_reqs(self, recv_reqs: List):
        work_reqs = [
            req
            for req in recv_reqs
            if isinstance(
                req,
                (
                    TokenizedGenerateReqInput,
                    TokenizedEmbeddingReqInput,
                    BatchTokenizedGenerateReqInput,
                    BatchTokenizedEmbeddingReqInput,
                ),
            )
        ]
        control_reqs = [
            req
            for req in recv_reqs
            if not isinstance(
                req,
                (
                    TokenizedGenerateReqInput,
                    TokenizedEmbeddingReqInput,
                    BatchTokenizedGenerateReqInput,
                    BatchTokenizedEmbeddingReqInput,
                ),
            )
        ]
        return work_reqs, control_reqs

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
                    self.send_to_tokenizer.send_output(output, recv_req)
                else:
                    if self.recv_from_rpc is not None:
                        self.recv_from_rpc.send_pyobj(output)

        self._check_pending_flush()
        host_staging = getattr(self, "agentic_host_staging_manager", None)
        if host_staging is not None:
            host_staging.poll()
        self._drain_agentic_kv_waiting_queue()

    def init_req_max_new_tokens(self, req):
        req.sampling_params.max_new_tokens = min(
            (
                req.sampling_params.max_new_tokens
                if req.sampling_params.max_new_tokens is not None
                else 1 << 30
            ),
            self.max_req_len - len(req.origin_input_ids) - 1,
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
                fake_input_ids = [1] * seq_length
                recv_req.input_ids = fake_input_ids

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
                token_type_ids=recv_req.token_type_ids,
                custom_logit_processor=recv_req.custom_logit_processor,
                require_reasoning=recv_req.require_reasoning,
                return_hidden_states=recv_req.return_hidden_states,
                return_routed_experts=recv_req.return_routed_experts,
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
                    self.metrics_collector if self.enable_metrics else None
                ),
                extra_key=recv_req.extra_key,
                routing_key=recv_req.routing_key,
                http_worker_ipc=recv_req.http_worker_ipc,
                dllm_config=self.dllm_config,
                time_stats=recv_req.time_stats,
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
                    self.stream_output([req], req.return_logprob)
                    return

        elif session_id in self.session_controller:
            # Session exists: create request from session
            session = self.session_controller.get(session_id)
            req = session.create_req(
                recv_req,
                self.tokenizer,
                self.model_config.vocab_size,
                eos_token_ids=self.model_config.hf_eos_token_id,
            )
            # TODO: set trace context
            if self.enable_metrics:
                req.time_stats.set_metrics_collector(self.metrics_collector)
            if isinstance(req.finished_reason, FINISH_ABORT):
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return

        else:
            # Session ID provided but session not found
            req = Req(
                recv_req.rid,
                recv_req.input_text,
                recv_req.input_ids,
                recv_req.sampling_params,
                vocab_size=self.model_config.vocab_size,
            )
            req.tokenizer = self.tokenizer
            req.set_finish_with_abort(
                f"Invalid request: session id {session_id} does not exist"
            )
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
            if self.pad_input_ids_func:
                req.origin_input_ids = self.pad_input_ids_func(
                    req.origin_input_ids, image_inputs
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
                new_input_tokens = req.fill_ids[matched_len:]

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

    def _agentic_service_p_workset_leases(self) -> None:
        """Service background restore intents at an allocator-safe boundary."""

        broker = getattr(self, "agentic_p_workset_broker", None)
        if broker is not None:
            reserve_tokens = 0
            chunked_req = getattr(self, "chunked_req", None)
            private_suffix = (
                None
                if chunked_req is None
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
            return False
        if manifest.kv_layout_hash and manifest.kv_layout_hash != runtime.layout_hash:
            logger.error(
                "AgenticKV Direct layout mismatch snapshot=%s source=%s destination=%s",
                manifest.snapshot_id,
                manifest.kv_layout_hash,
                runtime.layout_hash,
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
                    and current.state is SnapshotState.DIRECT_LOADING
                    and current.claim_id == entry.claim_id
                ):
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
                if failed is not None:
                    failed.add(entry.request.snapshot_id)
        logger.warning(
            "AgenticKV early_direct_drop snapshot=%s reason=%s",
            entry.request.snapshot_id,
            reason,
        )

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
                    "lease_states=%s",
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
                    continue
            else:
                # Preserve the established 1P behavior: its arrival markers
                # are untargeted and require no route-resolution handshake.
                target_domain = None

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
                # Reject it before touching the shared workset broker so it
                # cannot contend with or cancel the Slow restore owner.
                continue
            broker = self.agentic_p_workset_broker
            workset_owner = AgenticPWorksetLeaseBroker.direct_owner(request.snapshot_id)
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
                if now - entry.completed_at >= bind_timeout:
                    if self.tp_size > 1:
                        self._agentic_mark_tp_direct_failed(
                            entry, reason="request_bind_timeout"
                        )
                    else:
                        self._agentic_drop_early_direct_receive(
                            entry,
                            snapshot_store,
                            release_claim=False,
                            reason="request_bind_timeout",
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
                    elif current.state is SnapshotState.CONSUMED:
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
                    if completed.state is not SnapshotState.CONSUMED:
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
                self.agentic_tp_direct_local_failed.add(snapshot_id)
                self.agentic_p_workset_broker.request_release(
                    snapshot_id,
                    workset_lease,
                    io_attempt=(
                        None if entry is None else getattr(entry, "io_attempt", None)
                    ),
                )
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
    ) -> bool:
        """Publish one TP-wide abort without involving the model scheduler.

        Return ``False`` only when the authoritative lifecycle already says
        the complete TP group was consumed; that state must never be rolled
        back by a delayed timeout observation.
        """

        snapshot_id = request.snapshot_id
        mailbox = self.agentic_tp_direct_mailbox
        poll_lock = getattr(self, "agentic_early_direct_poll_lock", nullcontext())
        with poll_lock:
            active_item = self.agentic_tp_direct_admission_active.get(snapshot_id)
        if active_item is None:
            return False
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
            mailbox.publish_receipt(snapshot_id, 3)
            return False

        expected_claim_id = (
            "direct-early-tp:"
            f"{os.getenv('SGLANG_AGENTIC_KV_ENGINE_ID', 'prefill')}:"
            f"{snapshot_id}"
        )
        if (
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
        self.agentic_tp_direct_local_failed.add(snapshot_id)
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
        mailbox.publish_local_progress(snapshot_id, -1)
        mailbox.publish_receipt(snapshot_id, -1)
        logger.warning(
            "AgenticKV tp_direct_background_abort snapshot=%s reason=%s "
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
                if snapshot_id in getattr(self, "agentic_tp_direct_local_failed", ()):
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
            if group_status < 0:
                receipt = mailbox.receipt(snapshot_id)
                if receipt is not None and int(receipt) < 0:
                    # The authoritative claim was already released and the
                    # native scheduler only needs to consume this abort once.
                    # Avoid repeating ledger I/O and warning logs every 5 ms
                    # while a long Prefill forward delays that control tick.
                    continue
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
                with poll_lock:
                    if snapshot_id in self.agentic_tp_direct_admission_active:
                        mailbox.publish_receipt(snapshot_id, 4)
                continue
            if group_status < 3 or entry is None or entry.group_committed:
                continue
            completed = snapshot_store.complete_direct_group(
                entry.manifest, entry.claim_id
            )
            if completed.state is not SnapshotState.CONSUMED:
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
        if tp_size > 1:
            direct_actions = getattr(self, "_agentic_tp_direct_actions", {})
            action = direct_actions.get(request.snapshot_id)
            if action == "commit_bind":
                if entry.prepared_req is not req:
                    return True
                entry.prepared_req = None
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
                return True

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
            self._agentic_drop_early_direct_receive(
                entry,
                self._agentic_snapshot_store(),
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
            self._agentic_drop_early_direct_receive(
                entry,
                self._agentic_snapshot_store(),
                release_claim=False,
                reason="workset_marker_underestimated",
            )
            req._agentic_kv_gate_complete = True
            req._agentic_kv_fallback = "workset_marker_underestimated"
            return False

        if not self.agentic_p_workset_broker.begin_bind(
            request.snapshot_id, entry.workset_lease
        ):
            self._agentic_drop_early_direct_receive(
                entry,
                self._agentic_snapshot_store(),
                release_claim=False,
                reason="workset_ownership_lost",
            )
            req._agentic_kv_gate_complete = True
            req._agentic_kv_fallback = "early_direct_workset_ownership_lost"
            return False

        # Record ownership before the first Radix mutation so every exception
        # path can remove the exact request-generation branch.
        if tp_size > 1:
            entry.prepared_req = req
        req._agentic_direct_parent_token_count = len(parent_tokens)
        try:
            result = self.tree_cache.insert(
                InsertParams(
                    key=RadixKey(parent_tokens, req.extra_key),
                    value=entry.device_indices,
                    priority=getattr(req, "priority", 0) or 0,
                )
            )
        except Exception:
            self.agentic_p_workset_broker.abort_bind(
                request.snapshot_id,
                entry.workset_lease,
                parent_bound=False,
            )
            logger.exception("Failed to bind early Direct KV for %s", req.rid)
            if tp_size > 1:
                return self._agentic_fail_tp_direct_bind(
                    entry,
                    req,
                    reason="radix_insert_failed",
                )
            self._agentic_drop_early_direct_receive(
                entry,
                self._agentic_snapshot_store(),
                release_claim=False,
                reason="radix_insert_failed",
            )
            req._agentic_kv_gate_complete = True
            req._agentic_kv_fallback = "early_direct_radix_insert_failed"
            return False
        self.agentic_p_workset_broker.commit_parent_bound(
            request.snapshot_id, entry.workset_lease
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
            return self._agentic_finalize_early_direct_bind(
                req,
                request,
                entry,
                existing_tokens=int(result.prefix_len),
                tp_size=tp_size,
                marker_store=marker_store,
                admit=tp_size == 1,
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
            self._agentic_drop_early_direct_receive(
                entry,
                self._agentic_snapshot_store(),
                release_claim=False,
                reason="radix_prepare_failed",
            )
            req._agentic_kv_gate_complete = True
            req._agentic_kv_fallback = "early_direct_radix_prepare_failed"
            return False

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
            return False
        parent_tokens = req.origin_input_ids[: manifest.token_count]
        if (
            len(parent_tokens) != manifest.token_count
            or token_ids_digest(parent_tokens) != manifest.token_digest
        ):
            req._agentic_kv_fallback = "direct_token_digest_mismatch"
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

    def _agentic_poll_direct_load(self, req: Req) -> bool:
        receiver = getattr(req, "_agentic_direct_receiver", None)
        if receiver is None:
            return False
        if getattr(req, "_agentic_direct_rank_received", False):
            snapshot_store = req._agentic_kv_snapshot_store
            manifest = req._agentic_direct_manifest
            current = snapshot_store.load(manifest.request, require_ready=False)
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
                receiver.clear()
                self._agentic_clear_direct_receiver(receiver, manifest)
                req._agentic_kv_gate_complete = True
                req._agentic_kv_fallback = "direct_workset_ownership_lost"
                return False
            inserted = False
            req._agentic_direct_parent_token_count = len(keys)
            try:
                result = self.tree_cache.insert(
                    InsertParams(
                        key=RadixKey(keys, req.extra_key),
                        value=device_indices,
                        priority=getattr(req, "priority", 0) or 0,
                    )
                )
                inserted = True
                self.agentic_p_workset_broker.commit_parent_bound(
                    manifest.snapshot_id, workset_lease
                )
            except Exception:
                self.agentic_p_workset_broker.abort_bind(
                    manifest.snapshot_id,
                    workset_lease,
                    parent_bound=inserted,
                )
                current = snapshot_store.load(manifest.request, require_ready=False)
                if (
                    current is not None
                    and current.state is SnapshotState.DIRECT_LOADING
                ):
                    try:
                        snapshot_store.mark_failed(
                            current, reason="direct_radix_insert_failed"
                        )
                    except Exception:
                        logger.exception(
                            "Failed to close direct snapshot after Radix error"
                        )
                self._agentic_clear_direct_receiver(receiver, manifest)
                logger.exception("Failed to insert direct KV for %s", req.rid)
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
                req._agentic_kv_gate_complete = True
                req._agentic_kv_fallback = "direct_radix_insert_failed"
                return False
            if result.prefix_len:
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
                receiver.clear()
                self._agentic_clear_direct_receiver(receiver, manifest)
                req._agentic_kv_gate_complete = True
                req._agentic_kv_fallback = "direct_radix_prepare_failed"
                logger.exception("Failed to protect direct KV for %s", req.rid)
                return False
            logger.info(
                "AgenticKV direct_radix_verify snapshot=%s device_tokens=%d "
                "host_tokens=%d",
                manifest.snapshot_id,
                len(immediate_match.device_indices),
                immediate_match.host_hit_length,
            )
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
                    try:
                        latest = snapshot_store.load(
                            manifest.request, require_ready=False
                        )
                        if (
                            latest is not None
                            and latest.state is SnapshotState.DIRECT_LOADING
                        ):
                            snapshot_store.mark_failed(
                                latest,
                                reason="direct_completion_marker_failed",
                            )
                    except Exception:
                        logger.exception(
                            "Failed to close direct manifest for %s", req.rid
                        )
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
            if completed.state is not SnapshotState.CONSUMED:
                req._agentic_direct_rank_received = True
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
        if (
            getattr(self, "tp_size", 1) > 1
            and getattr(self, "_agentic_tp_host_timeout_snapshot", None)
            == metadata.parent.snapshot_id
        ):
            # TP0 selected this exact stale Host generation for recompute and
            # published the decision through the native request broadcast.
            # Apply it before consulting rank-local Host discovery state so
            # both ranks take the same branch in this scheduler iteration.
            req._agentic_kv_gate_complete = True
            req._agentic_kv_fallback = "timeout:shared_host"
            req._agentic_kv_queue_class = "slow"
            logger.warning(
                "TP group timed out waiting for shared-Host parent snapshot "
                "of %s; falling back to recompute",
                req.rid,
            )
            return False
        host_staging = getattr(self, "agentic_host_staging_manager", None)
        if host_staging is not None:
            if getattr(self, "tp_size", 1) > 1:
                host_action = getattr(self, "_agentic_tp_host_actions", {}).get(
                    metadata.parent.snapshot_id
                )
                if host_action == "abort":
                    host_staging.abort_request(req.rid, metadata.parent)
                    host_staging.rollback_bound_parent(req, metadata.parent)
                    if hasattr(req, "_agentic_tp_host_failed"):
                        delattr(req, "_agentic_tp_host_failed")
                    req._agentic_kv_gate_complete = True
                    req._agentic_kv_fallback = "tp_shared_host_group_failed"
                    self.agentic_tp_host_local_admitted.add(metadata.parent.snapshot_id)
                    return False
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
                # A shared-Host ledger entry can remain HOST_WRITING,
                # SPILLING, etc. after its producer fails to publish the final
                # ready state.  gate_request() intentionally reports such an
                # entry as owned, but it must not bypass the request-level
                # snapshot-ready timeout forever.  Only time out metadata-only
                # waiters; an H2D copy already in flight has its own completion
                # path and normally lasts only milliseconds.
                timeout = max(0.0, envs.SGLANG_AGENTIC_KV_READY_TIMEOUT.get())
                if (
                    host_gate
                    and not self._agentic_io_active(req)
                    and not host_staging.snapshot_ready(metadata.parent)
                    and time.monotonic() - started_at >= timeout
                ):
                    if getattr(self, "tp_size", 1) > 1:
                        # A rank-local timeout must never let one TP rank
                        # recompute while its peer still waits for Host KV.
                        # TP0 publishes the exact fallback generation through
                        # the native request broadcast below.
                        return True
                    logger.warning(
                        "Timed out waiting %.1fs for shared-Host parent snapshot "
                        "of %s; falling back to recompute",
                        timeout,
                        req.rid,
                    )
                    req._agentic_kv_gate_complete = True
                    req._agentic_kv_fallback = "timeout:shared_host"
                    return False
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
            SnapshotState.P_LOADING,
        }:
            # These are producer/receiver progress states, not evidence that
            # the parent KV is unavailable.  Under c640 the Direct manager can
            # remain in one of them for several seconds; admitting the child
            # here silently turns transport congestion into a full recompute.
            if time.monotonic() - started_at < timeout:
                return True
            logger.warning(
                "Timed out waiting %.1fs for in-progress parent snapshot %s "
                "of req %s (state=%s); falling back to recompute",
                timeout,
                metadata.parent.snapshot_id,
                req.rid,
                manifest.state.value,
            )
            req._agentic_kv_gate_complete = True
            req._agentic_kv_fallback = f"timeout:{manifest.state.value}"
            return False

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
        if time.monotonic() - started_at >= timeout:
            state = "missing" if manifest is None else manifest.state.value
            logger.warning(
                "Timed out waiting %.1fs for parent snapshot %s of req %s "
                "(state=%s); "
                "falling back to recompute",
                timeout,
                metadata.parent.snapshot_id,
                req.rid,
                state,
            )
            req._agentic_kv_gate_complete = True
            req._agentic_kv_fallback = f"timeout:{state}"
            return False
        return True

    @staticmethod
    def _agentic_queue_class(req: Req) -> str:
        value = getattr(req, "_agentic_kv_queue_class", "fast")
        return value if value in {"fast", "slow", "new"} else "fast"

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
                self, "agentic_tp_direct_local_failed", ()
            ):
                local_status = -1
            if (
                request.snapshot_id
                in getattr(self, "agentic_tp_direct_local_admitted", ())
                and local_status >= 0
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
                        p_ready_dir
                        and decode_req.req.bootstrap_host != FAKE_BOOTSTRAP_HOST
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
            if receipt is not None and int(receipt) < 0:
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
                    host_staging._complete_shared_host_manifest(host_request)
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
            # Timeout is also a TP admission decision.  Rank 0 chooses one
            # exact stale Host waiter and piggybacks it on SGLang's existing
            # request broadcast; peers never make this decision from their
            # slightly different local queue timestamps.
            timeout = max(0.0, envs.SGLANG_AGENTIC_KV_READY_TIMEOUT.get())
            now = time.monotonic()
            for req, started_at in self.agentic_kv_waiting_queue:
                if now - started_at < timeout or self._agentic_io_active(req):
                    continue
                metadata = AgenticRequestMetadata.from_req(req)
                parent = metadata.parent if metadata is not None else None
                if parent is None or host_staging.snapshot_ready(parent):
                    continue
                entry = host_staging.ledger.get(parent.snapshot_id)
                if entry is not None and entry.get("state") in {
                    "host_reserved",
                    "host_writing",
                    "aborting",
                    "spilling",
                }:
                    host_timeout_snapshot = parent.snapshot_id
                    break
        return {
            self._AGENTIC_TP_CONTROL_KEY: True,
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
            self._agentic_tp_decode_admit_keys = [
                (str(rid), int(room))
                for rid, room in control.get("decode_admit_keys", ())
            ]
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
                    elif active_item is not None and active_item[4] is not None:
                        # No receiver owns the lease yet.  If a concurrent
                        # start already reserved it, request_release refuses
                        # the tokenless release and that start observes the
                        # terminal marker above before registering.
                        self.agentic_p_workset_broker.request_release(
                            snapshot_id, active_item[4]
                        )
                with direct_poll_lock:
                    self.agentic_tp_direct_local_admitted.discard(snapshot_id)
                    self.agentic_tp_direct_local_failed.discard(snapshot_id)
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

    def _drain_agentic_kv_waiting_queue(self) -> None:
        """Progress active KV I/O and admit pending work in arrival order.

        A request in this queue owns metadata only.  Each scheduler iteration
        first polls already-started transfers. New Direct, Slow, and initial
        requests then share one arrival-ordered compute-admission queue; their
        I/O engines and credits remain independent. A small admission batch
        amortizes scheduler ticks that contain long Prefill kernels.
        """
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
            slow_io_cap = max(
                1, int(os.environ.get("SGLANG_AGENTIC_KV_SELECTED_IO_CAP", "1"))
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
        owned = [
            req
            for req in combined
            if (
                getattr(req, "_agentic_workset_suffix_indices", None) is not None
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
                if (
                    current is not None
                    and current.state is SnapshotState.DIRECT_LOADING
                ):
                    try:
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
                    # Every P request first enters the same scheduler-owned
                    # queue. Parent turns start as fast and may later be
                    # reclassified as slow after snapshot lookup. Initial
                    # requests are last, with deterministic aging to prevent
                    # starvation. This queue owns only metadata.
                    req._agentic_kv_queue_class = (
                        "fast" if metadata.parent is not None else "new"
                    )
                    req._agentic_kv_wait_enqueued = True
                    enqueued_at = time.monotonic()
                    req._agentic_kv_wait_started_at = enqueued_at
                    self.agentic_kv_waiting_queue.append((req, enqueued_at))
                    return
                # Direct is already resident; bypass the metadata-only
                # lifecycle queue and enter native Prefill admission.
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
            self.send_to_tokenizer.send_output(abort_req, req)
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

        self.send_to_tokenizer.send_output(
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
                self.send_to_tokenizer.send_output(
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
            token_type_ids=recv_req.token_type_ids,
            routed_dp_rank=recv_req.routed_dp_rank,
            priority=recv_req.priority,
            dimensions=recv_req.dimensions,
            lora_id=recv_req.lora_id,
            http_worker_ipc=recv_req.http_worker_ipc,
            time_stats=recv_req.time_stats,
        )
        req.tokenizer = self.tokenizer

        # Handle multimodal inputs
        if recv_req.image_inputs is not None:
            image_inputs = self._get_multimodal_inputs(recv_req.image_inputs)
            # Expand a single image token into multiple dummy tokens for receiving image embeddings
            # The `pad_input_ids_func` is model-specific and may be None for
            # embedding models or models not requiring special padding.
            # If None, `req.origin_input_ids` is expected to be correctly populated already.
            if self.pad_input_ids_func:
                req.origin_input_ids = self.pad_input_ids_func(
                    req.origin_input_ids, image_inputs
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
        self.tree_cache.cache_unfinished_req(req, chunked=True)

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

        batch.req_pool_indices = torch.tensor(
            [r.req_pool_idx for r in reqs], dtype=torch.int64, device=device
        )
        seq_lens = [len(r.origin_input_ids) + len(r.output_ids) - 1 for r in reqs]
        batch.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=device)
        batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        batch.seq_lens_sum = sum(seq_lens)
        # output_ids = last generated token, used as input_ids by prepare_for_decode
        batch.output_ids = torch.tensor(
            [r.output_ids[-1] for r in reqs], dtype=torch.int64, device=device
        )

        # Set logprob fields if any request needs them
        if batch.return_logprob:
            batch.top_logprobs_nums = [r.top_logprobs_num for r in reqs]
            batch.token_ids_logprobs = [list(r.origin_input_ids) for r in reqs]

        # Build sampling info from scratch for these requests
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch, self.model_config.vocab_size
        )
        # todo hisparse, maybe other info to contain for the new batch
        return batch

    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
        # Physical workset allocation/free is intentionally confined to this
        # scheduler-safe point.  Direct and Slow I/O workers remain fully
        # asynchronous and consume only immutable grants.
        self._agentic_service_p_workset_leases()
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
        # for load reporting (num_running_reqs via /get_load).
        # Runs outside the last_batch block so stale requests are cleaned
        # even when no new batches arrive (e.g. traffic stops).
        if self.running_batch.is_prefill_only:
            self.running_batch.filter_batch()

        if self.dllm_config is not None:
            new_batch = self.get_new_batch_dllm()
        else:
            new_batch = self.get_new_batch_prefill()

        need_mlp_sync = self.require_mlp_sync
        if need_mlp_sync and not self.spec_algorithm.is_none():
            # NOTE: This branch makes sure prefill and decode batches will not be mixed when spec and dp-attn is enabled.
            # Before merging the new batch into running batch:
            # 1. All new batches are none -> need_mlp_sync remains true (sync is needed for decode batch).
            # 2. All new batches are some (prefill / idle) -> we do not need prepare mlp sync one more time.
            new_batch = self.maybe_prepare_mlp_sync_batch(new_batch)
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
        ret = self.maybe_prepare_mlp_sync_batch(ret, need_sync=need_mlp_sync)

        # Handle ngram embedding
        ret = self._maybe_prepare_ngram_embedding(ret)

        if ret:
            set_schedule_time_batch(ret)

        return ret

    def get_num_allocatable_reqs(self, running_bs):
        res = get_global_server_args().pp_max_micro_batch_size - running_bs
        if self.pp_size > 1:
            res = min(res, self.req_to_token_pool.available_size())
        return res

    def get_new_batch_prefill(self) -> Optional[ScheduleBatch]:
        prefill_delayer_single_pass = None
        if self.prefill_delayer:
            # Get token usage from several pools
            token_usage = None
            if self.is_hybrid_swa:
                _, _, full_token_usage, swa_token_usage, *_ = self._get_swa_token_info()
                token_usage = max(full_token_usage, swa_token_usage)
            if self.is_hybrid_ssm:
                _, _, full_token_usage, mamba_token_usage, *_ = (
                    self._get_mamba_token_info()
                )
                token_usage = (
                    max(token_usage, mamba_token_usage)
                    if token_usage is not None
                    else max(full_token_usage, mamba_token_usage)
                )
            if token_usage is None:
                _, token_usage, _, _ = self._get_token_info()

            assert token_usage is not None
            prefill_delayer_single_pass = PrefillDelayerSinglePassExecutor(
                self.prefill_delayer, token_usage=token_usage
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

        if self.enable_priority_preemption:
            # Reset batch_is_full to try preemption with a prefill adder.
            self.running_batch.batch_is_full = False

        if (
            self.running_batch.batch_is_full or len(self.waiting_queue) == 0
        ) and self.chunked_req is None:
            return None

        running_bs = len(self.running_batch.reqs)

        # Ignore the check if self.chunked_req is not None.
        # In the non-PP case, when self.chunked_req is not None, num_allocatable_reqs should always be greater than 0,
        # as the space for the chunked requests has just been released.
        # In PP case, chunked requests (or dllm requests) can start in one microbatch and end in another microbatch, so the max_running_requests per microbatch should not be strict.
        # Instead, we should always allow chunked requests to be added, otherwise, there will be a memory leak.
        if (
            self.get_num_allocatable_reqs(running_bs) <= 0
            and self.chunked_req is not None
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
            self.new_token_ratio,
            self.max_prefill_tokens,
            chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
            self.priority_scheduling_preemption_threshold,
            max_prefill_bs=self.max_prefill_bs,
            max_running_requests=self.max_running_requests,
            prefill_max_requests=self.server_args.prefill_max_requests,
            prefill_delayer_single_pass=prefill_delayer_single_pass,
            dllm_config=self.dllm_config,
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
                workset_suffix_indices is not None
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
            running_loras = {req.lora_id for req in self.running_batch.reqs}

        # Get requests from the waiting queue to a new prefill batch
        for req in self.waiting_queue:
            if self.enable_lora and req.lora_id not in running_loras:
                if self.enable_lora_overlap_loading:
                    # For overlapping loading of LoRA weights with computation, we will load each adapter one at a time,
                    # as opposed to loading them in one batch
                    res = self.lora_overlap_loader.try_overlap_load_lora(
                        req.lora_id, running_loras
                    )
                    if not res:
                        continue
                else:
                    new_lora_set = {req.lora_id} | running_loras
                    if not self.tp_worker.model_runner.lora_manager.validate_lora_batch(
                        new_lora_set
                    ):
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
                -(-len(req.fill_ids) // self.page_size) * self.page_size
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
            res = adder.add_one_req(
                req,
                has_chunked_req=(self.chunked_req is not None),
                truncation_align_size=self.truncation_align_size,
            )
            if p_ready_credit is not None and len(adder.can_run_list) > can_run_before:
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
                # revert matched mamba idx to avoid memory leak, if req is not added
                if not added and req.mamba_pool_idx is not None:
                    self.tree_cache.req_to_token_pool.mamba_pool.free(
                        req.mamba_pool_idx.unsqueeze(-1)
                    )
                    req.mamba_pool_idx = None
                break

        # Update waiting queue
        can_run_list: List[Req] = adder.can_run_list
        if len(can_run_list) == 0:
            return None

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
            self.chunked_req.is_chunked += 1

        # Record for logging prefill stats after forward
        self.adder = adder
        self.can_run_list = can_run_list
        self.running_bs = len(self.running_batch.reqs)

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
        self.max_prefill_bs = max(self.max_prefill_bs, len(can_run_list))
        if self.enable_hierarchical_cache:
            # todo (zhiqiang): disable cuda graph execution if hicache loading triggered
            new_batch.hicache_consumer_index = (
                self.tree_cache.ready_to_load_host_cache()
            )

        new_batch.prepare_for_extend()

        # Record prefill stats for logging after forward
        new_batch.prefill_stats = PrefillStats.from_adder(
            adder, self.running_batch.reqs, self.enable_priority_scheduling
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
            self.running_batch.filter_batch(v1_spec_info_filtered=True)
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

    def update_running_batch(self, batch: ScheduleBatch) -> Optional[ScheduleBatch]:
        """Update the current running decoding batch."""
        initial_bs = batch.batch_size()

        batch.filter_batch(v1_spec_info_filtered=True)
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
            old_ratio = self.new_token_ratio
            retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode(
                self.server_args
            )
            new_available_tokens = self.token_to_kv_pool_allocator.available_size()
            new_token_gained = new_available_tokens - old_available_tokens

            self.num_retracted_reqs = len(retracted_reqs)
            if self.enable_metrics and len(retracted_reqs) > 0:
                self.metrics_collector.increment_retracted_reqs(
                    num_retracted_reqs=len(retracted_reqs),
                    num_retracted_input_tokens=sum(
                        len(r.origin_input_ids) for r in retracted_reqs
                    ),
                    num_retracted_output_tokens=sum(
                        len(r.output_ids) for r in retracted_reqs
                    ),
                )
            self.new_token_ratio = new_token_ratio
            for req in reqs_to_abort:
                abort_reason: FINISH_ABORT = req.to_finish
                self.send_to_tokenizer.send_output(
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
            if kv_full_retract_flag:
                msg_details += (
                    f", #new_token_ratio: {old_ratio:.4f} -> {new_token_ratio:.4f}"
                )
            logger.warning(msg_prefix + msg_details)

            for req in retracted_reqs:
                self._add_request_to_queue(req, is_retracted=True)
                if self.enable_hisparse:
                    self.hisparse_coordinator.retract_req(req)
        else:
            self.new_token_ratio = max(
                self.new_token_ratio - self.new_token_ratio_decay,
                self.min_new_token_ratio,
            )

        if batch.batch_size() < initial_bs:
            batch.batch_is_full = False

        if batch.is_empty():
            return batch

        # Update batch tensors
        batch.prepare_for_decode()
        return batch

    def record_batch_in_overlap(self, model_worker_batch: ModelWorkerBatch):
        # FIXME(lsyin): hacky way to keep a reference to avoid GPU tensors being freed by torch GC
        # NOTE: More Reliable: record all tensors into the forward stream
        # NOTE: - for all future tensors, we shall always read from future map
        #       - for all non-future tensors (produced only by schedule stream),
        #       we shall keep its reference not being release during all the forwarding pass
        self.batch_record_ct = (self.batch_record_ct + 1) % 2
        self.batch_record_buf[self.batch_record_ct] = model_worker_batch

    def run_batch(
        self,
        batch: ScheduleBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[GenerationBatchResult, EmbeddingBatchResult]:
        """Run a batch."""
        self.forward_ct += 1

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
        self._profile_batch_predicate(batch)
        if self.forward_sleep_time is not None:
            logger.info(f"Scheduler.run_batch sleep {self.forward_sleep_time}s")
            time.sleep(self.forward_sleep_time)

        # Capture prefill start time for EXTEND mode
        if batch.forward_mode == ForwardMode.EXTEND:
            set_time_batch(batch.reqs, "set_prefill_run_batch_start_time")

        # Place holder handling for pd-disagg decode event loop
        if batch.forward_mode.is_prebuilt():
            return self._run_batch_prebuilt(batch)

        # Run forward
        if self.is_generation:
            if self.spec_algorithm.is_none() or self.enable_overlap:
                # In most cases, we use the model worker batch to run the forward.
                worker_batch_or_batch = batch.get_model_worker_batch()
            else:
                # In speculative decoding v1 (non-overlap) case, we use the batch directly.
                # TODO(lsyin): delete this branch after unifying the abstraction.
                worker_batch_or_batch = batch

            if self.enable_overlap:
                model_worker_batch = worker_batch_or_batch
                self.record_batch_in_overlap(model_worker_batch)

                # Sampling info will be modified during forward, so we store a copy.
                model_worker_batch.sampling_info = (
                    model_worker_batch.sampling_info.copy_for_forward()
                )

                bs = len(model_worker_batch.seq_lens)
                future_indices = self.future_map.alloc_future_indices(bs)

                with self.forward_stream_ctx, self.record_bubble_metrics(batch):
                    self.forward_stream.wait_stream(self.schedule_stream)
                    self.future_map.resolve_future(model_worker_batch)
                    with self.record_forward_metrics(batch):
                        batch_result = self.model_worker.forward_batch_generation(
                            model_worker_batch
                            # here pp is not compatible with overlap
                        )
                    # FIXME(lsyin): maybe move this to forward_batch_generation
                    batch_result.copy_done = self.device_module.Event()
                    if batch_result.delay_sample_func is None:
                        self.future_map.store_to_map(future_indices, batch_result)
                        batch_result.copy_to_cpu(return_logprob=batch.return_logprob)
                    else:
                        batch_result.future_indices = future_indices

                # FIXME(lsyin): move this assignment elsewhere
                future_indices_or_next_token_ids = -future_indices.indices

                if batch.is_spec_v2:
                    # FIXME(lsyin): tmp code for spec v2
                    # We only keep future indices for next draft input

                    batch.spec_info = batch_result.next_draft_input
                    batch.spec_info.future_indices = future_indices

                    # batch.spec_info = EagleDraftInput(
                    #     future_indices=future_indices,
                    #     verify_done=batch_result.next_draft_input.verify_done,
                    # )

                    # The future value, usually for next batch preparation
                    # Current implementation strictly synchronizes the seq_lens
                    batch.seq_lens = batch_result.next_draft_input.new_seq_lens
            elif self.enable_pdmux and batch.forward_mode.is_split_prefill():
                batch_result = self.tp_worker.forward_batch_split_prefill(batch)
                future_indices_or_next_token_ids = batch_result.next_token_ids
            else:
                kwargs = (
                    {"pp_proxy_tensors": pp_proxy_tensors}
                    if self.spec_algorithm.is_none()
                    else {}
                )
                with self.record_forward_metrics(batch):
                    batch_result = self.model_worker.forward_batch_generation(
                        worker_batch_or_batch, **kwargs
                    )
                if agentic_tp_debug:
                    logger.info(
                        "AgenticTP batch_exit ct=%d mode=%s rids=%s",
                        self.forward_ct,
                        batch.forward_mode,
                        [req.rid for req in batch.reqs],
                    )
                future_indices_or_next_token_ids = batch_result.next_token_ids
                self.update_cache_from_scheduler(batch, batch_result)

            # NOTE: future_indices_or_next_token_ids is used in ScheduleBatch,
            #       which can probably be replaced by future_indices later [TODO(lsyin)].
            #       we shall still keep the original outputs, e.g. next_token_ids
            #       in the GenerationBatchOutput for processing after copy_done.
            batch.output_ids = future_indices_or_next_token_ids

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
            model_worker_batch = batch.get_model_worker_batch()

            if self.enable_overlap:
                self.record_batch_in_overlap(model_worker_batch)
                with self.forward_stream_ctx, self.record_bubble_metrics(batch):
                    self.forward_stream.wait_stream(self.schedule_stream)
                    embeddings = self.tp_worker.forward_batch_embedding(
                        model_worker_batch
                    )
                    ret = EmbeddingBatchResult(embeddings=embeddings)
                    ret.copy_to_cpu()
            else:
                embeddings = self.tp_worker.forward_batch_embedding(model_worker_batch)
                ret = EmbeddingBatchResult(embeddings=embeddings)

        # Capture prefill end time for EXTEND mode
        if batch.forward_mode == ForwardMode.EXTEND:
            set_time_batch(batch.reqs, "set_prefill_run_batch_end_time")

        if (
            self.server_args.enable_dp_attention
            and self.server_args.elastic_ep_backend is not None
        ):
            # Get the tensors indicating rank activeness
            tp_active_ranks = self.tp_group.active_ranks.detach().cpu().numpy()
            tp_active_ranks_cpu = self.tp_group.active_ranks_cpu.detach().numpy()
            tp_active_ranks &= tp_active_ranks_cpu
            dp_active_ranks = tp_active_ranks.reshape(self.dp_size, -1).prod(axis=1)
            self.send_to_tokenizer.send_output(
                ActiveRanksOutput(status=dp_active_ranks.tolist())
            )

        return ret

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
            self.future_map.store_to_map(batch_result.future_indices, batch_result)
            batch_result.copy_to_cpu(return_logprob=self.cur_batch.return_logprob)

        # Release the closure and large GPU tensors that are no longer needed.
        # The delay_sample_func closure captures forward_batch (which holds
        # sampling_info with vocab_mask) and logits_output (which holds
        # next_token_logits). Without clearing these, they stay alive via
        # batch_result in result_queue and batch_record_buf until the next
        # iteration, causing a steady VRAM leak with structured output.
        batch_result.delay_sample_func = None
        if batch_result.logits_output is not None:
            batch_result.logits_output.next_token_logits = None

    def process_batch_result(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        if batch.forward_mode.is_decode():
            self.process_batch_result_decode(batch, result)
        elif batch.forward_mode.is_extend():
            if batch.is_dllm():
                self.process_batch_result_dllm(batch, result)
            elif self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.process_batch_result_disagg_prefill(batch, result)
            else:
                self.process_batch_result_prefill(batch, result)
        elif batch.forward_mode.is_prebuilt():
            self.process_batch_result_prebuilt(batch)
        elif batch.forward_mode.is_idle():
            self.process_batch_result_idle(batch, result)

        self.log_batch_result_stats(batch, result)
        self._maybe_clear_mm_inputs(batch)
        self.maybe_send_health_check_signal()

    def maybe_send_health_check_signal(self):
        if self.return_health_check_ipcs:
            # Return some signal for the health check.
            # This is used to prevent the health check signal being blocked by long context prefill.
            # However, one minor issue is that this code path does not check the status of detokenizer manager.
            self.send_to_tokenizer.send_output(
                HealthCheckOutput(
                    http_worker_ipc=self.return_health_check_ipcs.popleft()
                )
            )

    def _check_pending_flush(self):
        if self._pending_flush is None:
            return

        pending_req, deadline = self._pending_flush

        if self.is_fully_idle():
            success = self.flush_cache()
            self._pending_flush = None
            self.send_to_tokenizer.send_output(
                FlushCacheReqOutput(success=success), pending_req
            )
            return

        if time.monotonic() >= deadline:
            logging.warning(
                "Deferred flush_cache timed out while waiting for idle state."
            )
            self._pending_flush = None
            self.send_to_tokenizer.send_output(
                FlushCacheReqOutput(
                    success=False, message="Timed out waiting for idle state."
                ),
                pending_req,
            )

    def flush_cache_wrapped(
        self, recv_req: FlushCacheReqInput
    ) -> Optional[FlushCacheReqOutput]:
        if self._pending_flush is not None:
            return FlushCacheReqOutput(
                success=False,
                message="Another flush_cache is already in progress.",
            )

        timeout_s = float(recv_req.timeout_s or 0.0)
        if timeout_s <= 0.0:
            return FlushCacheReqOutput(success=self.flush_cache())

        if self.is_fully_idle():
            return FlushCacheReqOutput(success=self.flush_cache())

        self._pending_flush = (recv_req, time.monotonic() + timeout_s)
        return None

    def clear_hicache_storage_wrapped(self, recv_req: ClearHiCacheReqInput):
        if self.enable_hierarchical_cache:
            self.tree_cache.clear_storage_backend()
            logger.info("Hierarchical cache cleared successfully!")
            if_success = True
        else:
            logging.warning("Hierarchical cache is not enabled.")
            if_success = False
        return ClearHiCacheReqOutput(success=if_success)

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
            and (self.pp_size == 1 or all(x.is_empty() for x in self.running_mbs))
        )

        # Waiting queues: waiting + bootstrapping + preallocation + kv transfer (decode)
        idle &= len(self.waiting_queue) == 0
        idle &= len(self.agentic_kv_waiting_queue) == 0
        idle &= len(self.agentic_early_direct_receives) == 0

        if not for_health_check:
            # Grammar queue and prefill inflight queue may not produce batch
            # results instantly, but they still indicate the server is not idle.
            idle &= len(self.grammar_manager.grammar_queue) == 0
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                idle &= len(self.disagg_prefill_inflight_queue) == 0
                idle &= len(self.disagg_prefill_bootstrap_queue.queue) == 0

            if self.disaggregation_mode == DisaggregationMode.DECODE:
                idle &= len(self.disagg_decode_prealloc_queue.queue) == 0
                idle &= len(self.disagg_decode_transfer_queue.queue) == 0

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

    def flush_cache(self):
        """Flush the memory pool and cache."""
        if self.is_fully_idle():
            self.cur_batch = None
            self.last_batch = None
            self.tree_cache.reset()
            self.req_to_token_pool.clear()
            self.token_to_kv_pool_allocator.clear()
            self.grammar_manager.clear()
            self.reset_metrics()

            if self.draft_worker:
                self.draft_worker.clear_cache_pool()

            # TODO: allow optional empty cache
            torch.cuda.empty_cache()
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
        ret = vars(get_global_server_args())
        ret["last_gen_throughput"] = self.last_gen_throughput
        ret["memory_usage"] = {
            "weight": round(self.tp_worker.model_runner.weight_load_mem_usage, 2),
            "kvcache": round(
                self.token_to_kv_pool_allocator.get_kvcache().mem_usage, 2
            ),
            "token_capacity": int(self.max_total_num_tokens),
            "graph": round(self.tp_worker.model_runner.graph_mem_usage, 2),
        }
        ret["effective_max_running_requests_per_dp"] = self.max_running_requests

        if not self.spec_algorithm.is_none() and self.spec_total_num_forward_ct > 0:
            ret["avg_spec_accept_length"] = (
                self.spec_total_num_accepted_tokens / self.spec_total_num_forward_ct
            )

        if RECORD_STEP_TIME:
            ret["step_time_dict"] = self.step_time_dict

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
                v > self.max_running_requests // self.pp_size or v < 1
            ):
                logging.warning(
                    f"Updating {k} to {v} is rejected because it is out of the valid range [1, {self.max_running_requests // self.pp_size}]."
                )
                if_success = False
                break

        if if_success:
            if not self.spec_algorithm.is_none() and self.spec_total_num_forward_ct > 0:
                avg_spec_accept_length = (
                    self.spec_total_num_accepted_tokens / self.spec_total_num_forward_ct
                )
                logger.info(f"{avg_spec_accept_length=}")
            self.spec_total_num_accepted_tokens = self.spec_total_num_forward_ct = 0
            for k, v in server_args_dict.items():
                setattr(get_global_server_args(), k, v)
            logger.info(f"Global server args updated! {get_global_server_args()=}")
        return SetInternalStateReqOutput(
            updated=True,
            server_args=vars(get_global_server_args()),
        )

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
        # todo hisparse, release resources for abort requests in hisparse coordinator
        # Requests waiting for their parent generation have not entered any of
        # SGLang's ordinary queues yet, so abort them explicitly.
        remaining_agentic_waiters = []
        for req, started_at in self.agentic_kv_waiting_queue:
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                self._agentic_abort_cleanup(req)
                self.send_to_tokenizer.send_output(AbortReq(rid=req.rid), req)
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
            self.send_to_tokenizer.send_output(AbortReq(rid=req.rid), req)
            # For disaggregation decode mode, the request in the waiting queue has KV cache allocated.
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                if self.enable_hisparse:
                    self.hisparse_coordinator.request_finished(req)
                release_kv_cache(req, self.tree_cache)
            # For disaggregation prefill mode, free the metadata buffer index
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                release_req_to_metadata_buffer(
                    req, self.req_to_metadata_buffer_idx_allocator
                )

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
                        self.send_to_tokenizer.send_output(
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
            self.running_batch.filter_batch(v1_spec_info_filtered=True)
            if len(self.running_batch.reqs) != 0:
                retracted_reqs = self.running_batch.retract_all(self.server_args)
                for req in retracted_reqs:
                    self._add_request_to_queue(req)

            self.running_batch.batch_is_full = False
            self.chunked_req = None

    def continue_generation(self, recv_req: ContinueGenerationReqInput):
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
        return self.session_controller.open(recv_req)

    def close_session(self, recv_req: CloseSessionReqInput):
        self.session_controller.close(recv_req)

    def maybe_sleep_on_idle(self):
        if self.idle_sleeper is not None:
            self.idle_sleeper.maybe_sleep()

    def handle_freeze_gc(self, recv_req: FreezeGCReq):
        """Handle freeze_gc request: freeze scheduler's GC and forward to detokenizer."""
        freeze_gc("Scheduler")
        self.send_to_detokenizer.send_output(recv_req, recv_req)
        return None

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
            self.send_to_tokenizer.send_output(
                DumperControlReqOutput(success=True, response=response), recv_req
            )
        except Exception as e:
            print(f"[Scheduler] handle_dumper_control error: {e}", flush=True)
            self.send_to_tokenizer.send_output(
                DumperControlReqOutput(success=False, response=[], error=str(e)),
                recv_req,
            )

    # placeholder for override
    def update_cache_from_scheduler(
        self, schedule_batch: ScheduleBatch, batch_result: GenerationBatchResult
    ):
        pass


class IdleSleeper:
    """
    In setups which have long inactivity periods it is desirable to reduce
    system power consumption when sglang does nothing. This would lead not only
    to power savings, but also to more CPU thermal headroom when a request
    eventually comes. This is important in cases when multiple GPUs are connected
    as each GPU would otherwise pin one thread at 100% CPU usage.

    The simplest solution is to use zmq.Poller on all sockets that may receive
    data that needs handling immediately.
    """

    def __init__(self, sockets):
        self.poller = zmq.Poller()
        self.last_empty_time = real_time()
        for s in sockets:
            self.poller.register(s, zmq.POLLIN)

        self.empty_cache_interval = envs.SGLANG_EMPTY_CACHE_INTERVAL.get()

    def maybe_sleep(self):
        self.poller.poll(1000)
        if (
            self.empty_cache_interval > 0
            and real_time() - self.last_empty_time > self.empty_cache_interval
        ):
            self.last_empty_time = real_time()
            torch.cuda.empty_cache()


def is_health_check_generate_req(recv_req):
    rid = getattr(recv_req, "rid", None)
    return rid is not None and rid.startswith(HEALTH_CHECK_RID_PREFIX)


def is_work_request(recv_req):
    return isinstance(
        recv_req,
        (
            TokenizedGenerateReqInput,
            TokenizedEmbeddingReqInput,
            BatchTokenizedGenerateReqInput,
            BatchTokenizedEmbeddingReqInput,
        ),
    )


class SenderWrapper:
    def __init__(self, socket: zmq.Socket):
        self.socket = socket

    def send_output(
        self,
        output: Union[BaseReq, BaseBatchReq],
        recv_obj: Optional[Union[BaseReq, BaseBatchReq]] = None,
    ):
        if self.socket is None:
            return

        if (
            isinstance(recv_obj, BaseReq)
            and recv_obj.http_worker_ipc is not None
            and output.http_worker_ipc is None
        ):
            # handle communicator reqs for multi-http worker case
            output.http_worker_ipc = recv_obj.http_worker_ipc

        self.socket.send_pyobj(output)


def dispatch_event_loop(scheduler: Scheduler):
    # Dispatch to the appropriate event loop based on the disaggregation mode
    server_args = scheduler.server_args
    disaggregation_mode: DisaggregationMode = scheduler.disaggregation_mode
    if disaggregation_mode == DisaggregationMode.NULL:
        if scheduler.enable_pdmux:
            scheduler.event_loop_pdmux()
        elif server_args.pp_size > 1:
            scheduler.event_loop_pp()
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


def configure_scheduler(
    server_args: ServerArgs,
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
    dp_rank = configure_scheduler(
        server_args, tp_rank, attn_cp_rank, moe_dp_rank, moe_ep_rank, pp_rank, dp_rank
    )

    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()

    # Set cpu affinity to this gpu process
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(
            server_args.pp_size, server_args.tp_size, server_args.nnodes, gpu_id
        )
    numa_node = get_numa_node_if_available(server_args, gpu_id)
    if numa_node is not None and not envs.SGLANG_NUMA_BIND_V2.get():
        numa_bind_to_node(numa_node)

    # Set up tracing
    if server_args.enable_trace:
        process_tracing_init(server_args.otlp_traces_endpoint, "sglang")
        thread_label = "Scheduler"
        if server_args.disaggregation_mode == "prefill":
            thread_label = "Prefill Scheduler"
        elif server_args.disaggregation_mode == "decode":
            thread_label = "Decode Scheduler"
        trace_set_thread_info(thread_label, tp_rank, dp_rank)

    # Create a scheduler and run the event loop
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

        # Run the event loop (blocks until shutdown)
        scheduler.run_event_loop()

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
