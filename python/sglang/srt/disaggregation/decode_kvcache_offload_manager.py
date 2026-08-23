from __future__ import annotations

import json
import hashlib
import logging
import os
import queue
import threading
import time
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.disaggregation.agentic_direct_transfer import (
    AgenticDirectRuntime,
    create_agentic_direct_runtime,
    debug_kv_digest,
)
from sglang.srt.disaggregation.agentic_early_claim import AgenticEarlyClaimStore
from sglang.srt.disaggregation.agentic_host_staging import (
    AgenticDHostStagingClient,
    AgenticDRelayWorker,
    SharedHostStagingLedger,
)
from sglang.srt.disaggregation.agentic_tp import rank_env_int
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.kv_events import OffloadedState
from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    AgenticRequestMetadata,
    SharedSnapshotEvictionController,
    SnapshotManifest,
    SnapshotEvictionController,
    SnapshotState,
    page_namespace,
    token_ids_digest,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    TransferBackend,
    kv_to_page_indices,
)
from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig, get_hash_str
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
)
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class DecodeKVCacheOffloadManager:
    """Manage decode-side KV cache offloading lifecycle and operations."""

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tp_group: torch.distributed.ProcessGroup,
        tree_cache: BasePrefixCache,
        server_args: ServerArgs,
    ) -> None:
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = server_args.page_size
        self.server_args = server_args
        self.request_counter = 0
        self.tree_cache = tree_cache
        self.agentic_enabled = envs.SGLANG_AGENTIC_KV_LIFECYCLE.get()
        self.agentic_hostless = (
            self.agentic_enabled
            and envs.SGLANG_AGENTIC_KV_HOST_STAGING.get()
            and envs.SGLANG_AGENTIC_KV_D_HOSTLESS.get()
        )
        env_stride = envs.SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE.get()
        if env_stride is None or env_stride <= 0:
            self.offload_stride = self.page_size
        else:
            self.offload_stride = max(
                self.page_size, (env_stride // self.page_size) * self.page_size
            )
        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
        self.tp_rank = torch.distributed.get_rank(group=self.tp_group)

        hicache_storage_backend_extra_config = {}
        if server_args.hicache_storage_backend_extra_config:
            try:
                hicache_storage_backend_extra_config = json.loads(
                    server_args.hicache_storage_backend_extra_config
                )
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid hicache storage backend extra config JSON: {e}"
                )

        self.decode_host_mem_pool = None
        self.cache_controller = None
        self.agentic_metadata_backend = None
        if self.agentic_hostless:
            if server_args.hicache_storage_backend != "mooncake":
                raise ValueError("D-hostless agentic staging requires Mooncake metadata")
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeStore,
            )

            storage_config = HiCacheStorageConfig(
                tp_rank=torch.distributed.get_rank(group=self.tp_group),
                tp_size=self.tp_world_size,
                pp_rank=0,
                pp_size=1,
                is_mla_model=isinstance(kv_cache, MLATokenToKVPool),
                enable_storage_metrics=False,
                is_page_first_layout=True,
                model_name=server_args.served_model_name,
                extra_config=hicache_storage_backend_extra_config,
            )
            self.agentic_metadata_backend = MooncakeStore(
                storage_config=storage_config, mem_pool=None
            )
            logger.info(
                "Agentic D-hostless mode enabled: skipped decode Host KV pool; "
                "Mooncake client is metadata-only"
            )
        else:
            if isinstance(kv_cache, MHATokenToKVPool):
                self.decode_host_mem_pool = MHATokenToKVPoolHost(
                    kv_cache,
                    server_args.hicache_ratio,
                    server_args.hicache_size,
                    self.page_size,
                    server_args.hicache_mem_layout,
                )
            elif isinstance(kv_cache, MLATokenToKVPool):
                self.decode_host_mem_pool = MLATokenToKVPoolHost(
                    kv_cache,
                    server_args.hicache_ratio,
                    server_args.hicache_size,
                    self.page_size,
                    server_args.hicache_mem_layout,
                )
            else:
                raise ValueError("Unsupported KV cache type for decode offload")

            self.cache_controller = HiCacheController(
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                mem_pool_host=self.decode_host_mem_pool,
                page_size=self.page_size,
                tp_group=tp_group,
                io_backend=server_args.hicache_io_backend,
                load_cache_event=threading.Event(),
                storage_backend=server_args.hicache_storage_backend,
                model_name=server_args.served_model_name,
                storage_backend_extra_config=hicache_storage_backend_extra_config,
            )

        self.ongoing_offload = {}
        self.ongoing_backup = {}
        self.offloaded_state = {}
        # Finished requests whose final response must not be exposed until the
        # generated KV is durable in external storage.  Keeping this gate in
        # the manager lets the scheduler continue submitting decode batches
        # while Device->Host->storage progresses asynchronously.
        self.pending_responses = {}
        self.ready_responses = []
        self.response_backup_timeout = 120.0
        self.agentic_snapshot_store = None
        self.agentic_direct_runtime = None
        self.agentic_relay_runtime = None
        self.agentic_direct_candidates = {}
        # TP0's native scheduler broadcast can arrive a few microseconds before
        # a follower has published the matching rank-local candidate.  Keep the
        # latest command by request-generation instead of dropping that edge;
        # candidate publication below consumes it immediately.  This remains a
        # metadata-only handoff and adds no collective or transport polling.
        self._agentic_tp_pending_candidate_commands = {}
        # Transport, preallocation control, and agentic lifecycle progress have
        # independent workers. A slow ledger/Host operation must never stop
        # the P->D receiver that feeds Decode. Scheduler-owned allocator,
        # Radix, and running-batch mutations still arrive as small commit
        # events and remain on the scheduler thread.
        self._decode_transport_async_enabled = (
            self.agentic_enabled
            and os.getenv("SGLANG_DECODE_IO_ASYNC_PROGRESS", "1").lower()
            in {"1", "true", "yes", "on"}
        )
        # TP1 keeps the proven fully asynchronous lifecycle.  TP>1 moves only
        # P->D handshake/polling off the scheduler; D->P path selection and
        # allocator mutation remain rank0-broadcast scheduler commands.
        self._decode_io_async_enabled = (
            self._decode_transport_async_enabled and self.tp_world_size == 1
        )
        self._decode_io_events: queue.SimpleQueue = queue.SimpleQueue()
        self._decode_io_stop = threading.Event()
        self._decode_io_threads = {}
        self._decode_io_wakeups = {
            name: threading.Event()
            for name in ("transfer", "prealloc", "agentic", "relay")
        }
        self._decode_prealloc_queue = None
        self._decode_transfer_queue = None
        self._decode_io_cuda_device = torch.cuda.current_device()
        legacy_interval = os.getenv("SGLANG_DECODE_IO_PROGRESS_INTERVAL_SECONDS", "0.005")
        self._decode_io_intervals = {
            "transfer": max(
                0.0005,
                float(
                    os.getenv(
                        "SGLANG_DECODE_TRANSFER_PROGRESS_INTERVAL_SECONDS",
                        legacy_interval,
                    )
                ),
            ),
            "prealloc": max(
                0.001,
                float(
                    os.getenv(
                        "SGLANG_DECODE_PREALLOC_PROGRESS_INTERVAL_SECONDS", "0.005"
                    )
                ),
            ),
            "agentic": max(
                0.001,
                float(os.getenv("SGLANG_AGENTIC_KV_D_CONTROL_POLL_SECONDS", "0.02")),
            ),
            "relay": max(
                0.001,
                float(os.getenv("SGLANG_AGENTIC_KV_RELAY_POLL_SECONDS", "0.005")),
            ),
        }
        self._decode_io_error_count = 0
        self._decode_io_last_error = None
        self._decode_scheduler_commit_events = 0
        self._decode_scheduler_commit_seconds = 0.0
        # A background transport worker may finish after this scheduler's
        # commit drain but before the idle memory checker runs.  Those pages
        # still belong to the completed request until the scheduler applies
        # the queued release, so include them in allocator accounting.
        self._decode_pending_release_tokens = 0
        self._decode_commit_interval = max(
            0.0,
            float(os.getenv("SGLANG_DECODE_IO_COMMIT_INTERVAL_SECONDS", "0.02")),
        )
        self._decode_commit_ready_at = 0.0
        self._agentic_cached_d_kv_usage = 0.0
        self.agentic_host_staging_client = None
        self.agentic_relay_worker = None
        self._agentic_relay_pending = None
        self.agentic_fast_threshold = max(
            0.0, envs.SGLANG_AGENTIC_KV_FAST_TOOL_THRESHOLD.get()
        )
        self.agentic_early_claim_store = None
        self.agentic_early_claim_post_timeout = max(
            0.1,
            float(os.getenv("SGLANG_AGENTIC_KV_EARLY_CLAIM_POST_TIMEOUT", "1.0")),
        )
        self.agentic_early_claim_poll_interval = max(
            0.01,
            float(os.getenv("SGLANG_AGENTIC_KV_EARLY_CLAIM_POLL_INTERVAL", "0.05")),
        )
        if self.agentic_enabled and os.getenv(
            "SGLANG_AGENTIC_KV_EARLY_CLAIM", "0"
        ).lower() in {"1", "true", "yes", "on"}:
            early_claim_dir = os.getenv("SGLANG_AGENTIC_KV_EARLY_CLAIM_DIR", "")
            if not early_claim_dir:
                p_ready_dir = os.getenv("SGLANG_PD_P_READY_DIR", "")
                early_claim_dir = os.path.join(p_ready_dir, "early-claims")
            self.agentic_early_claim_store = AgenticEarlyClaimStore(early_claim_dir)
            logger.info(
                "AgenticKV fast_arrival enabled directory=%s "
                "tool_timeout=%.3f admission_timeout=%.3f",
                early_claim_dir,
                self.agentic_fast_threshold,
                self.agentic_early_claim_post_timeout,
            )
        if self.agentic_enabled:
            backend = (
                self.agentic_metadata_backend
                if self.agentic_hostless
                else self.cache_controller.storage_backend
            )
            snapshot_store_factory = getattr(backend, "agentic_snapshot_store", None)
            if snapshot_store_factory is None:
                raise ValueError(
                    "SGLANG_AGENTIC_KV_LIFECYCLE requires the Mooncake HiCache backend"
                )
            self.agentic_snapshot_store = snapshot_store_factory()
            try:
                tool_means = json.loads(
                    envs.SGLANG_AGENTIC_KV_TOOL_MEAN_SECONDS.get()
                )
                tool_means = {
                    str(name): float(seconds) for name, seconds in tool_means.items()
                }
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "SGLANG_AGENTIC_KV_TOOL_MEAN_SECONDS must be a JSON object"
                ) from exc
            ledger_path = envs.SGLANG_AGENTIC_KV_LEDGER_PATH.get()
            if self.agentic_hostless:
                # P owns Host capacity and cold Mooncake spill admission.  D
                # only publishes small lifecycle manifests in this mode.
                self.agentic_eviction_controller = None
            elif ledger_path:
                capacity_bytes = int(
                    envs.SGLANG_AGENTIC_KV_CAPACITY_GIB.get() * (1024**3)
                )
                self.agentic_eviction_controller = (
                    SharedSnapshotEvictionController(
                        self.agentic_snapshot_store,
                        ledger_path=ledger_path,
                        capacity_bytes=capacity_bytes,
                        high_watermark=envs.SGLANG_AGENTIC_KV_HIGH_WATERMARK.get(),
                        expected_tool_seconds=tool_means,
                        reservation_ttl_seconds=envs.SGLANG_AGENTIC_KV_STALE_SECONDS.get(),
                    )
                )
            else:
                capacity_bytes = int(
                    envs.SGLANG_AGENTIC_KV_CAPACITY_GIB.get() * (1024**3)
                )
                # Single-D compatibility mode. Multi-D V1 launchers always
                # provide a /dev/shm ledger path.
                self.agentic_eviction_controller = SnapshotEvictionController(
                    self.agentic_snapshot_store,
                    capacity_bytes=capacity_bytes,
                    high_watermark=envs.SGLANG_AGENTIC_KV_HIGH_WATERMARK.get(),
                    expected_tool_seconds=tool_means,
                )
            if self.agentic_fast_threshold > 0:
                direct_port = envs.SGLANG_AGENTIC_KV_DIRECT_BOOTSTRAP_PORT.get()
                if direct_port <= 0:
                    raise ValueError(
                        "Fast agentic D->P transfer requires "
                        "SGLANG_AGENTIC_KV_DIRECT_BOOTSTRAP_PORT on every D"
                    )
                self.agentic_direct_runtime = create_agentic_direct_runtime(
                    role=DisaggregationMode.PREFILL,
                    kv_pool=kv_cache,
                    server_args=server_args,
                    engine_rank=self.tp_rank,
                    pp_rank=0,
                    gpu_id=torch.cuda.current_device(),
                    total_kv_heads=(
                        getattr(kv_cache, "head_num", 1) * self.tp_world_size
                    ),
                    bootstrap_port=direct_port,
                )
                if envs.SGLANG_AGENTIC_KV_HOST_STAGING.get():
                    staging_ledger_path = (
                        envs.SGLANG_AGENTIC_KV_STAGING_LEDGER_PATH.get()
                        or f"{ledger_path}.staging"
                    )
                    if not ledger_path and not envs.SGLANG_AGENTIC_KV_STAGING_LEDGER_PATH.get():
                        raise ValueError(
                            "P Host staging requires SGLANG_AGENTIC_KV_LEDGER_PATH "
                            "or SGLANG_AGENTIC_KV_STAGING_LEDGER_PATH"
                        )
                    staging_ledger = SharedHostStagingLedger(staging_ledger_path)
                    relay_enabled = envs.SGLANG_AGENTIC_KV_RELAY_ENABLED.get()
                    source_numa = rank_env_int(
                        "SGLANG_AGENTIC_KV_GPU_NUMA_NODE",
                        "SGLANG_AGENTIC_KV_TP_NUMA_NODES",
                        tp_rank=self.tp_rank,
                    )
                    arena_numa = rank_env_int(
                        "SGLANG_AGENTIC_KV_ARENA_NUMA_NODE",
                        "SGLANG_AGENTIC_KV_TP_ARENA_NUMA_NODES",
                        tp_rank=self.tp_rank,
                    )
                    arena_domain = int(
                        os.environ.get("SGLANG_AGENTIC_KV_PREFILL_DOMAIN", "-1")
                    )
                    self.agentic_host_staging_client = AgenticDHostStagingClient(
                        staging_ledger,
                        kv_cache,
                        self.page_size,
                        direct_runtime=self.agentic_direct_runtime,
                        relay_enabled=relay_enabled,
                        source_numa_node=source_numa,
                        arena_numa_node=arena_numa,
                        arena_domain=arena_domain,
                        tp_rank=self.tp_rank,
                        tp_size=self.tp_world_size,
                        direct_cross_numa_gib_per_second=(
                            envs.SGLANG_AGENTIC_KV_DIRECT_CROSS_NUMA_GIBPS.get()
                        ),
                        nvlink_gib_per_second=(
                            envs.SGLANG_AGENTIC_KV_RELAY_NVLINK_GIBPS.get()
                        ),
                        relay_stale_seconds=(
                            envs.SGLANG_AGENTIC_KV_RELAY_STALE_SECONDS.get()
                        ),
                    )
                    if (
                        relay_enabled
                        and source_numa >= 0
                        and source_numa == arena_numa
                    ):
                        relay_id = envs.SGLANG_AGENTIC_KV_RELAY_ID.get()
                        if not relay_id:
                            relay_id = f"d-relay:{os.getpid()}"
                        # The stock P->D NIXL receiver manager is created later
                        # by DecodePreallocQueue.  Reuse it instead of
                        # registering this full KV pool a third time with UCX.
                        self._agentic_relay_pending = {
                            "ledger": staging_ledger,
                            "relay_id": relay_id,
                            "numa_node": source_numa,
                            "device_pool": kv_cache,
                            "slot_mib": envs.SGLANG_AGENTIC_KV_STAGING_SLOT_MIB.get(),
                            "slot_count": envs.SGLANG_AGENTIC_KV_STAGING_SLOTS.get(),
                            "d2h_gib_per_second": (
                                envs.SGLANG_AGENTIC_KV_RELAY_D2H_GIBPS.get()
                            ),
                        }
        logger.info("Enable offload kv cache for decode side")

    def attach_agentic_relay_manager(
        self, normal_decode_kv_manager, metadata_index_allocator
    ) -> None:
        """Attach after DecodePreallocQueue creates the stock D receiver."""

        pending = self._agentic_relay_pending
        if pending is None or self.agentic_relay_worker is not None:
            return
        isolate_relay = os.getenv(
            "SGLANG_AGENTIC_KV_ISOLATE_RELAY_PROGRESS", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if isolate_relay:
            # A relay touches the shared Host ledger before/after its NIXL
            # receive.  Reusing the normal P->D receiver manager therefore
            # couples a potentially blocking ledger/relay operation to the
            # latency-critical P->D notification drain.  Give the relay its
            # own NIXL agent and one-byte auxiliary mailbox so both progress
            # domains can run independently.  This is opt-in to preserve the
            # proven single-P path exactly.
            kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
            runtime = create_agentic_direct_runtime(
                role=DisaggregationMode.DECODE,
                kv_pool=kv_cache,
                server_args=self.server_args,
                engine_rank=torch.distributed.get_rank(group=self.tp_group),
                pp_rank=0,
                gpu_id=torch.cuda.current_device(),
                total_kv_heads=getattr(kv_cache, "head_num", 1),
            )
            self.agentic_relay_runtime = runtime
            relay_aux_index = 0
        else:
            relay_aux_index = metadata_index_allocator.alloc()
            if relay_aux_index is None:
                raise RuntimeError("not enough metadata rows for agentic NUMA relay")
            runtime = AgenticDirectRuntime(
                manager=normal_decode_kv_manager,
                aux_buffer=torch.empty(0),
                transfer_backend=TransferBackend(
                    self.server_args.disaggregation_transfer_backend
                ),
            )
        self.agentic_relay_worker = AgenticDRelayWorker(
            ledger=pending["ledger"],
            relay_id=pending["relay_id"],
            numa_node=pending["numa_node"],
            device_pool=pending["device_pool"],
            token_allocator=self.token_to_kv_pool_allocator,
            receiver_runtime=runtime,
            page_size=self.page_size,
            slot_mib=pending["slot_mib"],
            slot_count=pending["slot_count"],
            d2h_gib_per_second=pending["d2h_gib_per_second"],
            relay_aux_index=relay_aux_index,
        )
        self._agentic_relay_progress_isolated = isolate_relay
        self._agentic_relay_pending = None

    def start_decode_io_progress_worker(self, prealloc_queue, transfer_queue) -> None:
        """Move transport/control progress off the Decode scheduler thread.

        V1 is TP=1, so no cross-rank collective is required by the background
        poller.  Allocator and request-state commits deliberately remain on the
        scheduler thread and arrive through ``_decode_io_events``.
        """

        transport_async = getattr(
            self,
            "_decode_transport_async_enabled",
            getattr(self, "_decode_io_async_enabled", False),
        )
        if not transport_async or self._decode_io_threads:
            return
        self._decode_prealloc_queue = prealloc_queue
        self._decode_transfer_queue = transfer_queue
        prealloc_queue.enable_async_progress()
        transfer_queue.enable_async_progress()
        steps = {
            "transfer": self._decode_transfer_progress,
            "prealloc": prealloc_queue.background_progress,
        }
        if getattr(self, "tp_world_size", 1) == 1:
            steps["agentic"] = self._decode_agentic_progress
        if (
            self.agentic_relay_worker is not None
            and getattr(self, "_agentic_relay_progress_isolated", False)
        ):
            steps["relay"] = self.agentic_relay_worker.poll
        for name, step in steps.items():
            thread = threading.Thread(
                target=self._decode_progress_loop,
                args=(name, step),
                name=f"sglang-decode-{name}-{os.getpid()}",
                daemon=True,
            )
            self._decode_io_threads[name] = thread
            thread.start()
        logger.info(
            "Decode I/O async progress enabled transfer_ms=%.3f "
            "prealloc_ms=%.3f agentic_ms=%.3f relay_isolated=%s tp=%d",
            self._decode_io_intervals["transfer"] * 1000.0,
            self._decode_io_intervals["prealloc"] * 1000.0,
            self._decode_io_intervals["agentic"] * 1000.0,
            "relay" in steps,
            getattr(self, "tp_world_size", 1),
        )

    def _decode_transfer_progress(self) -> None:
        """Advance the stock P->D receiver without unrelated control work."""

        transfer_queue = self._decode_transfer_queue
        if transfer_queue is not None:
            transfer_queue.background_progress()
        relay_worker = self.agentic_relay_worker
        if relay_worker is not None and not getattr(
            self, "_agentic_relay_progress_isolated", False
        ):
            # Compatibility path for the proven 1P setup, where relay and
            # stock P->D intentionally share one NIXL manager.
            relay_worker.poll()

    def _decode_agentic_progress(self) -> None:
        self._check_agentic_direct_progress(progress_relay=False)

    def _decode_progress_pending(self, name: str) -> int:
        if name == "transfer":
            return len(getattr(self._decode_transfer_queue, "queue", ()))
        if name == "prealloc":
            return len(getattr(self._decode_prealloc_queue, "queue", ())) + int(
                getattr(self._decode_prealloc_queue, "_async_metadata_pending_count", 0)
            )
        if name == "relay":
            worker = getattr(self, "agentic_relay_worker", None)
            return int(worker is not None and worker.active is not None)
        return len(self.agentic_direct_candidates)

    def _decode_progress_loop(self, name: str, step) -> None:
        """Run one non-scheduler progress domain without cross-domain HOL."""

        if self._decode_io_cuda_device is not None:
            torch.cuda.set_device(self._decode_io_cuda_device)
        interval = self._decode_io_intervals[name]
        wakeup = self._decode_io_wakeups[name]
        cycles = 0
        total_seconds = 0.0
        max_seconds = 0.0
        errors = 0
        last_stats_at = time.monotonic()
        while not self._decode_io_stop.is_set():
            started_at = time.perf_counter()
            try:
                step()
            except Exception as exc:
                errors += 1
                self._decode_io_error_count += 1
                self._decode_io_last_error = repr(exc)
                logger.exception("Decode %s progress failed; retrying", name)
            elapsed = time.perf_counter() - started_at
            cycles += 1
            total_seconds += elapsed
            max_seconds = max(max_seconds, elapsed)
            now = time.monotonic()
            if now - last_stats_at >= 30.0:
                logger.info(
                    "Decode %s progress stats cycles=%d avg_us=%.3f "
                    "max_ms=%.3f pending=%d errors=%d",
                    name,
                    cycles,
                    total_seconds * 1e6 / max(1, cycles),
                    max_seconds * 1000.0,
                    self._decode_progress_pending(name),
                    errors,
                )
                cycles = 0
                total_seconds = 0.0
                max_seconds = 0.0
                errors = 0
                last_stats_at = now
            wakeup.wait(max(0.0, interval - elapsed))
            wakeup.clear()

    def _enqueue_agentic_release(self, req, start_offset: int) -> None:
        """Queue the only scheduler-owned mutation needed by D->P progress."""

        if not getattr(self, "_decode_io_async_enabled", False):
            if self.tp_world_size > 1:
                metadata = AgenticRequestMetadata.from_req(req)
                if metadata is None:
                    raise RuntimeError("TP agentic release lost request metadata")
                pending = getattr(self, "_agentic_tp_pending_releases", None)
                if pending is None:
                    pending = self._agentic_tp_pending_releases = {}
                pending.setdefault(
                    metadata.current.snapshot_id, (req, int(start_offset))
                )
                return
            self._release_finished_req(req, start_offset)
            return
        start_offset = int(start_offset)
        committed_len = int(getattr(req, "kv_committed_len", 0))
        allocated_len = int(getattr(req, "kv_allocated_len", committed_len))
        page_size = max(1, int(getattr(self, "page_size", 1)))
        if page_size > 1:
            allocated_len = ceil_align(allocated_len, page_size)
        reserved_tokens = max(0, allocated_len - start_offset)
        self._decode_pending_release_tokens = int(
            getattr(self, "_decode_pending_release_tokens", 0)
        ) + reserved_tokens
        was_empty = self._decode_io_events.empty()
        self._decode_io_events.put(
            ("release_finished", req, start_offset, reserved_tokens)
        )
        if was_empty:
            self._decode_commit_ready_at = (
                time.monotonic() + getattr(self, "_decode_commit_interval", 0.0)
            )

    def tp_pending_release_snapshot(self):
        """Return TP0's oldest release ready for the native recv broadcast."""

        pending = getattr(self, "_agentic_tp_pending_releases", None)
        if not pending:
            return None
        return next(iter(pending))

    def tp_candidate_commands(self) -> list[dict]:
        """Return TP0's authoritative D->P command for each live generation.

        These commands describe logical lifecycle decisions only.  Every TP
        rank owns a different KV-head shard, but followers must never infer a
        route from markers or manifests.  They merely execute the command
        selected here by rank 0 against their local shard.
        """

        if self.tp_world_size <= 1 or self.tp_rank != 0:
            return []
        commands = []
        for snapshot_id, candidate in self.agentic_direct_candidates.items():
            manifest = candidate.get("manifest")
            if candidate.get("staging"):
                action = "slow"
            elif candidate.get("sent") or (
                manifest is not None
                and manifest.state is SnapshotState.DIRECT_LOADING
            ):
                action = "direct"
            else:
                action = "wait"
            if candidate.get("tp_announced_action") == action:
                continue
            candidate["tp_announced_action"] = action
            command = {"snapshot_id": str(snapshot_id), "action": action}
            if action == "slow" and manifest is not None:
                command["manifest"] = manifest.to_bytes()
                command["prefill_domain"] = int(
                    candidate.get(
                        "selected_prefill_domain",
                        self.agentic_host_staging_client.arena_domain,
                    )
                )
                command["arena_numa_nodes"] = list(
                    candidate.get(
                        "selected_arena_numa_nodes",
                        self._prefill_domain_numa_nodes(
                            int(command["prefill_domain"])
                        ),
                    )
                )
            commands.append(command)
        return commands

    def _apply_tp_candidate_command(self, command) -> bool:
        """Apply one TP0 command, or retain it until the local shard exists."""

        snapshot_id = str(command["snapshot_id"])
        candidate = self.agentic_direct_candidates.get(snapshot_id)
        if candidate is None:
            pending = getattr(self, "_agentic_tp_pending_candidate_commands", None)
            if pending is None:
                pending = self._agentic_tp_pending_candidate_commands = {}
            pending[snapshot_id] = dict(command)
            return False
        candidate["tp_command"] = str(command["action"])
        if command.get("prefill_domain") is not None:
            domain = int(command["prefill_domain"])
            numa_nodes = [int(value) for value in command["arena_numa_nodes"]]
            candidate["selected_prefill_domain"] = domain
            candidate["selected_arena_numa_nodes"] = numa_nodes
        manifest_bytes = command.get("manifest")
        if manifest_bytes is not None:
            candidate["manifest"] = SnapshotManifest.from_bytes(manifest_bytes)
        pending = getattr(self, "_agentic_tp_pending_candidate_commands", None)
        if pending is not None:
            pending.pop(snapshot_id, None)
        return True

    def apply_tp_candidate_commands(self, commands) -> None:
        """Install rank-0 lifecycle commands on this rank's physical shards."""

        if self.tp_world_size <= 1:
            return
        for command in commands or ():
            DecodeKVCacheOffloadManager._apply_tp_candidate_command(self, command)

    def _prefill_domain_numa_nodes(self, domain: int) -> list[int]:
        raw = os.getenv("SGLANG_AGENTIC_KV_PREFILL_TP_NUMA_DOMAINS", "").strip()
        domains = [value for value in raw.split(";") if value]
        if 0 <= int(domain) < len(domains):
            nodes = [int(value) for value in domains[int(domain)].split(",")]
            if len(nodes) == self.tp_world_size:
                return nodes
        return [
            int(self.agentic_host_staging_client.arena_numa_node)
            for _ in range(self.tp_world_size)
        ]

    def _select_slow_prefill_domain(self) -> tuple[int, list[int]]:
        """Choose a logical P from the Router's cached pressure snapshot.

        This is a tiny /dev/shm read performed only when a request actually
        falls back.  It never queries P synchronously and therefore cannot
        block Decode or the fast path.
        """

        fallback = int(self.agentic_host_staging_client.arena_domain)
        if os.getenv(
            "SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS", ""
        ).strip().lower() not in {"1", "true", "yes", "on"}:
            return fallback, self._prefill_domain_numa_nodes(fallback)
        path = os.getenv("SGLANG_AGENTIC_KV_PREFILL_LOAD_PATH", "").strip()
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if time.time() - float(payload.get("published_at", 0.0)) > 2.0:
                raise ValueError("stale Prefill pressure snapshot")
            scored = []
            for item in payload.get("domains", ()):
                domain = int(item["domain"])
                hbm_capacity = max(1, int(item.get("hbm_capacity_tokens", 0)))
                arena_capacity = max(1, int(item.get("arena_capacity_bytes", 0)))
                pending_tokens = max(0, int(item.get("pending_tokens", 0)))
                hbm_used = max(0, int(item.get("hbm_used_tokens", 0)))
                arena_used = max(0, int(item.get("arena_used_bytes", 0)))
                pending_requests = max(0, int(item.get("pending_requests", 0)))
                scheduler_waiting = max(0, int(item.get("scheduler_waiting", 0)))
                score = (
                    pending_tokens / hbm_capacity
                    + hbm_used / hbm_capacity
                    + 2.0 * arena_used / arena_capacity
                    + 0.01 * (pending_requests + scheduler_waiting)
                )
                scored.append((score, domain))
            if not scored:
                raise ValueError("empty Prefill pressure snapshot")
            scored.sort()
            selected = int(scored[0][1])
            logger.info(
                "AgenticKV slow_prefill_select P=%d scores=%s",
                selected,
                [(domain, round(score, 4)) for score, domain in scored],
            )
            return selected, self._prefill_domain_numa_nodes(selected)
        except (OSError, TypeError, ValueError, KeyError):
            logger.warning(
                "AgenticKV slow_prefill_select_fallback P=%d path=%s",
                fallback,
                path,
            )
            return fallback, self._prefill_domain_numa_nodes(fallback)

    def _assign_slow_prefill_target(self, candidate) -> None:
        if "selected_prefill_domain" in candidate:
            return
        domain, numa_nodes = self._select_slow_prefill_domain()
        candidate["selected_prefill_domain"] = domain
        candidate["selected_arena_numa_nodes"] = numa_nodes

    def commit_tp_release(self, snapshot_id: str) -> None:
        """Apply a TP release selected by the scheduler's native broadcast.

        No collective is introduced here: ``Scheduler.recv_requests`` already
        delivered the same snapshot id to every TP rank before this method is
        called.  A peer that has not locally polled the group-visible terminal
        manifest yet can still resolve the request from its live candidate.
        """

        snapshot_id = str(snapshot_id)
        pending = getattr(self, "_agentic_tp_pending_releases", None)
        item = None if not pending else pending.pop(snapshot_id, None)
        candidate = self.agentic_direct_candidates.pop(snapshot_id, None)
        req = item[0] if item is not None else (
            None if candidate is None else candidate.get("req")
        )
        start_offset = int(item[1]) if item is not None else 0
        if req is None:
            raise RuntimeError(
                f"TP Decode release {snapshot_id} has no local request"
            )
        if req.req_pool_idx != -1:
            self._release_finished_req(req, start_offset)
        if candidate is not None:
            self._cleanup_agentic_direct_sender(candidate)
            self._agentic_release_early_claim(candidate, "tp_release_commit")
        logger.info(
            "AgenticKV tp_decode_release_commit snapshot=%s req=%s",
            snapshot_id,
            req.rid,
        )

    def _drain_decode_io_events(self) -> None:
        """Apply bounded, allocator-safe commits without doing transport work."""

        max_events = max(
            1, int(os.getenv("SGLANG_DECODE_IO_MAX_COMMITS_PER_STEP", "32"))
        )
        if self._decode_io_events.empty():
            return
        now = time.monotonic()
        if (
            now < getattr(self, "_decode_commit_ready_at", 0.0)
            and self._decode_io_events.qsize() < max_events
        ):
            return
        started_at = time.perf_counter()
        committed = 0
        allocator = getattr(self, "token_to_kv_pool_allocator", None)
        grouped_free = allocator is not None and hasattr(
            allocator, "free_group_begin"
        ) and not getattr(allocator, "debug_mode", False)
        if grouped_free:
            allocator.free_group_begin()
        try:
            while committed < max_events:
                try:
                    kind, req, value, reserved_tokens = (
                        self._decode_io_events.get_nowait()
                    )
                except queue.Empty:
                    break
                if kind != "release_finished":
                    logger.error("Unknown Decode I/O completion event: %s", kind)
                    continue
                # A duplicated/stale completion must never free a recycled request
                # slot.  The worker removes a candidate before publishing another
                # event, while this guard also makes shutdown/fail-soft idempotent.
                if req.req_pool_idx != -1:
                    self._release_finished_req(req, value)
                self._decode_pending_release_tokens = max(
                    0,
                    int(getattr(self, "_decode_pending_release_tokens", 0))
                    - int(reserved_tokens),
                )
                committed += 1
        finally:
            # Paged allocators otherwise run torch.unique/torch.cat once per
            # completed request.  Grouping preserves scheduler ownership and
            # request order while paying that CUDA allocator cost only once.
            if grouped_free:
                allocator.free_group_end()
        if self._decode_io_events.empty():
            self._decode_commit_ready_at = 0.0
        else:
            self._decode_commit_ready_at = now + getattr(
                self, "_decode_commit_interval", 0.0
            )
        if committed:
            self._decode_scheduler_commit_events += committed
            self._decode_scheduler_commit_seconds += time.perf_counter() - started_at

    def wake_decode_io_progress(self) -> None:
        if getattr(self, "_decode_io_async_enabled", False):
            for wakeup in self._decode_io_wakeups.values():
                wakeup.set()

    @property
    def agentic_reserved_token_count(self) -> int:
        """Permanent D-HBM tokens reserved by the NUMA relay, if any."""

        worker = self.agentic_relay_worker
        return int(worker.reserved_token_count) if worker is not None else 0

    @property
    def agentic_pending_release_token_count(self) -> int:
        """Completed-request KV waiting for a scheduler-owned free commit."""

        reserved = int(getattr(self, "_decode_pending_release_tokens", 0))
        # TP>1 uses the scheduler's existing native broadcast to commit a
        # release on every rank in lockstep.  Between local I/O completion and
        # that broadcast, the request's protected prefix is already accounted
        # by Radix, but its uncached/overallocated tail is owned by neither the
        # active batch nor Radix.  Include only that tail here so the idle
        # memory checker does not mistake the short hand-off window for a leak.
        pending = getattr(self, "_agentic_tp_pending_releases", None) or {}
        for req, _start_offset in pending.values():
            allocated_len = int(
                getattr(req, "kv_allocated_len", getattr(req, "kv_committed_len", 0))
            )
            if self.page_size > 1:
                allocated_len = ceil_align(allocated_len, self.page_size)
            protected_len = int(getattr(req, "cache_protected_len", 0))
            reserved += max(0, allocated_len - protected_len)
        return reserved

    @property
    def agentic_pending_release_req_count(self) -> int:
        """Request slots awaiting the same native TP release broadcast."""

        pending = getattr(self, "_agentic_tp_pending_releases", None) or {}
        return sum(1 for req, _ in pending.values() if req.req_pool_idx != -1)

    def offload_kv_cache(self, req) -> bool:
        """Offload incremental KV cache for decode side."""

        if (
            not self.agentic_hostless
            and (self.cache_controller is None or self.decode_host_mem_pool is None)
        ):
            return False

        metadata = AgenticRequestMetadata.from_req(req)
        custom_params = getattr(req.sampling_params, "custom_params", None) or {}
        if self.agentic_enabled and "agentic_request_id" in custom_params:
            logger.info(
                "AgenticKV request_seen req=%s snapshot=%s req_pool_idx=%d output_tokens=%d",
                req.rid,
                "missing" if metadata is None else metadata.current.snapshot_id,
                req.req_pool_idx,
                len(req.output_ids),
            )
        if req.req_pool_idx == -1 or len(req.output_ids) == 0:
            if self.agentic_enabled and metadata is not None:
                logger.warning(
                    "AgenticKV early_skip snapshot=%s req_pool_idx=%d output_tokens=%d",
                    metadata.current.snapshot_id,
                    req.req_pool_idx,
                    len(req.output_ids),
                )
            return False

        if self.agentic_enabled and metadata is not None:
            return self._offload_agentic_finished_snapshot(req, metadata)
        if self.agentic_hostless:
            # Hostless mode is deliberately scoped to request-generation
            # snapshots. Ordinary requests retain the normal finish/release path.
            return False
        if self.agentic_enabled and str(getattr(req, "extra_key", "")).startswith(
            "agentic-v1:"
        ):
            logger.warning(
                "AgenticKV metadata_missing req=%s extra_key=%s custom_keys=%s",
                req.rid,
                req.extra_key,
                sorted((getattr(req.sampling_params, "custom_params", None) or {}).keys()),
            )

        token_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx]
        if token_indices.dim() == 0 or token_indices.numel() == 0:
            return False

        # Prefill side offloads page-aligned origin_input_ids, decode side offloads the incremental part
        all_tokens = req.origin_input_ids + req.output_ids[:-1]
        prefill_offloaded_len = (
            len(req.origin_input_ids) // self.page_size * self.page_size
        )
        state = self.offloaded_state.get(req.rid)
        if state is None:
            prefill_hashes = self._compute_prefix_hash(
                req.origin_input_ids[:prefill_offloaded_len]
            )
            last_prefill_hash = (
                prefill_hashes[-1] if prefill_offloaded_len > 0 else None
            )
            state = OffloadedState(
                prefill_len=prefill_offloaded_len,
                inc_len=0,
                last_hash=last_prefill_hash,
            )
            self.offloaded_state[req.rid] = state
        incremental_total = len(all_tokens) - state.prefill_len
        incremental_new = incremental_total - state.inc_len
        incremental_aligned_len = (
            incremental_new // self.offload_stride * self.offload_stride
        )

        if incremental_aligned_len == 0:
            return False

        # Extract incremental tokens and indices for the newly available chunk
        start = state.prefill_len + state.inc_len
        end = start + incremental_aligned_len
        incremental_tokens = all_tokens[start:end]
        incremental_indices = token_indices[start:end]

        # Early free prefill-offloaded GPU memory
        if state.prefill_len > 0 and state.inc_len == 0:
            self.token_to_kv_pool_allocator.free(token_indices[: state.prefill_len])

        # Asynchronously offload incremental KV cache from device to host
        self.request_counter += 1
        ack_id = self.request_counter
        host_indices = self.cache_controller.write(
            device_indices=incremental_indices.long(),
            node_id=ack_id,
        )
        if host_indices is None:
            logger.error(f"Not enough host memory for request {req.rid}")
            return False

        self.ongoing_offload[ack_id] = (
            req,
            host_indices,
            incremental_tokens,
            time.time(),
            start,
            end,
        )
        if req.finished():
            self.pending_responses[req.rid] = (req, time.monotonic())
        state.inc_len += incremental_aligned_len
        return True

    def _offload_agentic_finished_snapshot(
        self, req: Req, metadata: AgenticRequestMetadata
    ) -> bool:
        """Start D2H for one complete tool-call snapshot.

        This path is intentionally finish-only: D keeps normal decode batching
        behavior while the request is running, and terminal answers bypass
        Mooncake entirely.
        """

        if not req.finished():
            return False
        output_kind = metadata.classify_output(req.output_ids, req.tokenizer)
        if output_kind.value == "unknown":
            output_kind = metadata.classify_output(req.output_ids[:-1], req.tokenizer)
        finish_reason = getattr(req, "finished_reason", None)
        finish_type = None
        if finish_reason is not None:
            try:
                finish_type = finish_reason.to_json().get("type")
            except (AttributeError, TypeError):
                finish_type = None

        # Length/abort completions and explicit terminal markers end the
        # trajectory.  UNKNOWN is not necessarily terminal: application
        # parsers can append a repair observation after malformed output.  In
        # that case keep the same short-lived Direct candidate as a tool call;
        # the application confirmation channel decides whether to continue or
        # release it, and unconfirmed candidates still fail soft without
        # entering Shared Arena/Mooncake.
        if finish_type in {"length", "abort"} or output_kind.value == "terminal":
            logger.info(
                "AgenticKV final_skip snapshot=%s output_tokens=%d "
                "finish_reason=%s output_kind=%s",
                metadata.current.snapshot_id,
                len(req.output_ids),
                finish_type or "unknown",
                output_kind.value,
            )
            return False

        all_tokens = req.origin_input_ids + req.output_ids[:-1]
        aligned_len = len(all_tokens) // self.page_size * self.page_size
        if aligned_len == 0:
            self._publish_agentic_failure(metadata, "empty_aligned_snapshot")
            return False
        all_tokens = all_tokens[:aligned_len]
        producer_store = getattr(self, "agentic_early_claim_store", None)
        producer_id = None
        if int(getattr(self, "tp_world_size", 1)) > 1:
            engine_id = os.getenv("SGLANG_AGENTIC_KV_ENGINE_ID", "decode")
            producer_id = f"{engine_id}:{req.rid}"
        owns_generation = True
        if producer_store is not None:
            if int(getattr(self, "tp_world_size", 1)) > 1 and self.tp_rank != 0:
                # Logical producer election belongs to rank 0.  Followers wait
                # for its atomically-published decision, then only pin and
                # transfer their local KV-head shard when the same D engine won.
                owns_generation = producer_store.wait_generation_producer(
                    metadata.current,
                    producer_id,
                )
            else:
                owns_generation = producer_store.claim_generation_producer(
                    metadata.current, producer_id=producer_id
                )
        if not owns_generation:
            # The original execution remains authoritative. This duplicate
            # still returns its deterministic model response to unblock the
            # retrying caller, but must not mutate lifecycle state owned by
            # another D worker.
            logger.warning(
                "AgenticKV duplicate_generation_skip snapshot=%s req=%s",
                metadata.current.snapshot_id,
                req.rid,
            )
            return False
        if self.agentic_direct_runtime is not None:
            try:
                if self._publish_agentic_direct_candidate(req, metadata, all_tokens):
                    return True
            except Exception:
                logger.exception(
                    "Failed to publish direct D->P candidate %s; falling back",
                    metadata.current.snapshot_id,
                )
        if self.agentic_hostless:
            # There is intentionally no D Host data path in this mode.  A
            # metadata/direct setup failure is fail-soft: release the finished
            # request normally and let the next P turn recompute its prefix.
            self._publish_agentic_failure(metadata, "direct_setup_failed")
            return False
        return self._start_agentic_slow_snapshot(req, metadata, all_tokens)

    def _publish_agentic_direct_candidate(
        self,
        req: Req,
        metadata: AgenticRequestMetadata,
        all_tokens,
    ) -> bool:
        room = int.from_bytes(
            hashlib.sha256(metadata.current.storage_id.encode()).digest()[:8],
            "little",
        ) & ((1 << 63) - 1)
        manifest = SnapshotManifest(
            request=metadata.current,
            page_keys=(),
            token_count=len(all_tokens),
            byte_size=0,
            state=SnapshotState.DIRECT_READY,
            tool_type=metadata.tool_type,
            tool_started_at=time.time(),
            token_digest=token_ids_digest(all_tokens),
            direct_bootstrap_addr=self.agentic_direct_runtime.bootstrap_addr,
            direct_room=room,
            tp_size=self.tp_world_size,
            kv_layout_hash=self.agentic_direct_runtime.layout_hash,
        )
        # Multi-P routing is fixed by the D worker's NUMA domain.  Publish the
        # destination together with DIRECT_READY so Router ingress never waits
        # on a P load query before acknowledging the next turn.
        if self.tp_rank == 0:
            if not self._publish_agentic_route(
                metadata.current,
                route="direct_ready",
                snapshot_tokens=len(all_tokens),
            ):
                return False
            self.agentic_snapshot_store.publish_direct_offer(manifest)
        source_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(all_tokens)
        ]
        source_digest = debug_kv_digest(
            self.agentic_direct_runtime.kv_pool, source_indices
        )
        if source_digest is not None:
            logger.info(
                "AgenticKV d_source_digest snapshot=%s digest=%s",
                manifest.snapshot_id,
                source_digest,
            )
        sender = self.agentic_direct_runtime.sender_class(
            mgr=self.agentic_direct_runtime.manager,
            bootstrap_addr=self.agentic_direct_runtime.bootstrap_addr,
            bootstrap_room=room,
            dest_tp_ranks=[self.tp_rank],
            pp_rank=0,
        )
        self.agentic_direct_candidates[manifest.snapshot_id] = {
            "req": req,
            "metadata": metadata,
            "tokens": list(all_tokens),
            "manifest": manifest,
            "sender": sender,
            "created_at": time.monotonic(),
            "claimed_at": None,
            "sent": False,
            "fallback_retry_at": 0.0,
            # Mooncake manifest lookup is synchronous.  Do not put one RPC per
            # pending candidate on every Decode scheduler tick: with many D
            # workers that control-plane work can leave an otherwise healthy
            # Decode batch waiting between token forwards.
            "manifest_next_poll_at": 0.0,
            "early_claim_next_poll_at": 0.0,
            "fast_arrival_seen": False,
            "fast_arrival_seen_at": None,
        }
        pending_command = self._agentic_tp_pending_candidate_commands.pop(
            manifest.snapshot_id, None
        )
        if pending_command is not None:
            self._apply_tp_candidate_command(pending_command)
        self.wake_decode_io_progress()
        logger.info(
            "AgenticKV direct_offer snapshot=%s tokens=%d room=%d threshold_s=%.3f",
            manifest.snapshot_id,
            manifest.token_count,
            room,
            self.agentic_fast_threshold,
        )
        return True

    def _agentic_try_early_claim(self, candidate, now: float) -> str:
        """Return absent or arrived for the next-turn ingress marker."""

        store = getattr(self, "agentic_early_claim_store", None)
        if store is None:
            return "absent"
        if candidate.get("fast_arrival_seen"):
            return "arrived"
        if now < candidate.get("early_claim_next_poll_at", 0.0):
            return "absent"
        candidate["early_claim_next_poll_at"] = (
            now + self.agentic_early_claim_poll_interval
        )
        manifest = candidate["manifest"]
        marker = store.read_arrival(
            manifest.request,
            not_before=manifest.created_at,
            max_age_seconds=max(
                5.0,
                self.agentic_fast_threshold
                + self.agentic_early_claim_post_timeout
                + 1.0,
            ),
        )
        if marker is None:
            return "absent"
        arrival_offset = max(
            0.0, float(marker["arrived_at"]) - float(manifest.created_at)
        )
        if arrival_offset > self.agentic_fast_threshold:
            # The marker exists, but the tool did not finish inside the fast
            # window.  It must use Host staging instead of extending D-HBM
            # ownership for another admission window.
            return "absent"
        candidate["fast_arrival_seen"] = True
        # Express the router's wall-clock arrival on this candidate's
        # monotonic timeline so scheduler polling delay does not silently
        # extend the additional P-admission window.
        candidate["fast_arrival_seen_at"] = (
            candidate["created_at"] + arrival_offset
        )
        logger.info(
            "AgenticKV fast_arrival_seen snapshot=%s tokens=%d "
            "tool_elapsed_s=%.6f admission_timeout_s=%.3f",
            manifest.snapshot_id,
            manifest.token_count,
            arrival_offset,
            self.agentic_early_claim_post_timeout,
        )
        return "arrived"

    def _agentic_release_early_claim(self, candidate, reason: str) -> None:
        store = getattr(self, "agentic_early_claim_store", None)
        if store is None:
            return
        manifest = candidate.get("manifest")
        if manifest is None:
            return
        was_seen = bool(candidate.pop("fast_arrival_seen", False))
        if getattr(self, "tp_world_size", 1) > 1:
            # The marker is a level-trigger shared by every D rank.  Removing
            # it when the first shard changes state makes later ranks misread
            # the same generation as an unconfirmed tool call.  Generation
            # ids are unique and the run-owned ready directory is deleted at
            # shutdown, so retaining these tiny files is bounded and safe.
            if was_seen:
                logger.info(
                    "AgenticKV fast_arrival_retained_tp snapshot=%s reason=%s",
                    manifest.snapshot_id,
                    reason,
                )
            return
        try:
            store.remove_arrival(manifest.request)
            store.remove_tool(manifest.request)
        except Exception:
            logger.exception(
                "AgenticKV early_claim release failed snapshot=%s reason=%s",
                manifest.snapshot_id,
                reason,
            )
            return
        if was_seen:
            logger.info(
                "AgenticKV fast_arrival_released snapshot=%s reason=%s",
                manifest.snapshot_id,
                reason,
            )

    def _agentic_try_final_confirmation(self, candidate) -> bool:
        store = getattr(self, "agentic_early_claim_store", None)
        if store is None:
            return False
        manifest = candidate["manifest"]
        marker = store.read_final(
            manifest.request,
            not_before=manifest.created_at,
            # not_before prevents an old generation marker from matching.
            # Keep the current marker visible across a temporarily stalled D
            # control loop instead of expiring it after only a few seconds.
            max_age_seconds=max(
                60.0,
                envs.SGLANG_AGENTIC_KV_STALE_SECONDS.get(),
                self.agentic_fast_threshold + 2.0,
            ),
        )
        return marker is not None

    def _agentic_try_tool_confirmation(self, candidate) -> bool:
        """Whether the application parser accepted this generation's tool call."""

        store = getattr(self, "agentic_early_claim_store", None)
        if store is None:
            # Legacy launchers have no parser-level decision channel.  Keep
            # their existing suffix-based behavior for compatibility.
            return True
        manifest = candidate["manifest"]
        marker = store.read_tool(
            manifest.request,
            not_before=manifest.created_at,
            max_age_seconds=max(
                60.0,
                envs.SGLANG_AGENTIC_KV_STALE_SECONDS.get(),
                self.agentic_fast_threshold + 2.0,
            ),
        )
        return marker is not None

    def _agentic_release_final_confirmation(self, candidate) -> None:
        store = getattr(self, "agentic_early_claim_store", None)
        if store is None:
            return
        if getattr(self, "tp_world_size", 1) > 1:
            return
        try:
            store.remove_final(candidate["manifest"].request)
        except Exception:
            logger.exception(
                "AgenticKV final confirmation cleanup failed snapshot=%s",
                candidate["manifest"].snapshot_id,
            )

    def _agentic_complete_final_candidate(self, candidate, now: float) -> bool:
        """Release a provisional Direct snapshot after the application ends."""

        if candidate.get("staging") or candidate.get("sent"):
            return False
        metadata = candidate["metadata"]
        manifest = self._agentic_direct_manifest(
            candidate, metadata, now, force=True
        )
        if manifest.state is SnapshotState.DIRECT_READY:
            terminal = self.agentic_snapshot_store.finalize_direct_offer(
                manifest,
                owner_id=f"d-final:{metadata.current.storage_id}",
            )
            if terminal is None:
                return False
        elif manifest.state is SnapshotState.DIRECT_LOADING:
            # P owns this generation.  Never let D invalidate P's in-flight
            # receive; the Direct completion/release path settles ownership.
            return False
        else:
            return False
        self._enqueue_agentic_release(candidate["req"], 0)
        self._cleanup_agentic_direct_sender(candidate)
        self._agentic_release_early_claim(candidate, "app_final")
        self._agentic_release_final_confirmation(candidate)
        self.agentic_direct_candidates.pop(manifest.snapshot_id, None)
        logger.info(
            "AgenticKV app_final_release snapshot=%s elapsed_s=%.6f",
            manifest.snapshot_id,
            now - candidate["created_at"],
        )
        return True

    def _agentic_fail_unconfirmed_tool_candidate(
        self, candidate, manifest, now: float
    ) -> bool:
        """Release an unconfirmed snapshot without declaring it terminal.

        A missing parser-level tool ACK only means that this generation is not
        eligible for Host/Mooncake preservation.  The application may still
        issue a repair generation, so publishing ``FINAL`` here would make a
        valid next turn contradict its parent state.  ``FAILED`` tells P to
        recompute that parent while preserving the trajectory semantics.
        """

        if candidate.get("staging") or candidate.get("sent"):
            return False
        if manifest.state is SnapshotState.DIRECT_LOADING:
            return False
        if manifest.state is not SnapshotState.DIRECT_READY:
            return False
        failed = self.agentic_snapshot_store.fail_direct_offer(
            manifest,
            owner_id=f"d-unconfirmed:{candidate['metadata'].current.storage_id}",
            reason="application_tool_unconfirmed",
        )
        if failed is None:
            return False
        self._enqueue_agentic_release(candidate["req"], 0)
        self._cleanup_agentic_direct_sender(candidate)
        self._agentic_release_early_claim(candidate, "unconfirmed_tool")
        self._agentic_release_final_confirmation(candidate)
        self.agentic_direct_candidates.pop(manifest.snapshot_id, None)
        logger.info(
            "AgenticKV unconfirmed_tool_recompute snapshot=%s elapsed_s=%.6f",
            manifest.snapshot_id,
            now - candidate["created_at"],
        )
        return True

    def _start_agentic_slow_snapshot(
        self,
        req: Req,
        metadata: AgenticRequestMetadata,
        all_tokens,
        direct_manifest: SnapshotManifest | None = None,
    ) -> bool | None:
        if self.cache_controller is None or self.decode_host_mem_pool is None:
            raise RuntimeError("D Host fallback is disabled in agentic hostless mode")
        aligned_len = len(all_tokens)
        if direct_manifest is not None and direct_manifest.state in {
            SnapshotState.DIRECT_READY,
            SnapshotState.DIRECT_LOADING,
        }:
            direct_manifest = self.agentic_snapshot_store.begin_slow_fallback(
                direct_manifest,
                owner_id=f"d:{metadata.current.storage_id}",
            )
            # P owns the direct claim.  Keep D KV and retry rather than racing
            # its manifest update or releasing a transfer still in progress.
            if direct_manifest is None:
                return None
        elif direct_manifest is not None and direct_manifest.state is not SnapshotState.SLOW_FALLBACK:
            raise RuntimeError(
                f"cannot start slow snapshot from {direct_manifest.state.value}"
            )
        token_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :aligned_len
        ]
        self.request_counter += 1
        ack_id = self.request_counter
        host_indices = self.cache_controller.write(
            device_indices=token_indices.long(),
            node_id=ack_id,
        )
        if host_indices is None:
            logger.error("Not enough D host memory for agentic snapshot %s", req.rid)
            self._publish_agentic_failure(
                metadata, "d_host_allocation_failed", direct_manifest
            )
            return False

        logical_hashes = self._compute_prefix_hash(all_tokens)
        namespace = page_namespace(metadata.current)
        backend = self.cache_controller.storage_backend
        manifest = None
        try:
            physical_keys, byte_size = backend.agentic_snapshot_layout(
                logical_hashes,
                host_indices,
                namespace,
            )
            manifest = SnapshotManifest(
                request=metadata.current,
                page_keys=physical_keys,
                token_count=aligned_len,
                byte_size=byte_size,
                state=SnapshotState.OFFLOADING,
                tool_type=metadata.tool_type,
                # The Agent receives the response immediately after this method;
                # use the same boundary for remaining-tool-time accounting.
                tool_started_at=time.time(),
                tp_size=self.tp_world_size,
            )
            if direct_manifest is not None:
                manifest = SnapshotManifest(
                    request=direct_manifest.request,
                    page_keys=physical_keys,
                    token_count=aligned_len,
                    byte_size=byte_size,
                    state=SnapshotState.SLOW_FALLBACK,
                    created_at=direct_manifest.created_at,
                    tool_type=metadata.tool_type,
                    tool_started_at=direct_manifest.tool_started_at,
                    claim_id=direct_manifest.claim_id,
                    token_digest=token_ids_digest(all_tokens),
                    tp_size=direct_manifest.tp_size,
                    kv_layout_hash=direct_manifest.kv_layout_hash,
                ).transition(SnapshotState.OFFLOADING)
        except Exception:
            logger.exception("Failed to describe agentic snapshot %s", req.rid)
            self._publish_agentic_failure(
                metadata, "snapshot_layout_failed", direct_manifest
            )

        reserved = manifest is not None and self._agentic_reserve(manifest)
        if manifest is not None and not reserved:
            logger.warning(
                "Agentic snapshot %s (%d bytes) exceeds its D Mooncake budget; "
                "next P turn will recompute",
                manifest.snapshot_id,
                manifest.byte_size,
            )
            self._publish_agentic_failure(
                metadata, "snapshot_budget_exceeded", direct_manifest
            )
            # D2H has already been submitted.  Keep the buffers alive until its
            # CUDA event completes, then release the request without publishing.
            manifest = None
        elif manifest is not None:
            try:
                if direct_manifest is None:
                    self.agentic_snapshot_store.begin_publish(manifest)
                else:
                    self.agentic_snapshot_store.continue_slow_publish(manifest)
            except Exception:
                self._agentic_cancel_reservation(manifest)
                logger.exception(
                    "Failed to begin agentic snapshot %s", manifest.snapshot_id
                )
                self._publish_agentic_failure(
                    metadata, "manifest_begin_failed", direct_manifest
                )
                manifest = None

        self.ongoing_offload[ack_id] = (
            req,
            host_indices,
            all_tokens,
            time.time(),
            0,
            aligned_len,
            manifest,
            namespace,
            logical_hashes,
        )
        logger.info(
            "AgenticKV slow_offload snapshot=%s tokens=%d bytes=%d from_direct=%s",
            metadata.current.snapshot_id,
            aligned_len,
            0 if manifest is None else manifest.byte_size,
            direct_manifest is not None,
        )
        # Do not add pending_responses: tool execution overlaps D2H and Put.
        return True

    def _start_agentic_host_staging(self, candidate, manifest) -> bool | None:
        """Offer a complete D-GPU snapshot to P; D ownership is retained."""

        self._assign_slow_prefill_target(candidate)
        metadata = candidate["metadata"]
        if manifest.state in {SnapshotState.DIRECT_READY, SnapshotState.DIRECT_LOADING}:
            fallback = self.agentic_snapshot_store.begin_slow_fallback(
                manifest, owner_id=f"d-host:{metadata.current.storage_id}"
            )
            if fallback is None:
                return None
        elif manifest.state is SnapshotState.SLOW_FALLBACK:
            fallback = manifest
        else:
            return False
        # begin_slow_fallback keeps the ownership claim for the complete slow
        # lifecycle.  Retain the returned manifest on the candidate before any
        # Shared-Arena operation that may raise, so a retry remains idempotent
        # instead of trying to reacquire its own persistent claim.
        candidate["manifest"] = fallback
        token_count = len(candidate["tokens"])
        token_indices = self.req_to_token_pool.req_to_token[
            candidate["req"].req_pool_idx, :token_count
        ]
        source_pages = kv_to_page_indices(
            token_indices.cpu().numpy(), self.page_size
        )
        logical_hashes = self._compute_prefix_hash(candidate["tokens"])
        bytes_per_page = sum(
            int(value)
            for value in self.agentic_direct_runtime.manager.kv_args.kv_item_lens
        )
        self.agentic_host_staging_client.offer(
            manifest=fallback,
            metadata=metadata,
            token_count=token_count,
            token_digest=token_ids_digest(candidate["tokens"]),
            logical_hashes=logical_hashes,
            byte_size=bytes_per_page * len(source_pages),
            arena_domain=int(
                candidate.get(
                    "selected_prefill_domain",
                    self.agentic_host_staging_client.arena_domain,
                )
            ),
            arena_numa_node=int(
                candidate.get(
                    "selected_arena_numa_nodes",
                    [self.agentic_host_staging_client.arena_numa_node]
                    * self.tp_world_size,
                )[self.tp_rank]
            ),
        )
        candidate["staging"] = True
        candidate["source_token_indices"] = token_indices
        logger.info(
            "AgenticKV host_staging_offer snapshot=%s tokens=%d pages=%d",
            fallback.snapshot_id,
            token_count,
            len(source_pages),
        )
        return True

    def _publish_agentic_failure(
        self,
        metadata: AgenticRequestMetadata,
        reason: str,
        current_manifest: SnapshotManifest | None = None,
    ) -> None:
        try:
            if current_manifest is not None:
                self.agentic_snapshot_store.mark_failed(
                    current_manifest,
                    reason=reason,
                    owner_claim_id=current_manifest.claim_id,
                )
            else:
                self.agentic_snapshot_store.publish_failure(
                    metadata.current,
                    reason=reason,
                    tool_type=metadata.tool_type,
                )
        except Exception:
            logger.exception(
                "Failed to publish agentic fallback marker %s (%s)",
                metadata.current.snapshot_id,
                reason,
            )
        self._publish_agentic_route(metadata.current, route="recompute")

    def _publish_agentic_route(
        self,
        request,
        *,
        route: str,
        snapshot_tokens: Optional[int] = None,
        prefill_domain: Optional[int] = None,
    ) -> bool:
        # Route markers are a multi-P coordination primitive.  Keeping this
        # behind the explicit 2P+ feature flag preserves the proven 1P path
        # byte-for-byte at runtime (no extra filesystem writes or fences).
        if os.environ.get(
            "SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS", ""
        ).strip().lower() not in {"1", "true", "yes", "on"}:
            return True
        store = getattr(self, "agentic_early_claim_store", None)
        if store is None:
            return True
        try:
            selected_domain = (
                int(prefill_domain)
                if prefill_domain is not None
                else int(os.environ.get("SGLANG_AGENTIC_KV_PREFILL_DOMAIN", "0"))
            )
            arena_numa = (
                self._prefill_domain_numa_nodes(selected_domain)[self.tp_rank]
                if prefill_domain is not None
                else envs.SGLANG_AGENTIC_KV_ARENA_NUMA_NODE.get()
            )
            store.publish_route(
                request,
                route=route,
                prefill_domain=selected_domain,
                arena_numa_node=(
                    arena_numa
                    if route in {"host_writing", "host_ready"}
                    else None
                ),
                snapshot_tokens=snapshot_tokens,
            )
            return True
        except (OSError, TypeError, ValueError):
            logger.exception(
                "Failed to publish agentic route snapshot=%s route=%s",
                request.snapshot_id,
                route,
            )
            return False

    def _agentic_reserve(self, manifest: SnapshotManifest) -> bool:
        if self.agentic_eviction_controller is None:
            return False
        if isinstance(
            self.agentic_eviction_controller, SharedSnapshotEvictionController
        ):
            return self.agentic_eviction_controller.reserve(manifest)
        return self.agentic_eviction_controller.reserve(manifest.byte_size)

    def _agentic_cancel_reservation(self, manifest: SnapshotManifest) -> None:
        if isinstance(
            self.agentic_eviction_controller, SharedSnapshotEvictionController
        ):
            self.agentic_eviction_controller.cancel(manifest)
        else:
            self.agentic_eviction_controller.cancel(manifest.byte_size)

    def check_offload_progress(self):
        """Check the progress of offload from device to host and backup from host to storage."""
        if getattr(self, "_decode_io_async_enabled", False):
            # One O(1) allocator read and bounded commit drain replace the old
            # synchronous O(pending snapshots) transport/control scan.
            size = max(1, int(self.token_to_kv_pool_allocator.size))
            available = max(
                0,
                min(size, int(self.token_to_kv_pool_allocator.available_size())),
            )
            self._agentic_cached_d_kv_usage = (size - available) / size
            self._drain_decode_io_events()
            self.wake_decode_io_progress()
        else:
            self._check_agentic_direct_progress()
        cc = self.cache_controller

        if cc is None:
            # D-hostless mode has no local D2H or Host->Store work queues.
            return

        qsizes = torch.tensor(
            [
                len(cc.ack_write_queue),
                cc.ack_backup_queue.qsize(),
            ],
            dtype=torch.int,
        )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                qsizes, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )

        n_write, n_backup = map(int, qsizes.tolist())
        self._check_offload_progress(n_write)
        self._check_backup_progress(n_backup)

        now = time.monotonic()
        timed_out = [
            req_id
            for req_id, (_, started_at) in self.pending_responses.items()
            if now - started_at >= self.response_backup_timeout
        ]
        if timed_out:
            raise TimeoutError(
                "Timed out waiting for Decode KV backup for requests "
                + ", ".join(timed_out[:8])
            )

    def _agentic_direct_kv_usage(self) -> float:
        """Return physical D KV occupancy as a fraction of the token pool."""

        if getattr(self, "_decode_io_async_enabled", False):
            return float(self._agentic_cached_d_kv_usage)

        size = max(1, int(self.token_to_kv_pool_allocator.size))
        available = max(
            0, min(size, int(self.token_to_kv_pool_allocator.available_size()))
        )
        return (size - available) / size

    def _agentic_direct_ready_timeout(
        self, low_pressure_timeout: float
    ) -> tuple[float, float]:
        """Keep cheap direct offers longer while D has ample KV headroom.

        A DIRECT_READY offer has not been claimed by P yet.  Falling back only
        because P's scheduler did not revisit the next-turn request within the
        short fast-tool window creates a slow-path avalanche at high D fanout.
        Under real D memory pressure we retain the short deadline; otherwise
        the existing handshake timeout is also the bounded ready grace period.
        """

        try:
            high_watermark = float(
                os.environ.get(
                    "SGLANG_AGENTIC_KV_DIRECT_D_HBM_HIGH_WATERMARK", "0.70"
                )
            )
        except ValueError:
            logger.exception("Invalid direct D-HBM high watermark")
            raise
        if not 0.0 < high_watermark < 1.0:
            raise ValueError(
                "SGLANG_AGENTIC_KV_DIRECT_D_HBM_HIGH_WATERMARK must be in "
                f"(0, 1), got {high_watermark}"
            )
        usage = self._agentic_direct_kv_usage()
        timeout = (
            self.agentic_fast_threshold
            if usage >= high_watermark
            else max(self.agentic_fast_threshold, float(low_pressure_timeout))
        )
        return timeout, usage

    def _agentic_direct_manifest(
        self,
        candidate,
        metadata: AgenticRequestMetadata,
        now: float,
        *,
        force: bool = False,
    ) -> SnapshotManifest:
        """Return a cached direct manifest, periodically refreshing Mooncake.

        Direct claims are control-plane state and do not need token-rate
        polling.  A bounded polling interval preserves prompt handoff latency
        while preventing O(pending direct offers) synchronous Mooncake reads
        on every Decode scheduler iteration.  Callers force one final refresh
        at a fallback boundary so a concurrent P claim is never ignored.
        """

        manifest = candidate["manifest"]
        if not force and now < candidate.get("manifest_next_poll_at", 0.0):
            return manifest
        try:
            poll_interval = float(
                os.environ.get(
                    "SGLANG_AGENTIC_KV_DIRECT_MANIFEST_POLL_INTERVAL", "0.10"
                )
            )
        except ValueError:
            logger.exception("Invalid direct manifest poll interval")
            raise
        if poll_interval <= 0.0:
            raise ValueError(
                "SGLANG_AGENTIC_KV_DIRECT_MANIFEST_POLL_INTERVAL must be > 0, "
                f"got {poll_interval}"
            )
        current = self.agentic_snapshot_store.load(
            metadata.current, require_ready=False
        )
        if current is not None:
            manifest = current
            candidate["manifest"] = current
        candidate["manifest_next_poll_at"] = now + poll_interval
        return manifest

    def _check_agentic_direct_progress(self, *, progress_relay: bool = True) -> None:
        if self.tp_world_size > 1 and self.tp_rank != 0:
            self._check_agentic_tp_follower_progress(progress_relay=progress_relay)
            return
        relay_worker = getattr(self, "agentic_relay_worker", None)
        if progress_relay and relay_worker is not None:
            try:
                relay_worker.poll()
            except Exception:
                # A relay failure must not terminate Decode.  Source D workers
                # retain their original KV and the ledger falls back to direct
                # cross-NUMA D2H after the relay drains its DMA.
                logger.exception("Agentic relay worker poll failed")
        if not self.agentic_direct_candidates:
            return
        now = time.monotonic()
        handshake_timeout = max(
            self.agentic_fast_threshold,
            envs.SGLANG_AGENTIC_KV_DIRECT_HANDSHAKE_TIMEOUT.get(),
        )
        staging_entries = None
        for snapshot_id, candidate in list(self.agentic_direct_candidates.items()):
            req = candidate["req"]
            metadata = candidate["metadata"]
            if self._agentic_try_final_confirmation(
                candidate
            ) and self._agentic_complete_final_candidate(candidate, now):
                continue
            if candidate.get("staging"):
                staging_ledger = getattr(
                    self.agentic_host_staging_client, "ledger", None
                )
                if staging_ledger is None:
                    outcome = self.agentic_host_staging_client.progress(
                        candidate, candidate["source_token_indices"]
                    )
                else:
                    if staging_entries is None:
                        # All D-side staging candidates share one /dev/shm ledger.
                        # Decode must not reread and reparse that complete JSON
                        # document once per candidate on every scheduler tick.
                        staging_entries = staging_ledger.snapshot_entries()
                    outcome = self.agentic_host_staging_client.progress(
                        candidate,
                        candidate["source_token_indices"],
                        entry_snapshot=staging_entries.get(snapshot_id),
                    )
                if outcome == "host_ready":
                    # This is the only slow-path release point: P has ACKed all
                    # chunk D2H events and committed the complete Host snapshot.
                    if not self._publish_agentic_route(
                        metadata.current,
                        route="host_ready",
                        snapshot_tokens=candidate["manifest"].token_count,
                        prefill_domain=candidate.get("selected_prefill_domain"),
                    ):
                        # In multi-P mode this marker commits the P-domain
                        # destination.  Retain D KV and retry if publishing it
                        # fails instead of routing the next turn incorrectly.
                        continue
                    self._enqueue_agentic_release(req, 0)
                    self._cleanup_agentic_direct_sender(candidate)
                    self._agentic_release_early_claim(candidate, "host_ready")
                    self.agentic_direct_candidates.pop(snapshot_id, None)
                    logger.info("AgenticKV d_release_after_p_host snapshot=%s", snapshot_id)
                elif outcome == "failed":
                    # D still owns the complete HBM copy here.  Hostless mode
                    # deliberately has no large emergency D Host pool: mark
                    # this generation unavailable, then release it intact so
                    # the next P turn safely recomputes rather than deadlocking.
                    if self.agentic_hostless:
                        self._publish_agentic_failure(
                            metadata, "p_host_staging_failed", candidate["manifest"]
                        )
                        self._enqueue_agentic_release(req, 0)
                        self._cleanup_agentic_direct_sender(candidate)
                        self._agentic_release_early_claim(
                            candidate, "host_staging_failed"
                        )
                        self.agentic_direct_candidates.pop(snapshot_id, None)
                        logger.warning(
                            "AgenticKV host_staging_fail_soft snapshot=%s; "
                            "next turn will recompute",
                            snapshot_id,
                        )
                        continue
                    # Compatibility mode retains the original emergency path.
                    try:
                        started = self._start_agentic_slow_snapshot(
                            req,
                            metadata,
                            candidate["tokens"],
                            direct_manifest=candidate["manifest"],
                        )
                    except Exception:
                        candidate["fallback_retry_at"] = now + 0.1
                        logger.exception(
                            "Emergency Mooncake fallback failed for %s; retaining D HBM",
                            snapshot_id,
                        )
                        continue
                    if started:
                        self._agentic_release_early_claim(
                            candidate, "emergency_slow_path"
                        )
                        self.agentic_direct_candidates.pop(snapshot_id, None)
                    elif started is False and req.req_pool_idx != -1:
                        self._enqueue_agentic_release(req, 0)
                        self._agentic_release_early_claim(
                            candidate, "emergency_recompute"
                        )
                        self.agentic_direct_candidates.pop(snapshot_id, None)
                continue
            should_fallback = False
            try:
                poll = candidate["sender"].poll()
            except Exception:
                logger.exception("Direct D->P sender failed for %s", snapshot_id)
                poll = KVPoll.Failed

            if candidate["sent"]:
                if poll == KVPoll.Success:
                    if not candidate.get("local_send_complete"):
                        candidate["local_send_complete"] = True
                        logger.info(
                            "AgenticKV direct_rank_send_complete snapshot=%s "
                            "rank=%d/%d elapsed_s=%.6f",
                            snapshot_id,
                            self.tp_rank,
                            self.tp_world_size,
                            now - candidate["created_at"],
                        )
                elif poll == KVPoll.Failed:
                    should_fallback = True
                # TP=1 retains the old zero-metadata completion path.  TP>1
                # keeps every source shard pinned until the logical manifest
                # records ACKs from all destination ranks, so a late rank
                # failure can still fall back without losing half a snapshot.
                if not should_fallback and self.tp_world_size == 1:
                    self._enqueue_agentic_release(req, 0)
                    self._cleanup_agentic_direct_sender(candidate)
                    self._agentic_release_early_claim(candidate, "direct_complete")
                    self.agentic_direct_candidates.pop(snapshot_id, None)
                    continue

            manifest = self._agentic_direct_manifest(candidate, metadata, now)
            if not candidate["sent"] and manifest.state is SnapshotState.DIRECT_LOADING:
                if candidate["claimed_at"] is None:
                    candidate["claimed_at"] = now
                if poll == KVPoll.WaitingForInput:
                    token_indices = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, : len(candidate["tokens"])
                    ]
                    send_digest = debug_kv_digest(
                        self.agentic_direct_runtime.kv_pool, token_indices
                    )
                    if send_digest is not None:
                        debug_page_indices = kv_to_page_indices(
                            token_indices.cpu().numpy(), self.page_size
                        ).tolist()
                        logger.info(
                            "AgenticKV d_send_digest snapshot=%s digest=%s "
                            "page_indices=%s",
                            snapshot_id,
                            send_digest,
                            debug_page_indices,
                        )
                    page_indices = kv_to_page_indices(
                        token_indices.cpu().numpy(), self.page_size
                    )
                    candidate["sender"].init(len(page_indices), aux_index=0)
                    candidate["sender"].send(page_indices)
                    candidate["sent"] = True
                elif poll == KVPoll.Failed or (
                    now - candidate["claimed_at"] >= handshake_timeout
                ):
                    should_fallback = True
            elif not candidate["sent"] and manifest.state is SnapshotState.DIRECT_READY:
                direct_elapsed = now - candidate["created_at"]
                if getattr(self, "agentic_early_claim_store", None) is not None:
                    early_status = self._agentic_try_early_claim(candidate, now)
                    if early_status == "arrived":
                        arrived_at = candidate["fast_arrival_seen_at"]
                        if now - arrived_at >= self.agentic_early_claim_post_timeout:
                            should_fallback = True
                    elif direct_elapsed >= self.agentic_fast_threshold:
                        should_fallback = True
                else:
                    ready_timeout, d_kv_usage = self._agentic_direct_ready_timeout(
                        handshake_timeout
                    )
                    if direct_elapsed >= ready_timeout:
                        should_fallback = True
                    elif (
                        direct_elapsed >= self.agentic_fast_threshold
                        and not candidate.get("extended_wait_logged")
                    ):
                        candidate["extended_wait_logged"] = True
                        logger.info(
                            "AgenticKV direct_wait_extended snapshot=%s "
                            "elapsed_s=%.6f timeout_s=%.3f d_kv_usage=%.4f",
                            snapshot_id,
                            direct_elapsed,
                            ready_timeout,
                            d_kv_usage,
                        )
            elif manifest.state is SnapshotState.CONSUMED:
                # P marks CONSUMED only after its receiver observed success.
                self._enqueue_agentic_release(req, 0)
                self._cleanup_agentic_direct_sender(candidate)
                self._agentic_release_early_claim(candidate, "consumed")
                self.agentic_direct_candidates.pop(snapshot_id, None)

            elif (
                manifest.state is SnapshotState.SLOW_FALLBACK
                and getattr(self, "tp_world_size", 1) > 1
                and self.agentic_host_staging_client is not None
                and not candidate.get("staging")
            ):
                # Another D rank won the request-level fallback transition.
                # This rank still owns its physical KV-head shard and must
                # join the same Shared-Arena group before releasing D HBM.
                try:
                    started = self._start_agentic_host_staging(candidate, manifest)
                except Exception:
                    candidate["fallback_retry_at"] = now + 0.1
                    logger.exception(
                        "Agentic TP fallback join failed for %s; retrying",
                        snapshot_id,
                    )
                    continue
                if started:
                    self._publish_agentic_route(
                        metadata.current,
                        route="host_writing",
                        snapshot_tokens=candidate["manifest"].token_count,
                        prefill_domain=candidate.get("selected_prefill_domain"),
                    )
                    continue
            elif manifest.state not in {
                SnapshotState.DIRECT_READY,
                SnapshotState.DIRECT_LOADING,
            }:
                logger.warning(
                    "Direct snapshot %s moved to %s before transfer; releasing D KV",
                    snapshot_id,
                    manifest.state.value,
                )
                self._enqueue_agentic_release(req, 0)
                self._agentic_release_early_claim(
                    candidate, f"manifest_{manifest.state.value}"
                )
                self.agentic_direct_candidates.pop(snapshot_id, None)

            if should_fallback:
                if now < candidate["fallback_retry_at"]:
                    continue
                # Close the final-marker/fallback race.  The application may
                # finish parsing an invalid tool-looking response after the
                # first check above; recheck immediately before changing the
                # manifest to SLOW_FALLBACK or offering any Host extent.
                if self._agentic_try_final_confirmation(
                    candidate
                ) and self._agentic_complete_final_candidate(candidate, now):
                    continue
                # A P claim may race the ready deadline between periodic
                # polls.  Refresh once here; DIRECT_LOADING owns the snapshot
                # and must be allowed to finish instead of entering fallback.
                manifest = self._agentic_direct_manifest(
                    candidate, metadata, now, force=True
                )
                if not self._agentic_try_tool_confirmation(candidate):
                    # A configured suffix is only a provisional hint.  Native
                    # serving continues a trajectory only after its tool
                    # parser accepts the call.  Without that application ACK,
                    # fail soft to recomputation; never allocate Shared Arena
                    # or Mooncake storage for the generation.
                    if self._agentic_fail_unconfirmed_tool_candidate(
                        candidate, manifest, now
                    ):
                        continue
                    candidate["fallback_retry_at"] = now + 0.05
                    continue
                if (
                    not candidate["sent"]
                    and manifest.state is SnapshotState.DIRECT_LOADING
                ):
                    candidate["claimed_at"] = candidate["claimed_at"] or now
                    continue
                self._agentic_release_early_claim(candidate, "slow_fallback")
                logger.info(
                    "AgenticKV direct_fallback snapshot=%s elapsed_s=%.6f "
                    "state=%s d_kv_usage=%.4f",
                    snapshot_id,
                    now - candidate["created_at"],
                    manifest.state.value,
                    self._agentic_direct_kv_usage(),
                )
                try:
                    if self.agentic_host_staging_client is not None:
                        started = self._start_agentic_host_staging(candidate, manifest)
                    else:
                        started = self._start_agentic_slow_snapshot(
                            req,
                            metadata,
                            candidate["tokens"],
                            direct_manifest=manifest,
                        )
                except Exception:
                    # A storage metadata operation can be transiently busy.
                    # Never terminate the Decode scheduler while it still owns
                    # the complete D-GPU snapshot; retain it and retry shortly.
                    candidate["fallback_retry_at"] = now + 0.1
                    logger.exception(
                        "Agentic direct fallback transition failed for %s; retrying",
                        snapshot_id,
                    )
                    continue
                if started is None:
                    continue
                if started and candidate.get("staging"):
                    # Shared-Arena now owns the slow-path lifecycle.  Publish
                    # its NUMA-local P immediately so Router can redirect an
                    # already-submitted request while D2H continues.  D still
                    # retains source HBM until the later HOST_READY ACK.
                    self._publish_agentic_route(
                        metadata.current,
                        route="host_writing",
                        snapshot_tokens=candidate["manifest"].token_count,
                        prefill_domain=candidate.get("selected_prefill_domain"),
                    )
                    # The original reverse-NIXL room is reused by relay chunk
                    # zero.  It is cleaned after HOST_READY or direct fallback.
                    continue
                if not started and req.req_pool_idx != -1:
                    self._enqueue_agentic_release(req, 0)
                self._cleanup_agentic_direct_sender(candidate)
                self.agentic_direct_candidates.pop(snapshot_id, None)

    def _check_agentic_tp_follower_progress(
        self, *, progress_relay: bool = True
    ) -> None:
        """Execute TP0 commands without making route or lifecycle decisions."""

        relay_worker = getattr(self, "agentic_relay_worker", None)
        if progress_relay and relay_worker is not None:
            relay_worker.poll()
        for candidate in list(self.agentic_direct_candidates.values()):
            action = candidate.get("tp_command", "wait")
            if action == "wait":
                continue
            if action == "slow":
                manifest = candidate.get("manifest")
                if not candidate.get("staging"):
                    if (
                        manifest is None
                        or manifest.state is not SnapshotState.SLOW_FALLBACK
                    ):
                        continue
                    if not self._start_agentic_host_staging(candidate, manifest):
                        continue
                client = self.agentic_host_staging_client
                if client is not None:
                    client.progress(candidate, candidate["source_token_indices"])
                continue

            # DIRECT means rank 0 observed the group-visible P claim.  The
            # follower performs only its local NIXL send and never times out,
            # falls back, publishes a route, or frees KV independently.
            sender = candidate["sender"]
            poll = sender.poll()
            if not candidate.get("sent") and poll == KVPoll.WaitingForInput:
                req = candidate["req"]
                token_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, : len(candidate["tokens"])
                ]
                page_indices = kv_to_page_indices(
                    token_indices.cpu().numpy(), self.page_size
                )
                sender.init(len(page_indices), aux_index=0)
                sender.send(page_indices)
                candidate["sent"] = True
            elif candidate.get("sent") and poll == KVPoll.Success:
                candidate["local_send_complete"] = True

    def _cleanup_agentic_direct_sender(self, candidate) -> None:
        runtime = getattr(self, "agentic_direct_runtime", None)
        if runtime is None:
            return
        manager = runtime.manager
        room = candidate["manifest"].direct_room
        if room is None:
            return
        manager.request_status.pop(room, None)
        manager.failure_records.pop(room, None)
        transfer_infos = getattr(manager, "transfer_infos", None)
        if transfer_infos is not None:
            transfer_infos.pop(room, None)

    def is_response_pending(self, req_id: str) -> bool:
        return req_id in self.pending_responses

    def pop_ready_responses(self):
        ready = self.ready_responses
        self.ready_responses = []
        return ready

    def _check_offload_progress(self, finish_count):
        """Check the progress of offload from device to host."""
        while finish_count > 0:
            _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                record = self.ongoing_offload.pop(ack_id)
                (
                    req,
                    host_indices,
                    incremental_tokens,
                    start_time,
                    start,
                    end,
                ) = record[:6]
                agentic = len(record) > 6

                prior_hash = (
                    self.offloaded_state[req.rid].last_hash
                    if req.rid in self.offloaded_state
                    else None
                )
                if req.finished():
                    self._release_finished_req(req, start)
                else:
                    kv_indices = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, start:end
                    ]
                    self.token_to_kv_pool_allocator.free(kv_indices)

                if agentic:
                    manifest, namespace, logical_hashes = record[6:]
                    if manifest is None:
                        self.decode_host_mem_pool.free(host_indices)
                    else:
                        try:
                            ack_id = self.cache_controller.write_storage(
                                host_indices,
                                incremental_tokens,
                                hash_value=logical_hashes,
                                storage_namespace=namespace,
                            )
                            self.ongoing_backup[ack_id] = (
                                req.rid,
                                host_indices,
                                start_time,
                                manifest,
                            )
                        except Exception:
                            self._agentic_cancel_reservation(manifest)
                            try:
                                self.agentic_snapshot_store.fail_publish(manifest)
                            except Exception:
                                logger.exception(
                                    "Failed to clean agentic snapshot %s after Put submission error",
                                    manifest.snapshot_id,
                                )
                            self.decode_host_mem_pool.free(host_indices)
                            logger.exception(
                                "Failed to submit agentic snapshot Put %s",
                                manifest.snapshot_id,
                            )
                    last_hash = logical_hashes[-1]
                else:
                    last_hash = self._trigger_backup(
                        req, host_indices, incremental_tokens, start_time, prior_hash
                    )
                if req.rid in self.offloaded_state:
                    self.offloaded_state[req.rid].last_hash = last_hash
            finish_count -= 1

    def _release_finished_req(self, req: Req, start_offset: int):
        if not self.tree_cache.disable:
            if start_offset != 0:
                raise RuntimeError(
                    "PD Decode Radix release requires a complete request snapshot"
                )
            is_agentic_generation = hasattr(req, "_pd_transport_extra_key")
            committed_len = int(getattr(req, "kv_committed_len", 0))
            release_kv_cache(
                req,
                self.tree_cache,
                is_insert=not is_agentic_generation,
            )
            if is_agentic_generation:
                release_generation = getattr(
                    self.tree_cache, "release_request_generation_cache", None
                )
                if release_generation is None:
                    raise RuntimeError(
                        "PD agentic Decode Radix cache must support "
                        "request-generation release"
                    )
                released = release_generation(
                    req,
                    committed_len=committed_len,
                    event_prefix="d_generation_release",
                    allow_shared_ancestors=True,
                )
                logger.info(
                    "AgenticKV d_generation_release tokens=%d req=%s extra_key=%s",
                    released,
                    req.rid,
                    req.extra_key,
                )
            if req.rid in self.offloaded_state:
                del self.offloaded_state[req.rid]
            return
        kv_committed_len = req.pop_committed_kv_cache()
        start = start_offset
        end = kv_committed_len
        # Free the incremental part of the request (NSA-aware)
        kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, start:end]
        self.token_to_kv_pool_allocator.free(kv_indices)

        # Free over-allocated KV cache slots (e.g. from speculative decoding v2).
        # Without spec v2, start_p == end_p so this is a no-op.
        start_p, end_p = req.pop_overallocated_kv_cache()
        if self.page_size > 1:
            start_p = ceil_align(start_p, self.page_size)
        if start_p < end_p:
            overalloc_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, start_p:end_p
            ]
            self.token_to_kv_pool_allocator.free(overalloc_indices)

        self.req_to_token_pool.free(req)
        self.tree_cache.protected_size_ -= len(req.prefix_indices)
        if req.rid in self.offloaded_state:
            del self.offloaded_state[req.rid]

    def _check_backup_progress(self, finish_count):
        """Check the progress of backup from host to storage."""
        for _ in range(finish_count):
            storage_operation = self.cache_controller.ack_backup_queue.get()
            ack_id = storage_operation.id
            record = self.ongoing_backup.pop(ack_id)
            req_id, host_indices, start_time = record[:3]
            manifest = record[3] if len(record) > 3 else None

            if manifest is not None:
                if storage_operation.completed_tokens != manifest.token_count:
                    self._agentic_cancel_reservation(manifest)
                    try:
                        self.agentic_snapshot_store.fail_publish(manifest)
                    except Exception:
                        logger.exception(
                            "Failed to clean incomplete agentic snapshot %s",
                            manifest.snapshot_id,
                        )
                    logger.error(
                        "Incomplete agentic snapshot %s: stored=%d expected=%d",
                        manifest.snapshot_id,
                        storage_operation.completed_tokens,
                        manifest.token_count,
                    )
                else:
                    try:
                        ready = self.agentic_snapshot_store.commit_publish(
                            manifest.request
                        )
                        self.agentic_eviction_controller.commit(ready)
                        logger.info(
                            "AgenticKV mooncake_ready snapshot=%s tokens=%d bytes=%d elapsed_s=%.6f",
                            ready.snapshot_id,
                            ready.token_count,
                            ready.byte_size,
                            time.time() - start_time,
                        )
                    except Exception:
                        self._agentic_cancel_reservation(manifest)
                        try:
                            current = self.agentic_snapshot_store.load(
                                manifest.request, require_ready=False
                            )
                            if (
                                current is not None
                                and current.state is SnapshotState.OFFLOADING
                            ):
                                self.agentic_snapshot_store.fail_publish(current)
                        except Exception:
                            logger.exception(
                                "Failed to clean uncommitted agentic snapshot %s",
                                manifest.snapshot_id,
                            )
                        logger.exception(
                            "Failed to commit agentic snapshot %s",
                            manifest.snapshot_id,
                        )

            # Release host memory
            self.decode_host_mem_pool.free(host_indices)

            logger.debug(
                f"Finished backup request {req_id}, free host memory, len:{len(host_indices)}, cost time:{time.time() - start_time:.2f} seconds."
            )

            pending = self.pending_responses.pop(req_id, None)
            if pending is not None:
                self.ready_responses.append(pending[0])

    def _trigger_backup(
        self, req, host_indices, incremental_tokens, start_time, prior_hash
    ):
        """Trigger async backup from host to storage."""
        page_hashes = self._compute_prefix_hash(incremental_tokens, prior_hash)
        ack_id = self.cache_controller.write_storage(
            host_indices,
            incremental_tokens,
            hash_value=page_hashes,
        )
        self.ongoing_backup[ack_id] = (req.rid, host_indices, start_time)
        return page_hashes[-1] if len(page_hashes) > 0 else prior_hash

    def _compute_prefix_hash(self, tokens, prior_hash=""):
        page_hashes = []
        last_hash = prior_hash
        for offset in range(0, len(tokens), self.page_size):
            page_tokens = tokens[offset : offset + self.page_size]
            last_hash = get_hash_str(page_tokens, last_hash)
            page_hashes.append(last_hash)
        return page_hashes

    def finalize_release_on_finish(self, req: Req):
        """Free any remaining tail KV that was not offloaded due to non-aligned length."""
        if req.req_pool_idx == -1:
            return
        if not self.tree_cache.disable:
            # Radix owns prefix references as one request lifecycle.  The
            # legacy ChunkCache path below frees the prefill-aligned region and
            # tail separately, which would free shared prefix pages before
            # Radix can insert/unlock the finished request.
            self._release_finished_req(req, 0)
            return
        state = self.offloaded_state.get(req.rid)
        if state is None:
            prefill_len = len(req.origin_input_ids) // self.page_size * self.page_size
            inc_len = 0
        else:
            prefill_len = state.prefill_len
            inc_len = state.inc_len
        # If no incremental offload ever happened, the prefill-aligned part was never freed.
        # Free the prefill portion on request finish to avoid leaks.
        if prefill_len > 0 and inc_len == 0:
            token_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx]
            self.token_to_kv_pool_allocator.free(token_indices[:prefill_len])
            logger.info(
                f"Finalize release: freed prefill-aligned KV for req {req.rid}, len:{prefill_len}"
            )
        start_offset = prefill_len + inc_len
        self._release_finished_req(req, start_offset)
