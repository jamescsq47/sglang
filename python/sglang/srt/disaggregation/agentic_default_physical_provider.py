"""Default physical provider for the file-free multi-node V2 runtime.

This module composes existing NIXL Direct and source-local Host primitives; it
does not introduce another transport or allocator.  Rank zero owns path
policy, while every rank executes the immutable command for its shard.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)

from sglang.srt.disaggregation.agentic_group_protocol import (
    GenerationKey,
    GroupCommand,
    LinkIntent,
    LinkParticipant,
    Owner,
)
from sglang.srt.disaggregation.agentic_group_transfer import (
    CallbackPathHandler,
    GroupTransferPlan,
    PreparedRankTransfer,
    RankPathHandler,
    RemoteHostLoadExecutor,
    RemoteHostLoadPayload,
    TransferOperation,
)
from sglang.srt.disaggregation.agentic_hybrid_transfer import (
    p2d_mamba_checkpoint_tokens,
)
from sglang.srt.disaggregation.agentic_host_transfer import (
    HostDescriptorSet,
    build_host_restore_plan,
    make_host_restore_source_handler,
    make_no_io_host_endpoint_handler,
)
from sglang.srt.disaggregation.agentic_memory_authority import (
    LeaseKind,
    PhysicalMemoryLease,
    RequestGenerationAttempt,
)
from sglang.srt.disaggregation.agentic_multinode_policy import (
    D2PPolicyActor,
    P2DPolicyActor,
)
from sglang.srt.disaggregation.agentic_native_memory_adapter import (
    NativeRequestMemoryAdapter,
    NativeSourceSnapshot,
    common_page_prefix_tokens,
    token_page_chain_hashes,
    decode_state_slot_count,
    hybrid_state_allocator,
    prefill_state_slot_count,
)


_D2P_CHILD_ARRIVED = "d2p_child_arrived_v1"
_P2D_CHILD_ARRIVED = "p2d_child_arrived_v1"


@dataclass(slots=True)
class _D2PHostEntry:
    key: GenerationKey
    source_group: str
    token_count: int
    rank_bytes: tuple[int, ...]
    durable_at: float
    evicting: bool = False
from sglang.srt.disaggregation.agentic_nixl_direct_adapter import (
    DirectEndpoint,
    NixlDirectIOExecutor,
    NixlDirectOperation,
    NixlDirectShard,
)
from sglang.srt.disaggregation.agentic_remote_host_worker import (
    RemoteHostRankWorker,
)
from sglang.srt.disaggregation.agentic_source_host import (
    CudaSourceHostCopyBackend,
    HostDirection,
    SourceHostStorePayload,
    SourceLocalHostArena,
    make_source_host_eviction_handler,
    make_source_host_store_path,
)
from sglang.srt.disaggregation.agentic_transfer_queues import (
    TransferExecutor,
    TransferPath,
)
from sglang.srt.disaggregation.utils import DisaggregationMode, kv_to_page_indices


def _reverse_bootstrap_port() -> int:
    port = int(os.getenv("SGLANG_AGENTIC_KV_DIRECT_BOOTSTRAP_PORT", "0"))
    if port <= 0:
        raise RuntimeError(
            "multi-node Direct requires a dedicated "
            "SGLANG_AGENTIC_KV_DIRECT_BOOTSTRAP_PORT"
        )
    return port


def _values(command: GroupCommand) -> Mapping[str, Any]:
    value = command.payload.get("transfer", {})
    if not isinstance(value, Mapping):
        raise ValueError("transfer payload must be a mapping")
    return value


def _operation(command: GroupCommand) -> TransferOperation:
    header = command.payload.get("agentic_data_plane", {})
    if not isinstance(header, Mapping):
        raise ValueError("agentic_data_plane header must be a mapping")
    return TransferOperation(str(header["operation"]))


def _room(
    key: GenerationKey, path: TransferPath, target_group: str = ""
) -> int:
    raw = f"{key.run_id}:{key.snapshot_id}:{path.value}:{target_group}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") & (
        (1 << 63) - 1
    )


class _DispatchExecutor(TransferExecutor):
    def __init__(self, select: Callable[[Any], TransferExecutor]) -> None:
        self._select = select

    def submit(self, attempt, notify):
        payload = attempt.payload
        if isinstance(payload, _TargetPrepared):
            payload = payload.io_payload
            from sglang.srt.disaggregation.agentic_transfer_queues import (
                TransferAttempt,
            )

            attempt = TransferAttempt(
                snapshot_id=attempt.snapshot_id,
                attempt_id=attempt.attempt_id,
                lease_id=attempt.lease_id,
                path=attempt.path,
                payload=payload,
                channel=attempt.channel,
            )
        executor = self._select(payload)
        return executor, executor.submit(attempt, notify)

    def progress(self, handle):
        executor, local = handle
        return executor.progress(local)

    def request_cancel(self, handle, notify):
        executor, local = handle
        executor.request_cancel(local, notify)

    def install_started_callback(self, handle, callback):
        executor, local = handle
        install = getattr(executor, "install_started_callback", None)
        if callable(install):
            install(local, callback)
        else:
            callback()

    def close(self) -> None:
        # Concrete executors are closed by the provider exactly once.
        return None


class _DispatchHandler(RankPathHandler):
    def __init__(
        self,
        select: Callable[[GroupCommand], RankPathHandler],
        native_stream: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._select = select
        self._native_stream = native_stream
        self._chosen: dict[tuple[GenerationKey, int], RankPathHandler] = {}
        self._lock = threading.Lock()

    def _native_context(self):
        if self._native_stream is None:
            return nullcontext()
        return torch.cuda.stream(self._native_stream())

    @staticmethod
    def _key(command: GroupCommand):
        return command.key, command.attempt

    def prepare(self, command):
        handler = self._select(command)
        with self._lock:
            key = self._key(command)
            if key in self._chosen:
                raise RuntimeError("duplicate handler prepare for one attempt")
            self._chosen[key] = handler
        try:
            with self._native_context():
                return handler.prepare(command)
        except BaseException:
            with self._lock:
                self._chosen.pop(key, None)
            raise

    def _handler(self, command):
        with self._lock:
            return self._chosen[self._key(command)]

    def begin_io(self, command, prepared):
        self._handler(command).begin_io(command, prepared)

    def finish_io(self, command, prepared, completion):
        self._handler(command).finish_io(command, prepared, completion)

    def prepare_handoff(self, command, prepared, completion):
        with self._native_context():
            self._handler(command).prepare_handoff(command, prepared, completion)

    def commit(self, command, prepared, completion):
        try:
            with self._native_context():
                self._handler(command).commit(command, prepared, completion)
        finally:
            with self._lock:
                self._chosen.pop(self._key(command), None)

    def abort(self, command, prepared, completion):
        try:
            with self._native_context():
                self._handler(command).abort(command, prepared, completion)
        finally:
            with self._lock:
                self._chosen.pop(self._key(command), None)

    def retire(self, command) -> None:
        """Forget no-I/O participants that never execute commit/abort."""

        with self._lock:
            self._chosen.pop(self._key(command), None)

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._chosen)


@dataclass(slots=True)
class _TargetPrepared:
    lease: PhysicalMemoryLease
    req: Any
    io_payload: Any
    sampled_token_id: Optional[int] = None
    mamba_checkpoint_tokens: Optional[int] = None


class _TargetHandler(RankPathHandler):
    """Reserve, fill, bind and publish a target through one authority."""

    def __init__(self, provider: "AgenticDefaultPhysicalProvider", *, direct: bool):
        self.provider = provider
        self.direct = bool(direct)

    def prepare(self, command: GroupCommand) -> PreparedRankTransfer:
        # Parse immutable wire metadata before reserving physical memory.  A
        # malformed command must not create a lease that no prepared handler
        # exists to release.
        values = _values(command)
        sampled_value = values.get("sampled_token_id")
        sampled_token_id = (
            int(sampled_value) if sampled_value is not None else None
        )
        checkpoint_value = values.get("mamba_checkpoint_tokens")
        mamba_checkpoint_tokens = (
            int(checkpoint_value) if checkpoint_value is not None else None
        )
        lease, req = self.provider._reserve_target(command)
        try:
            payload = (
                self.provider._direct_target_payload(command, lease)
                if self.direct
                else self.provider._host_target_payload(command, lease)
            )
        except BaseException:
            self.provider.context.authority.request_release(lease.lease_id)
            self.provider.context.authority.commit_release(
                lease.lease_id, reason="target_prepare_failed"
            )
            self.provider._target_prepared.pop(lease.lease_id, None)
            raise
        if lease.kind is LeaseKind.DECODE_RESERVATION and sampled_token_id is None:
            self.provider.context.authority.request_release(lease.lease_id)
            self.provider.context.authority.commit_release(
                lease.lease_id, reason="target_prepare_missing_sampled_token"
            )
            self.provider._target_prepared.pop(lease.lease_id, None)
            raise RuntimeError("P2D target is missing sampled_token_id")
        if lease.kind is not LeaseKind.DECODE_RESERVATION:
            sampled_token_id = None
        self.provider._target_prepared[lease.lease_id] = _TargetPrepared(
            lease,
            req,
            payload,
            sampled_token_id,
            mamba_checkpoint_tokens,
        )
        # Only the physical executor payload is allowed to enter a transport
        # queue.  The full target state is retained by lease id for the
        # bind/publish barrier and is never stuffed into this frozen object.
        return PreparedRankTransfer(payload, physical_lease_id=lease.lease_id)

    def begin_io(self, command, prepared):
        if not self.provider.context.authority.begin_io(
            prepared.physical_lease_id, str(command.attempt)
        ):
            raise RuntimeError("target workset could not enter I/O")

    def finish_io(self, command, prepared, completion):
        # PreparedRankTransfer is frozen.  Recover the retained local state by
        # lease id instead of mutating the transport payload.
        success = completion.state.value == "succeeded"
        if not self.provider.context.authority.complete_io(
            prepared.physical_lease_id, str(command.attempt), success=success
        ):
            raise RuntimeError("stale target I/O completion")

    def _local(self, prepared: PreparedRankTransfer) -> _TargetPrepared:
        return self.provider._target_prepared[prepared.physical_lease_id]

    def prepare_handoff(self, command, prepared, completion):
        local = self._local(prepared)
        self.provider._bind_target(local)

    def commit(self, command, prepared, completion):
        local = self._local(prepared)
        self.provider._publish_target(local)
        if self.direct:
            prepared.transfer_payload.cleanup()
        self.provider._target_prepared.pop(local.lease.lease_id, None)

    def abort(self, command, prepared, completion):
        lease_id = prepared.physical_lease_id
        self.provider.context.authority.request_release(lease_id)
        self.provider.context.authority.commit_release(
            lease_id, reason="target_attempt_aborted"
        )
        if self.direct:
            prepared.transfer_payload.cleanup()
        self.provider._target_prepared.pop(lease_id, None)


class _InitialPHandler(RankPathHandler):
    def __init__(self, provider: "AgenticDefaultPhysicalProvider") -> None:
        self.provider = provider

    def prepare(self, command):
        if self.provider.role != "prefill":
            return PreparedRankTransfer(None, requires_io=False)
        lease, req = self.provider._reserve_target(command)
        self.provider._target_prepared[lease.lease_id] = _TargetPrepared(
            lease, req, None
        )
        return PreparedRankTransfer(
            None, physical_lease_id=lease.lease_id, requires_io=False
        )

    def begin_io(self, command, prepared):
        return None

    def finish_io(self, command, prepared, completion):
        return None

    def prepare_handoff(self, command, prepared, completion):
        if prepared.physical_lease_id is not None:
            self.provider._bind_target(
                self.provider._target_prepared[prepared.physical_lease_id]
            )

    def commit(self, command, prepared, completion):
        if prepared.physical_lease_id is not None:
            local = self.provider._target_prepared.pop(
                prepared.physical_lease_id
            )
            self.provider._publish_target(local)

    def abort(self, command, prepared, completion):
        if prepared.physical_lease_id is not None:
            lease_id = prepared.physical_lease_id
            self.provider.context.authority.request_release(lease_id)
            self.provider.context.authority.commit_release(
                lease_id, reason="initial_admission_aborted"
            )
            self.provider._target_prepared.pop(lease_id, None)


class AgenticDefaultPhysicalProvider:
    """Concrete scheduler/Radix/NIXL/Host provider for one TP rank."""

    def __init__(self, scheduler: Any, config: Any) -> None:
        self.scheduler = scheduler
        self.config = config
        self.role = str(config.role)
        self.adapter = NativeRequestMemoryAdapter(scheduler)
        self.context = None
        self._submitter: Optional[Callable[[GroupTransferPlan], Any]] = None
        self._source: dict[GenerationKey, tuple[Any, PhysicalMemoryLease, NativeSourceSnapshot]] = {}
        self._target_prepared: dict[int, _TargetPrepared] = {}
        self._pending_candidates: dict[GenerationKey, GroupTransferPlan] = {}
        self._pending_p2d_candidates: dict[GenerationKey, GroupTransferPlan] = {}
        self._early_p2d_targets: dict[GenerationKey, Mapping[str, Any]] = {}
        self._p2d_spill_timers: dict[GenerationKey, threading.Timer] = {}
        self._initial_capacity_wait: dict[GenerationKey, GroupTransferPlan] = {}
        self._p2d_route_lock = threading.RLock()
        self._route_callback_error: Optional[BaseException] = None
        self._host_results: dict[GenerationKey, HostDescriptorSet] = {}
        self._d2p_host_entries: dict[GenerationKey, _D2PHostEntry] = {}
        self._terminal_d2p: set[GenerationKey] = set()
        self._d2p_host_lock = threading.RLock()
        self._d2p_host_evictions = 0
        self._endpoint_capacity: dict[tuple[str, str], int] = {}
        self._policy: Optional[D2PPolicyActor] = None
        self._p2d_policy: Optional[P2DPolicyActor] = None
        default_direct_lanes = max(
            1, int(os.getenv("SGLANG_AGENTIC_MULTINODE_DIRECT_LANES", "4"))
        )
        self._direct_lanes = {
            TransferPath.D2P_DIRECT: max(
                1,
                int(
                    os.getenv(
                        "SGLANG_AGENTIC_MULTINODE_D2P_DIRECT_LANES",
                        str(default_direct_lanes),
                    )
                ),
            ),
            TransferPath.P2D_DIRECT: max(
                1,
                int(
                    os.getenv(
                        "SGLANG_AGENTIC_MULTINODE_P2D_DIRECT_LANES",
                        str(default_direct_lanes),
                    )
                ),
            ),
        }
        self._host_lanes = max(
            1, int(os.getenv("SGLANG_AGENTIC_MULTINODE_HOST_LANES", "4"))
        )
        self._paths_ready = False
        self._closers: list[Any] = []
        self._route_notifier = (
            ThreadPoolExecutor(max_workers=2, thread_name_prefix="dualpd-route-ready")
            if int(self.scheduler.tp_rank) == 0
            and (
                os.getenv("SGLANG_AGENTIC_ROUTE_CALLBACK_URL", "")
                or os.getenv(
                    "SGLANG_AGENTIC_DECODE_RESERVATION_CALLBACK_URL", ""
                )
            )
            else None
        )

    def state_allocators(self, scheduler):
        return hybrid_state_allocator(scheduler)

    def default_state_slot_counts(self, scheduler):
        return tuple(0 for _ in hybrid_state_allocator(scheduler))

    def install_submitter(self, submitter):
        self._submitter = submitter
        if (
            self.role == "prefill"
            and int(self.scheduler.tp_rank) == 0
            and self.config.endpoint_group == self.config.coordinator_group
        ):
            self._policy = D2PPolicyActor(
                direct_window_seconds=float(
                    os.getenv("SGLANG_AGENTIC_MULTINODE_DIRECT_WINDOW_SECONDS", "1")
                ),
                submit=submitter,
                make_direct=self._make_d2p_direct,
                make_host_store=self._make_d2p_host_store,
                make_host_restore=self._make_d2p_host_restore,
                make_recompute=self._make_d2p_recompute,
            )
            self._p2d_policy = P2DPolicyActor(
                submit=submitter,
                make_host_store=self._make_p2d_host_store,
                make_host_restore=self._make_p2d_host_restore,
            )

    def _ensure_paths(self, context) -> None:
        if self._paths_ready:
            return
        self.context = context
        scheduler = self.scheduler
        kv_pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
        args = scheduler.server_args
        tp_rank = int(scheduler.tp_rank)
        tp_size = int(self.config.tp_size)
        source_direction = "p2d" if self.role == "prefill" else "d2p"
        target_direction = "d2p" if self.role == "prefill" else "p2d"

        from sglang.srt.disaggregation.agentic_direct_transfer import (
            create_agentic_direct_runtime,
        )

        common = dict(
            kv_pool=kv_pool,
            server_args=args,
            engine_rank=tp_rank,
            pp_rank=int(scheduler.pp_rank),
            gpu_id=int(scheduler.gpu_id),
            total_kv_heads=scheduler.model_config.get_total_num_kv_heads(),
        )
        self._direct_source_runtime = create_agentic_direct_runtime(
            role=DisaggregationMode.PREFILL,
            # The stock PD bootstrap listener already owns
            # disaggregation_bootstrap_port on a P node.  Both directions use
            # an isolated Direct manager, so its sender must listen on the
            # dedicated reverse port on every node.
            bootstrap_port=_reverse_bootstrap_port(),
            **common,
        )
        self._direct_target_runtime = create_agentic_direct_runtime(
            role=DisaggregationMode.DECODE,
            **common,
        )
        self._direct_executors = {
            TransferPath.D2P_DIRECT: NixlDirectIOExecutor(
                max_workers=self._direct_lanes[TransferPath.D2P_DIRECT]
            ),
            TransferPath.P2D_DIRECT: NixlDirectIOExecutor(
                max_workers=self._direct_lanes[TransferPath.P2D_DIRECT]
            ),
        }

        self._host_arena = SourceLocalHostArena(
            direction=HostDirection(source_direction), device_pool=kv_pool
        )
        self._source_host_worker = RemoteHostRankWorker(
            kv_pool,
            int(scheduler.token_to_kv_pool_allocator.page_size),
            tp_rank,
            tp_size,
            source_direction,
            source_arena=self._host_arena,
        )
        self._source_host_worker.prewarm_source_arena()
        self._target_host_worker = RemoteHostRankWorker(
            kv_pool,
            int(scheduler.token_to_kv_pool_allocator.page_size),
            tp_rank,
            tp_size,
            target_direction,
        )
        self._host_store_path = make_source_host_store_path(
            self._host_arena,
            self._source_host_worker,
            CudaSourceHostCopyBackend(kv_pool),
            descriptor=self._host_source_payload,
            source_hbm_release=self._host_source_release,
            max_workers=self._host_lanes,
        )
        self._host_load_executor = RemoteHostLoadExecutor(
            self._target_host_worker, max_workers=self._host_lanes
        )
        self._host_restore_source = make_host_restore_source_handler(
            self._source_host_worker,
            source_group=self.config.endpoint_group,
            release_host_snapshot=self._host_arena.release_snapshot,
        )
        self._host_evict_source = make_source_host_eviction_handler(
            self._host_arena,
            self._source_host_worker,
        )
        self._no_io = make_no_io_host_endpoint_handler()
        self._target_direct = _TargetHandler(self, direct=True)
        self._target_host = _TargetHandler(self, direct=False)
        self._initial = _InitialPHandler(self)
        self._source_direct = self._make_source_direct_handler()
        self._handlers = {
            path: _DispatchHandler(
                lambda command, path=path: self._select_handler(path, command),
                native_stream=lambda: self.scheduler.schedule_stream,
            )
            for path in TransferPath
        }
        self._executors = self._make_executors()
        self._closers.extend(
            [
                *self._direct_executors.values(),
                self._host_store_path.executor,
                self._host_load_executor,
            ]
        )
        self._paths_ready = True

    def executors(self, context):
        self._ensure_paths(context)
        return self._executors

    def handlers(self, context):
        self._ensure_paths(context)
        return self._handlers

    def lanes(self, context):
        return {
            path: (
                self._direct_lanes[path]
                if path in {TransferPath.D2P_DIRECT, TransferPath.P2D_DIRECT}
                else self._host_lanes
            )
            for path in TransferPath
        }

    def pending_capacity(self, context):
        return {path: 256 for path in TransferPath}

    def _make_executors(self):
        result = {}
        for path in TransferPath:
            if path in {TransferPath.D2P_DIRECT, TransferPath.P2D_DIRECT}:
                direct = self._direct_executors[path]
                result[path] = _DispatchExecutor(
                    lambda payload, direct=direct: direct
                )
                continue

            def select(payload, self=self):
                if isinstance(payload, SourceHostStorePayload):
                    return self._host_store_path.executor
                if isinstance(payload, RemoteHostLoadPayload):
                    return self._host_load_executor
                raise TypeError("Host queue received an unknown physical payload")

            result[path] = _DispatchExecutor(select)
        return result

    def _local_is_source(self, path: TransferPath) -> bool:
        return (
            path in {TransferPath.D2P_DIRECT, TransferPath.D2P_HOST}
            and self.role == "decode"
        ) or (
            path in {TransferPath.P2D_DIRECT, TransferPath.P2D_HOST}
            and self.role == "prefill"
        )

    def _select_handler(self, path: TransferPath, command: GroupCommand):
        values = _values(command)
        if values.get("kind") == "initial_prefill":
            return self._initial
        source = self._local_is_source(path)
        operation = _operation(command)
        if operation is TransferOperation.DIRECT:
            return self._source_direct if source else self._target_direct
        if operation is TransferOperation.HOST_STORE:
            return self._host_store_path.handler if source else self._no_io
        if operation is TransferOperation.HOST_RESTORE:
            return self._host_restore_source if source else self._target_host
        if operation is TransferOperation.HOST_EVICT:
            return self._host_evict_source if source else self._no_io
        raise ValueError("unsupported transfer operation")

    def _source_entry(self, command: GroupCommand):
        entry = self._source.get(command.key)
        if entry is None:
            raise RuntimeError("source generation has no compute-complete snapshot")
        return entry

    @staticmethod
    def _page_prefix(snapshot: NativeSourceSnapshot, token_count: int):
        if token_count <= 0 or token_count > snapshot.token_count:
            raise ValueError("invalid source snapshot prefix")
        page_count = token_count * len(snapshot.page_indices) // snapshot.token_count
        return snapshot.page_indices[:page_count]

    def _make_source_direct_handler(self):
        def prepare(command: GroupCommand):
            _req, lease, snapshot = self._source_entry(command)
            bridge = (
                self.context.p_memory_bridge
                if self.role == "prefill"
                else self.context.d_memory_bridge
            )
            ready_event = bridge.forward_fence(lease.lease_id)
            values = _values(command)
            token_count = int(values["token_count"])

            def make_shard():
                state = snapshot.state_indices or None
                return NixlDirectShard(
                    bootstrap_addr=str(values["bootstrap_addr"]),
                    room=int(values["room"]),
                    page_indices=self._page_prefix(snapshot, token_count),
                    state_indices=state,
                    prefill_dp_rank=0,
                    destination_tp_ranks=(int(self.scheduler.tp_rank),),
                    pp_rank=int(self.scheduler.pp_rank),
                )

            operation = NixlDirectOperation(
                self._direct_source_runtime,
                DirectEndpoint.SOURCE,
                None,
                ready_event=ready_event,
                shard_factory=make_shard,
            )
            return PreparedRankTransfer(
                operation, physical_lease_id=lease.lease_id
            )

        def begin(command, prepared):
            if not self.context.authority.begin_io(
                prepared.physical_lease_id, str(command.attempt)
            ):
                raise RuntimeError("source lease could not enter Direct I/O")

        def finish(command, prepared, completion):
            if not self.context.authority.complete_io(
                prepared.physical_lease_id,
                str(command.attempt),
                success=completion.state.value == "succeeded",
            ):
                raise RuntimeError("stale source Direct completion")

        def commit(command, prepared, completion):
            req, lease, _snapshot = self._source_entry(command)
            bridge = (
                self.context.p_memory_bridge
                if self.role == "prefill"
                else self.context.d_memory_bridge
            )
            if not bridge.release_after_group_fence(
                lease.lease_id, reason=f"{command.key.snapshot_id}:direct_handoff"
            ):
                raise RuntimeError("source lease was not releasable")
            self._source.pop(command.key, None)
            self.context.registry.retire_after_source_fence(command.key)
            prepared.transfer_payload.cleanup()

        def abort(command, prepared, completion):
            prepared.transfer_payload.cleanup()

        return CallbackPathHandler(
            prepare,
            begin_io=begin,
            finish_io=finish,
            commit=commit,
            abort=abort,
        )

    def _target_req(self, command: GroupCommand):
        values = _values(command)
        generation = int(values.get("target_generation", command.key.generation))
        key = GenerationKey(
            command.key.run_id, command.key.request_id, generation
        )
        record = self.context.registry.get(key)
        if record is None:
            raise RuntimeError("target request-generation has not arrived")
        return record.req, key

    def _reserve_target(self, command: GroupCommand):
        values = _values(command)
        req, target_key = self._target_req(command)
        key = RequestGenerationAttempt(
            target_key.request_id, target_key.generation, command.attempt
        )
        if self.role == "prefill":
            parent = int(values.get("parent_tokens", 0))
            prompt = int(values["prompt_tokens"])
            count = prefill_state_slot_count(
                self.scheduler, imported_parent=parent > 0
            )
            lease = self.context.p_memory_bridge.reserve_workset(
                key,
                owner=str(values.get("kind", "d2p")),
                parent_tokens=parent,
                prompt_tokens=prompt,
                state_slot_counts=(() if not self.adapter.hybrid else (count,)),
            )
        else:
            prompt = int(values["prompt_tokens"])
            growth = int(values["decode_growth_tokens"])
            count = decode_state_slot_count(self.scheduler)
            lease = self.context.d_memory_bridge.reserve_decode(
                key,
                owner=str(values.get("kind", "p2d")),
                prompt_tokens=prompt,
                decode_growth_tokens=growth,
                state_slot_counts=(() if not self.adapter.hybrid else (count,)),
            )
        if lease is None:
            raise MemoryError("complete target workset is unavailable")
        return lease, req

    def _direct_target_state(self, lease: PhysicalMemoryLease):
        if not lease.state_indices:
            return None
        state = lease.state_indices[0]
        if self.role == "prefill":
            return (int(state[0]),)
        req_pool = self.scheduler.req_to_token_pool
        if not bool(getattr(req_pool, "enable_mamba_extra_buffer", False)):
            return (int(state[0]),)
        other = req_pool.get_mamba_ping_pong_other_idx(0)
        return (int(state[0]), int(state[1 + int(other)]))

    def _direct_target_payload(self, command, lease):
        values = _values(command)
        token_count = int(values["token_count"])
        ready_event = None
        if bool(getattr(lease.parent_indices, "is_cuda", False)):
            ready_event = torch.cuda.Event()
            ready_event.record()

        def make_shard():
            raw = lease.parent_indices[:token_count].detach().cpu().numpy()
            pages = tuple(
                int(value)
                for value in kv_to_page_indices(raw, lease.page_size).tolist()
            )
            return NixlDirectShard(
                bootstrap_addr=str(values["bootstrap_addr"]),
                room=int(values["room"]),
                page_indices=pages,
                state_indices=self._direct_target_state(lease),
                prefill_dp_rank=0,
                destination_tp_ranks=(int(self.scheduler.tp_rank),),
                pp_rank=int(self.scheduler.pp_rank),
            )

        operation = NixlDirectOperation(
            self._direct_target_runtime,
            DirectEndpoint.TARGET,
            None,
            ready_event=ready_event,
            shard_factory=make_shard,
        )
        return operation

    def _host_target_payload(self, command, lease):
        values = _values(command)
        descriptors = HostDescriptorSet.from_payload(values["host_shards"])
        shard = descriptors.shards[int(self.scheduler.tp_rank)]
        state = self._direct_target_state(lease)
        ready_event = None
        if bool(getattr(lease.parent_indices, "is_cuda", False)):
            ready_event = torch.cuda.Event()
            ready_event.record()

        def device_indices():
            values = tuple(
                int(value)
                for value in lease.parent_indices[: shard.token_count]
                .detach()
                .to(device="cpu", dtype=torch.int64)
                .tolist()
            )
            if len(values) != int(shard.token_count):
                raise ValueError(
                    "Host target lease does not cover the complete snapshot"
                )
            return values

        payload = RemoteHostLoadPayload(
            shard=shard.to_dict(),
            device_indices=device_indices,
            state_indices=state,
            ready_event=ready_event,
        )
        return payload

    def _bind_target(self, local: _TargetPrepared) -> None:
        if local.lease.kind is LeaseKind.PREFILL_WORKSET:
            self.context.p_memory_bridge.bind_workset(
                local.lease,
                local.req,
                bind_parent=self.adapter.bind_prefill_parent,
                release_bound=self.adapter.release_bound,
                release_unbound=self.adapter.release_prefill_unadopted,
            )
        else:
            token = local.sampled_token_id
            if token is None:
                raise RuntimeError("P2D handoff is missing the sampled Prefill token")
            appended = False
            if not local.req.output_ids:
                local.req.output_ids.append(int(token))
                appended = True
            elif int(local.req.output_ids[-1]) != int(token):
                raise RuntimeError("P2D sampled token disagrees with the target request")
            try:
                self.context.d_memory_bridge.bind_decode(
                    local.lease,
                    local.req,
                    bind_prompt=lambda lease, req: self.adapter.bind_decode_prompt(
                        lease,
                        req,
                        mamba_checkpoint_tokens=local.mamba_checkpoint_tokens,
                    ),
                    release_bound=self.adapter.release_bound,
                    release_unbound=self.adapter.release_decode_unadopted,
                )
            except BaseException:
                if appended:
                    local.req.output_ids.pop()
                raise

    def _publish_target(self, local: _TargetPrepared) -> None:
        if local.lease.kind is LeaseKind.PREFILL_WORKSET:
            event = self.context.p_memory_bridge.publish_bound(
                local.lease, local.req
            )
        else:
            event = self.context.d_memory_bridge.publish_bound(
                local.lease, local.req
            )
        if event is None:
            raise RuntimeError("group handoff failed to publish target ready")

    def _host_source_payload(self, command: GroupCommand):
        _req, lease, snapshot = self._source_entry(command)
        context = getattr(self, "context", None)
        role = getattr(self, "role", None)
        bridge = None
        if context is not None and role in {"prefill", "decode"}:
            bridge = (
                context.p_memory_bridge
                if role == "prefill"
                else context.d_memory_bridge
            )
        ready_event = None if bridge is None else bridge.forward_fence(lease.lease_id)
        values = _values(command)
        token_count = int(values["token_count"])
        return SourceHostStorePayload(
            extent_id=0,
            source_indices=snapshot.token_indices[:token_count],
            state_indices=(
                (lambda: snapshot.state_indices or None)
                if snapshot.state_slot_count
                else None
            ),
            # The registered-extent backend needs a CPU address mirror for
            # pure copy-engine DMA.  Materialize it only after the request's
            # Forward fence, in the I/O worker, so PREPARE remains a short
            # control transaction.
            source_indices_host=lambda: tuple(
                int(value)
                for value in snapshot.token_indices[:token_count]
                .detach()
                .to(device="cpu", dtype=torch.int64)
                .tolist()
            ),
            ready_event=ready_event,
        )

    def _host_source_release(self, command, _extent):
        _req, lease, _snapshot = self._source_entry(command)
        bridge = (
            self.context.p_memory_bridge
            if self.role == "prefill"
            else self.context.d_memory_bridge
        )
        if not bridge.release_after_group_fence(
            lease.lease_id, reason=f"{command.key.snapshot_id}:host_durable"
        ):
            raise RuntimeError("Host-durable source lease was not releasable")
        self._source.pop(command.key, None)
        self.context.registry.retire_after_source_fence(command.key)

    def _base_payload(self, snapshot: NativeSourceSnapshot, *, kind: str):
        return {
            "kind": kind,
            "token_count": snapshot.token_count,
            "page_chain_hashes": list(snapshot.page_chain_hashes),
            "state_slots": snapshot.state_slot_count or 1,
        }

    def on_request(self, context, record):
        self.context = context
        metadata = getattr(record.req.sampling_params, "custom_params", None) or {}
        if self.role == "prefill" and record.parent_key is None:
            if int(self.scheduler.tp_rank) != 0:
                return None
            prompt = len(record.req.origin_input_ids)
            return GroupTransferPlan(
                key=record.key,
                path=TransferPath.P2D_DIRECT,
                operation=TransferOperation.DIRECT,
                source_owner=Owner.NONE,
                target_owner=Owner.P_GPU,
                lease_id=f"init:{record.key.snapshot_id}",
                payload={
                    "kind": "initial_prefill",
                    "target_generation": record.key.generation,
                    "parent_tokens": 0,
                    "prompt_tokens": prompt,
                    "token_count": prompt,
                },
                target_group=self.config.endpoint_group,
            )
        return None

    def request_control_intent(self, context, record):
        if self.role == "decode":
            metadata = (
                getattr(record.req.sampling_params, "custom_params", None) or {}
            )
            return (
                record.key,
                _P2D_CHILD_ARRIVED,
                {
                    "target_group": self.config.endpoint_group,
                    "target_generation": record.key.generation,
                    "decode_reservation_id": str(
                        metadata.get("agentic_decode_reservation_id", "")
                    ),
                },
            )
        if self.role != "prefill" or record.parent_key is None:
            return None
        tokens = tuple(int(value) for value in record.req.origin_input_ids)
        page_size = int(context.authority.page_size)
        complete_tokens = tokens[: len(tokens) // page_size * page_size]
        return (
            record.parent_key,
            _D2P_CHILD_ARRIVED,
            {
                "target_group": self.config.endpoint_group,
                "target_generation": record.key.generation,
                "prompt_tokens": len(tokens),
                "page_chain_hashes": list(
                    token_page_chain_hashes(complete_tokens, page_size)
                ),
            },
        )

    def plan_prefill_complete(self, context, record, item):
        snapshot = self.adapter.source_snapshot(item.req, direction="p2d")
        self._source[record.key] = (item.req, item.lease, snapshot)
        if int(self.scheduler.tp_rank) != 0:
            return None
        runtime = self._direct_source_runtime
        if not runtime.bootstrap_addr:
            raise RuntimeError("P2D Direct source has no bootstrap address")
        growth = int(
            min(
                getattr(item.req.sampling_params, "max_new_tokens", 0),
                int(os.getenv("SGLANG_AGENTIC_MULTINODE_DECODE_GROWTH_TOKENS", "512")),
            )
        )
        payload = self._base_payload(snapshot, kind="p2d")
        payload.update(
            {
                "late_bind_pending": True,
                "target_generation": record.key.generation,
                "prompt_tokens": snapshot.token_count,
                "decode_growth_tokens": growth,
                "sampled_token_id": int(item.req.output_ids[-1]),
                "bootstrap_addr": runtime.bootstrap_addr,
            }
        )
        if self.adapter.hybrid:
            payload["mamba_checkpoint_tokens"] = p2d_mamba_checkpoint_tokens(
                item.req, int(self.scheduler.page_size)
            )
        plan = GroupTransferPlan(
            key=record.key,
            path=TransferPath.P2D_DIRECT,
            operation=TransferOperation.DIRECT,
            source_owner=Owner.P_GPU,
            target_owner=Owner.D_GPU,
            lease_id=f"p2d:{record.key.snapshot_id}",
            payload=payload,
            source_group=self.config.endpoint_group,
            target_group="",
        )
        self._notify_prefill_ready(record, snapshot.token_count, growth)
        if self.config.endpoint_group == self.config.coordinator_group:
            target = self._early_p2d_targets.pop(plan.key, None)
            if target is None:
                self._hold_p2d_until_target(plan)
                return None
            bound = self._bind_p2d_target(plan, target)
            if self._p2d_policy is None:
                raise RuntimeError("global coordinator has no P2D policy actor")
            self._p2d_policy.direct_submitted(bound)
            return bound
        return plan

    def _notify_prefill_ready(
        self, record, prompt_tokens: int, decode_growth_tokens: int
    ) -> None:
        """Notify Router after compute without blocking the P controller."""

        if self._route_notifier is None:
            return
        callback = os.getenv("SGLANG_AGENTIC_ROUTE_CALLBACK_URL", "").strip()
        room = getattr(record.req, "bootstrap_room", None)
        if not callback or room is None:
            raise RuntimeError("late-bound Prefill request lacks route callback metadata")
        body = json.dumps(
            {
                "bootstrap_room": room,
                "request_id": record.key.request_id,
                "generation": record.key.generation,
                "prompt_tokens": int(prompt_tokens),
                # D endpoint capacity reports physical free pages after one
                # allocator-wide growth floor.  Late binding therefore
                # reserves only the imported prompt for this request.
                "required_tokens": int(prompt_tokens),
                "source_group": self.config.endpoint_group,
            },
            separators=(",", ":"),
        ).encode()

        def post() -> None:
            error = None
            for retry in range(8):
                try:
                    request = urllib.request.Request(
                        callback,
                        data=body,
                        headers={"content-type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(request, timeout=5.0) as response:
                        if response.status != 200:
                            raise RuntimeError(
                                "late-binding Router rejected Prefill ready: "
                                f"{response.status}"
                            )
                    return
                except BaseException as caught:
                    error = caught
                    if retry != 7:
                        time.sleep(min(1.0, 0.05 * (2**retry)))
            logger.error(
                "Prefill-ready callback failed permanently key=%s error=%r",
                record.key.snapshot_id,
                error,
            )
            with self._p2d_route_lock:
                if self._route_callback_error is None:
                    self._route_callback_error = error

        self._route_notifier.submit(post)

    def _hold_p2d_until_target(self, candidate: GroupTransferPlan) -> None:
        """Wait briefly for D selection, then spill without retaining P HBM."""

        if self._p2d_policy is None:
            raise RuntimeError("global coordinator has no P2D policy actor")
        self._p2d_policy.direct_submitted(candidate)
        delay = max(
            0.0,
            float(os.getenv("SGLANG_AGENTIC_P2D_LATE_BIND_GRACE_SECONDS", "0.5")),
        )

        def spill() -> None:
            with self._p2d_route_lock:
                self._p2d_spill_timers.pop(candidate.key, None)
            self._p2d_policy.spill_if_unbound(candidate.key)

        timer = threading.Timer(delay, spill)
        timer.daemon = True
        with self._p2d_route_lock:
            self._pending_p2d_candidates[candidate.key] = candidate
            self._p2d_spill_timers[candidate.key] = timer
        timer.start()

    def _take_p2d_candidate(
        self, key: GenerationKey
    ) -> Optional[GroupTransferPlan]:
        with self._p2d_route_lock:
            candidate = self._pending_p2d_candidates.pop(key, None)
            timer = self._p2d_spill_timers.pop(key, None)
        if timer is not None:
            timer.cancel()
        return candidate

    @staticmethod
    def _bind_p2d_target(
        candidate: GroupTransferPlan, target: Mapping[str, Any]
    ) -> GroupTransferPlan:
        target_group = str(target.get("target_group", ""))
        if not target_group:
            raise ValueError("P2D target arrival omitted target_group")
        values = dict(candidate.payload)
        values.pop("late_bind_pending", None)
        values["target_group"] = target_group
        values["target_generation"] = int(
            target.get("target_generation", candidate.key.generation)
        )
        reservation_id = str(target.get("decode_reservation_id", ""))
        if reservation_id:
            values["decode_reservation_id"] = reservation_id
        values["room"] = _room(candidate.key, TransferPath.P2D_DIRECT, target_group)
        return replace(candidate, payload=values, target_group=target_group)

    def on_materialized(self, _context, plan: GroupTransferPlan, attempt: int) -> None:
        """Acknowledge replacement of one Router shadow D reservation.

        The callback is emitted only by the fixed group coordinator after all
        target TP ranks cross their real DMA fences.  It never owns or releases
        KV and is posted by an independent notifier thread.
        """

        if plan.path not in {TransferPath.P2D_DIRECT, TransferPath.P2D_HOST}:
            return
        if plan.operation not in {
            TransferOperation.DIRECT,
            TransferOperation.HOST_RESTORE,
        }:
            return
        reservation_id = str(plan.payload.get("decode_reservation_id", ""))
        if not reservation_id:
            return
        callback = os.getenv(
            "SGLANG_AGENTIC_DECODE_RESERVATION_CALLBACK_URL", ""
        ).strip()
        if not callback:
            logger.warning(
                "Decode reservation %s materialized without Router callback",
                reservation_id,
            )
            return
        if self._route_notifier is None:
            raise RuntimeError("Decode reservation callback has no notifier")
        body = json.dumps(
            {
                "reservation_id": reservation_id,
                "target_group": plan.target_group,
                "request_id": plan.key.request_id,
                "generation": plan.key.generation,
                "attempt": int(attempt),
            },
            separators=(",", ":"),
        ).encode()

        def post() -> None:
            error = None
            for retry in range(8):
                try:
                    request = urllib.request.Request(
                        callback,
                        data=body,
                        headers={"content-type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(request, timeout=5.0) as response:
                        if response.status != 200:
                            raise RuntimeError(
                                "Decode reservation callback rejected: "
                                f"{response.status}"
                            )
                    return
                except BaseException as caught:
                    error = caught
                    if retry != 7:
                        time.sleep(min(1.0, 0.05 * (2**retry)))
            logger.error(
                "Decode reservation callback failed permanently id=%s error=%r",
                reservation_id,
                error,
            )

        self._route_notifier.submit(post)

    def plan_decode_complete(self, context, record, item):
        snapshot = self.adapter.source_snapshot(item.req, direction="d2p")
        self._source[record.key] = (item.req, item.lease, snapshot)
        if int(self.scheduler.tp_rank) != 0:
            return None
        runtime = self._direct_source_runtime
        if not runtime.bootstrap_addr:
            raise RuntimeError("D2P Direct source has no bootstrap address")
        payload = self._base_payload(snapshot, kind="d2p_candidate")
        payload.update(
            {
                "bootstrap_addr": runtime.bootstrap_addr,
                "room": _room(record.key, TransferPath.D2P_DIRECT),
            }
        )
        return GroupTransferPlan(
            key=record.key,
            path=TransferPath.D2P_DIRECT,
            operation=TransferOperation.DIRECT,
            source_owner=Owner.D_GPU,
            target_owner=Owner.P_GPU,
            lease_id=f"d2p:{record.key.snapshot_id}",
            payload=payload,
            source_group=self.config.endpoint_group,
        )

    def _child_values(self, candidate, child_record):
        values = dict(candidate.payload)
        hashes = tuple(values["page_chain_hashes"])
        if isinstance(child_record, Mapping):
            child_hashes = tuple(child_record["page_chain_hashes"])
            pages = 0
            for parent_hash, child_hash in zip(hashes, child_hashes):
                if parent_hash != child_hash:
                    break
                pages += 1
            common = min(
                int(values["token_count"]),
                pages * int(self.context.authority.page_size),
            )
            prompt_tokens = int(child_record["prompt_tokens"])
            target_generation = int(child_record["target_generation"])
        else:
            req = child_record.req
            common = common_page_prefix_tokens(
                hashes,
                req.origin_input_ids,
                int(self.context.authority.page_size),
            )
            prompt_tokens = len(req.origin_input_ids)
            target_generation = child_record.key.generation
        if self.adapter.hybrid and common != int(values["token_count"]):
            # The transferred Mamba checkpoint is valid only at the frozen
            # pd_mamba boundary.  Never pair it with a shorter Attention prefix.
            raise RuntimeError("child diverges before the frozen Mamba checkpoint")
        if common <= 0:
            raise RuntimeError("D parent and P child share no complete KV page")
        values.update(
            {
                "kind": "d2p",
                "target_generation": target_generation,
                "parent_tokens": common,
                "prompt_tokens": prompt_tokens,
                "token_count": common,
            }
        )
        return values

    def _make_d2p_direct(self, candidate, child_record):
        timeout = max(
            0.0,
            float(
                os.getenv(
                    "SGLANG_AGENTIC_MULTINODE_DIRECT_ADMISSION_SECONDS", "1"
                )
            ),
        )
        target_group = (
            str(child_record.get("target_group", ""))
            if isinstance(child_record, Mapping)
            else self.config.endpoint_group
        )
        values = self._child_values(candidate, child_record)
        values["room"] = _room(
            candidate.key, TransferPath.D2P_DIRECT, target_group
        )
        return GroupTransferPlan(
            key=candidate.key,
            path=TransferPath.D2P_DIRECT,
            operation=TransferOperation.DIRECT,
            source_owner=Owner.D_GPU,
            target_owner=Owner.P_GPU,
            lease_id=candidate.lease_id,
            payload=values,
            source_group=candidate.source_group,
            target_group=target_group,
            admission_deadline=(time.monotonic() + timeout if timeout else 0.0),
        )

    def _make_d2p_host_store(self, candidate):
        values = dict(candidate.payload)
        values["kind"] = "d2p_host_store"
        return GroupTransferPlan(
            key=candidate.key,
            path=TransferPath.D2P_HOST,
            operation=TransferOperation.HOST_STORE,
            source_owner=Owner.D_GPU,
            target_owner=Owner.D_HOST,
            lease_id=f"d2h:{candidate.key.snapshot_id}",
            payload=values,
            source_group=candidate.source_group,
        )

    def _make_d2p_host_restore(self, candidate, child_record, descriptors):
        values = self._child_values(candidate, child_record)
        values["kind"] = "d2p_host_restore"
        values["target_group"] = (
            str(child_record.get("target_group", ""))
            if isinstance(child_record, Mapping)
            else self.config.endpoint_group
        )
        return build_host_restore_plan(
            self._make_d2p_host_store(candidate),
            descriptors,
            lease_id=f"h2p:{candidate.key.snapshot_id}",
            target_owner=Owner.P_GPU,
            target_transfer=values,
        )

    def _make_d2p_recompute(self, parent_key, child_record):
        if not isinstance(child_record, Mapping):
            raise TypeError("multi-node recompute requires a wire child descriptor")
        target_group = str(child_record.get("target_group", ""))
        if not target_group:
            raise ValueError("recompute child omitted target_group")
        generation = int(child_record["target_generation"])
        prompt_tokens = int(child_record["prompt_tokens"])
        child_key = GenerationKey(
            parent_key.run_id, parent_key.request_id, generation
        )
        return GroupTransferPlan(
            key=child_key,
            path=TransferPath.P2D_DIRECT,
            operation=TransferOperation.DIRECT,
            source_owner=Owner.NONE,
            target_owner=Owner.P_GPU,
            lease_id=f"recompute:{child_key.snapshot_id}",
            payload={
                "kind": "initial_prefill",
                "recompute_required": True,
                "evicted_parent_generation": parent_key.generation,
                "target_generation": generation,
                "parent_tokens": 0,
                "prompt_tokens": prompt_tokens,
                "token_count": prompt_tokens,
            },
            target_group=target_group,
        )

    def _make_d2p_host_evict(
        self, entry: _D2PHostEntry, *, reason: str = "d2p_host_high_watermark"
    ):
        return GroupTransferPlan(
            key=entry.key,
            path=TransferPath.D2P_HOST,
            operation=TransferOperation.HOST_EVICT,
            source_owner=Owner.D_HOST,
            target_owner=Owner.NONE,
            lease_id=f"evict:{entry.key.snapshot_id}",
            payload={
                "kind": "d2p_host_evict",
                "token_count": entry.token_count,
                "reason": str(reason),
            },
            source_group=entry.source_group,
        )

    def _track_d2p_host(
        self,
        plan: GroupTransferPlan,
        descriptors: HostDescriptorSet,
    ) -> None:
        entry = _D2PHostEntry(
            key=plan.key,
            source_group=plan.source_group,
            token_count=int(descriptors.shards[0].token_count),
            rank_bytes=tuple(int(shard.byte_size) for shard in descriptors.shards),
            durable_at=time.monotonic(),
        )
        with self._d2p_host_lock:
            old = self._d2p_host_entries.get(plan.key)
            if old is not None and old != entry:
                raise RuntimeError("D2P Host accounting changed for one generation")
            self._d2p_host_entries[plan.key] = entry
            terminal = plan.key in self._terminal_d2p
        if terminal:
            self._submit_terminal_d2p_eviction(plan.key)
        self._schedule_d2p_host_evictions(entry.source_group)

    def application_final(self, _context, key: GenerationKey) -> None:
        """Release a parent generation that will never receive a child."""

        with self._d2p_host_lock:
            self._terminal_d2p.add(key)
        if self._submit_terminal_d2p_eviction(key):
            return
        # The Host eviction may already have committed before the application
        # terminal edge arrived. Retire that detached EVICTED tombstone
        # atomically in the policy actor. Active D/Host/Direct states return
        # False and retain the terminal edge until their physical fence closes.
        retire = getattr(self._policy, "application_final", None)
        if callable(retire) and retire(key):
            self._forget_terminal_d2p(key)

    def _submit_terminal_d2p_eviction(self, key: GenerationKey) -> bool:
        if self._policy is None or getattr(self, "_submitter", None) is None:
            return False
        with self._d2p_host_lock:
            entry = self._d2p_host_entries.get(key)
            if entry is None or entry.evicting:
                return False
            if not self._policy.begin_eviction(key):
                return False
            entry.evicting = True
        try:
            self._submitter(
                self._make_d2p_host_evict(entry, reason="application_final")
            )
        except BaseException:
            with self._d2p_host_lock:
                current = self._d2p_host_entries.get(key)
                if current is not None:
                    current.evicting = False
            self._policy.eviction_rejected(key)
            raise
        return True

    def _forget_d2p_host(self, key: GenerationKey) -> None:
        lock = getattr(self, "_d2p_host_lock", None)
        if lock is None:
            return
        with lock:
            self._d2p_host_entries.pop(key, None)

    def _forget_terminal_d2p(self, key: GenerationKey) -> None:
        lock = getattr(self, "_d2p_host_lock", None)
        terminal = getattr(self, "_terminal_d2p", None)
        if lock is None or terminal is None:
            return
        with lock:
            terminal.discard(key)

    def _schedule_d2p_host_evictions(
        self,
        source_group: str,
        *,
        force: bool = False,
    ) -> int:
        """Evict complete unclaimed generations from 90% down to 75%."""

        if self._policy is None or getattr(self, "_submitter", None) is None:
            return 0
        capacity = int(
            float(os.getenv("SGLANG_AGENTIC_MULTINODE_D2P_HOST_GIB", "64"))
            * 1024**3
        )
        high = float(
            os.getenv("SGLANG_AGENTIC_MULTINODE_D2P_HOST_HIGH_WATERMARK", "0.90")
        )
        low = float(
            os.getenv("SGLANG_AGENTIC_MULTINODE_D2P_HOST_LOW_WATERMARK", "0.75")
        )
        if not (0.0 < low < high < 1.0):
            raise ValueError("D2P Host watermarks must satisfy 0 < low < high < 1")
        selected: list[_D2PHostEntry] = []
        with self._d2p_host_lock:
            entries = [
                entry
                for entry in self._d2p_host_entries.values()
                if entry.source_group == source_group
            ]
            rank_count = max((len(entry.rank_bytes) for entry in entries), default=0)
            usage = [0] * rank_count
            for entry in entries:
                for rank, byte_size in enumerate(entry.rank_bytes):
                    usage[rank] += byte_size
            if not usage or (not force and max(usage) < high * capacity):
                return 0
            target = int(low * capacity)
            candidates = sorted(
                (entry for entry in entries if not entry.evicting),
                key=lambda entry: (entry.token_count, entry.durable_at),
            )
            for entry in candidates:
                if not self._policy.begin_eviction(entry.key):
                    continue
                entry.evicting = True
                selected.append(entry)
                for rank, byte_size in enumerate(entry.rank_bytes):
                    usage[rank] -= byte_size
                if usage and max(usage) <= target:
                    break
        for entry in selected:
            try:
                self._submitter(self._make_d2p_host_evict(entry))
            except BaseException:
                with self._d2p_host_lock:
                    current = self._d2p_host_entries.get(entry.key)
                    if current is not None:
                        current.evicting = False
                self._policy.eviction_rejected(entry.key)
                raise
        if selected:
            logger.warning(
                "Agentic V2 D2P Host pressure source=%s selected=%d "
                "high=%.2f low=%.2f",
                source_group,
                len(selected),
                high,
                low,
            )
        return len(selected)

    def _make_p2d_host_store(self, direct):
        values = dict(direct.payload)
        values["kind"] = "p2d_host_store"
        return GroupTransferPlan(
            key=direct.key,
            path=TransferPath.P2D_HOST,
            operation=TransferOperation.HOST_STORE,
            source_owner=Owner.P_GPU,
            target_owner=Owner.P_HOST,
            lease_id=f"p2h:{direct.key.snapshot_id}",
            payload=values,
            source_group=direct.source_group,
        )

    def _make_p2d_host_restore(self, direct, descriptors):
        values = dict(direct.payload)
        values["kind"] = "p2d_host_restore"
        values["target_group"] = direct.target_group
        return build_host_restore_plan(
            self._make_p2d_host_store(direct),
            descriptors,
            lease_id=f"h2d:{direct.key.snapshot_id}",
            target_owner=Owner.D_GPU,
            target_transfer=values,
        )

    def progress_diagnostics(self) -> dict[str, object]:
        """Expose policy queues without advancing or mutating them."""

        with self._p2d_route_lock:
            pending_candidates = len(self._pending_p2d_candidates)
            early_targets = len(self._early_p2d_targets)
            spill_timers = len(self._p2d_spill_timers)
            initial_wait = len(self._initial_capacity_wait)
            pending_sample = tuple(
                key.snapshot_id for key in tuple(self._pending_p2d_candidates)[:4]
            )
            early_sample = tuple(
                key.snapshot_id for key in tuple(self._early_p2d_targets)[:4]
            )
        phases = (
            {} if self._p2d_policy is None else self._p2d_policy.phase_counts()
        )
        d2p_phases = (
            {} if self._policy is None else self._policy.phase_counts()
        )
        with self._d2p_host_lock:
            d2p_host_entries = len(self._d2p_host_entries)
            d2p_host_bytes = sum(
                sum(entry.rank_bytes) for entry in self._d2p_host_entries.values()
            )
        return {
            "p2d_pending_candidates": pending_candidates,
            "p2d_early_targets": early_targets,
            "p2d_spill_timers": spill_timers,
            "initial_capacity_wait": initial_wait,
            "p2d_pending_sample": pending_sample,
            "p2d_early_sample": early_sample,
            "p2d_phases": phases,
            "d2p_phases": d2p_phases,
            "d2p_host_entries": d2p_host_entries,
            "d2p_host_gib": round(d2p_host_bytes / 1024**3, 3),
            "d2p_host_evictions": self._d2p_host_evictions,
        }

    def decide_intent(self, context, intent: LinkIntent, candidate):
        if intent.kind == _D2P_CHILD_ARRIVED:
            if self._policy is None:
                raise RuntimeError("global coordinator has no D2P policy actor")
            self._policy.child_arrived(intent.key, dict(intent.payload))
            return None
        if intent.kind == _P2D_CHILD_ARRIVED:
            target = dict(intent.payload)
            pending = self._take_p2d_candidate(intent.key)
            if pending is None:
                self._early_p2d_targets[intent.key] = target
                return None
            plan = self._bind_p2d_target(pending, target)
            if self._p2d_policy is None:
                raise RuntimeError("global coordinator has no P2D policy actor")
            return self._p2d_policy.bind_target(plan)
        if candidate is None:
            raise ValueError(f"unsupported control intent {intent.kind!r}")
        if (
            candidate.path is TransferPath.P2D_DIRECT
            and candidate.operation is TransferOperation.DIRECT
            and candidate.source_owner is Owner.P_GPU
        ):
            target = self._early_p2d_targets.pop(candidate.key, None)
            if target is None:
                self._hold_p2d_until_target(candidate)
                return None
            plan = self._bind_p2d_target(candidate, target)
            if self._p2d_policy is None:
                raise RuntimeError("global coordinator has no P2D policy actor")
            self._p2d_policy.direct_submitted(plan)
            return plan
        # Only the fixed P coordinator owns D→P policy.  P→D is already a
        # complete plan produced by P0.
        if candidate.path is not TransferPath.D2P_DIRECT:
            return candidate
        if self.role != "prefill" or int(self.scheduler.tp_rank) != 0:
            return candidate
        self._pending_candidates[candidate.key] = candidate
        self._policy.offer_parent(candidate)
        # The policy actor submits Direct when child arrives or Host at the
        # deadline.  Rejecting this intent does not lose it.
        return None

    def _request_target_group(self, record, role: str) -> str:
        metadata = getattr(record.req.sampling_params, "custom_params", None) or {}
        key = f"agentic_{role}_group"
        value = str(metadata.get(key, ""))
        if not value:
            # Fixed one-P/one-D links predate explicit HTTP route metadata.
            value = str(getattr(self.config, "peer_group", ""))
        if not value:
            raise RuntimeError(f"request omitted {key}")
        return value

    def capacity_snapshot(
        self, *, remote_role: str, endpoint_group: str, available_tokens: int
    ) -> None:
        if endpoint_group:
            self._endpoint_capacity[(str(remote_role), str(endpoint_group))] = max(
                0, int(available_tokens)
            )
        if str(remote_role) != "prefill" or not endpoint_group:
            return
        remaining = max(0, int(available_tokens))
        retry = []
        with self._p2d_route_lock:
            for key, plan in tuple(self._initial_capacity_wait.items()):
                if plan.target_group != str(endpoint_group):
                    continue
                required = max(0, int(plan.payload.get("prompt_tokens", 0)))
                if required > remaining:
                    continue
                self._initial_capacity_wait.pop(key, None)
                retry.append(plan)
                remaining -= required
        for plan in retry:
            if self._submitter is None:
                raise RuntimeError("initial admission retry has no submitter")
            self._submitter(plan)

    def on_committed(self, context, plan, attempt):
        if plan.payload.get("kind") == "initial_prefill":
            with self._p2d_route_lock:
                self._initial_capacity_wait.pop(plan.key, None)
        if (
            self._policy is not None
            and plan.operation is TransferOperation.DIRECT
            and plan.path is TransferPath.D2P_DIRECT
        ):
            self._policy.committed(plan.key)
            self._forget_terminal_d2p(plan.key)
        elif (
            self._policy is not None
            and plan.operation is TransferOperation.HOST_RESTORE
            and plan.path is TransferPath.D2P_HOST
        ):
            self._policy.committed(plan.key)
            self._forget_d2p_host(plan.key)
            self._forget_terminal_d2p(plan.key)
            # This group commit follows all target READ receipts and all
            # source-shard Host releases, so it is the authoritative TP-wide
            # capacity epoch (not a rank-local allocator callback).
            self._policy.host_memory_available()
        elif (
            self._policy is not None
            and plan.operation is TransferOperation.HOST_EVICT
            and plan.path is TransferPath.D2P_HOST
        ):
            # application_final can race an already-submitted high-watermark
            # eviction.  Terminal ownership wins regardless of the reason
            # captured in that older immutable plan: this generation has no
            # child and must not be published as RECOMPUTE_REQUIRED.
            with self._d2p_host_lock:
                terminal = plan.key in self._terminal_d2p
            if terminal or plan.payload.get("reason") == "application_final":
                self._policy.cancelled(plan.key)
                self._forget_terminal_d2p(plan.key)
            else:
                self._policy.evicted(plan.key)
            self._forget_d2p_host(plan.key)
            self._d2p_host_evictions += 1
            self._policy.host_memory_available()
        elif (
            self._p2d_policy is not None
            and plan.path is TransferPath.P2D_DIRECT
            and plan.operation is TransferOperation.DIRECT
        ) or (
            self._p2d_policy is not None
            and plan.path is TransferPath.P2D_HOST
            and plan.operation is TransferOperation.HOST_RESTORE
        ):
            self._p2d_policy.committed(plan.key)
            if (
                plan.path is TransferPath.P2D_HOST
                and plan.operation is TransferOperation.HOST_RESTORE
            ):
                self._p2d_policy.host_memory_available()
        # These are transient physical-attempt inputs, not lifecycle history.
        # Retaining them would grow memory with every agent turn.
        self._pending_candidates.pop(plan.key, None)
        self._host_results.pop(plan.key, None)

    def on_committed_results(self, context, plan, attempt, results):
        if plan.operation is not TransferOperation.HOST_STORE:
            return
        if plan.path is TransferPath.D2P_HOST:
            if self._policy is None:
                return
            source_group = plan.source_group
        elif plan.path is TransferPath.P2D_HOST:
            if self._p2d_policy is None:
                return
            source_group = plan.source_group
        else:
            return
        descriptors = HostDescriptorSet.from_dma_results(
            results, source_group=source_group
        )
        self._host_results[plan.key] = descriptors
        if plan.path is TransferPath.D2P_HOST:
            self._policy.host_durable(plan.key, descriptors)
            self._track_d2p_host(plan, descriptors)
        else:
            self._p2d_policy.host_durable(plan.key, descriptors)

    def on_aborted(self, context, plan, attempt, reason):
        if str(reason).startswith("runtime_shutdown"):
            if self._policy is not None:
                self._policy.cancelled(plan.key)
            if self._p2d_policy is not None:
                self._p2d_policy.cancelled(plan.key)
            self._pending_candidates.pop(plan.key, None)
            self._host_results.pop(plan.key, None)
            return
        if plan.payload.get("kind") == "initial_prefill":
            if "complete target workset is unavailable" not in str(reason):
                raise RuntimeError(
                    f"non-capacity initial admission failure for {plan.key}: {reason}"
                )
            with self._p2d_route_lock:
                self._initial_capacity_wait[plan.key] = plan
            return
        if self._policy is not None and plan.path is TransferPath.D2P_DIRECT:
            self._policy.direct_rejected(plan.key)
        elif (
            self._policy is not None
            and plan.path is TransferPath.D2P_HOST
            and plan.operation is TransferOperation.HOST_EVICT
        ):
            with self._d2p_host_lock:
                entry = self._d2p_host_entries.get(plan.key)
                if entry is not None:
                    entry.evicting = False
            self._policy.eviction_rejected(plan.key)
        elif (
            self._policy is not None
            and plan.path is TransferPath.D2P_HOST
            and plan.operation is TransferOperation.HOST_RESTORE
        ):
            self._policy.restore_capacity_rejected(plan.key)
        elif (
            self._p2d_policy is not None
            and plan.path is TransferPath.P2D_DIRECT
            and plan.operation is TransferOperation.DIRECT
        ):
            self._p2d_policy.direct_rejected(plan.key)
        elif (
            self._p2d_policy is not None
            and plan.path is TransferPath.P2D_HOST
            and plan.operation is TransferOperation.HOST_RESTORE
        ):
            self._p2d_policy.restore_capacity_rejected(plan.key)
        elif plan.operation is TransferOperation.HOST_STORE:
            # Host-full is ordinary backpressure: the source remains the sole
            # owner and retries on an arena-release edge.  Other Host-store
            # failures are engineering/correctness failures and must stop the
            # runtime instead of being disguised as capacity pressure.
            if "source-local Host arena is full" not in str(reason):
                raise RuntimeError(
                    f"non-capacity Host-store failure for {plan.key}: {reason}"
                )
            if plan.path is TransferPath.D2P_HOST and self._policy is not None:
                self._policy.host_store_rejected(plan.key)
                self._schedule_d2p_host_evictions(
                    plan.source_group,
                    force=True,
                )
            elif plan.path is TransferPath.P2D_HOST and self._p2d_policy is not None:
                self._p2d_policy.host_store_rejected(plan.key)

    def memory_available(
        self,
        *,
        remote_role: str,
        endpoint_group: str = "",
        available_tokens: Optional[int] = None,
    ) -> None:
        """Consume a TCP capacity edge; never called from a scheduler poll."""

        if remote_role == "decode" and self._p2d_policy is not None:
            self._p2d_policy.memory_available(
                available_tokens, endpoint_group=endpoint_group
            )
        elif remote_role == "prefill" and self._policy is not None:
            self._policy.memory_available(
                available_tokens, endpoint_group=endpoint_group
            )

    def close(self):
        if self._policy is not None:
            self._policy.close()
        with self._p2d_route_lock:
            timers = tuple(self._p2d_spill_timers.values())
            self._p2d_spill_timers.clear()
            self._initial_capacity_wait.clear()
        for timer in timers:
            timer.cancel()
        if self._route_notifier is not None:
            self._route_notifier.shutdown(wait=False, cancel_futures=True)
        for value in reversed(self._closers):
            close = getattr(value, "close", None)
            if close is not None:
                close()
        if self._paths_ready:
            self._host_arena.close()

    def check_health(self) -> None:
        with self._p2d_route_lock:
            error = self._route_callback_error
        if error is not None:
            raise RuntimeError("Prefill-ready callback permanently failed") from error


def create_default_physical_provider(scheduler, config):
    return AgenticDefaultPhysicalProvider(scheduler, config)


__all__ = [
    "AgenticDefaultPhysicalProvider",
    "create_default_physical_provider",
]
