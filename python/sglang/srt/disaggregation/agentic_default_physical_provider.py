"""Default physical provider for the file-free multi-node V2 runtime.

This module composes existing NIXL Direct and source-local Host primitives; it
does not introduce another transport or allocator.  Rank zero owns path
policy, while every rank executes the immutable command for its shard.
"""

from __future__ import annotations

import hashlib
import os
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch

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
    decode_state_slot_count,
    hybrid_state_allocator,
    prefill_state_slot_count,
)
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


def _room(key: GenerationKey, path: TransferPath) -> int:
    raw = f"{key.run_id}:{key.snapshot_id}:{path.value}".encode()
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
            )
        executor = self._select(payload)
        return executor, executor.submit(attempt, notify)

    def progress(self, handle):
        executor, local = handle
        return executor.progress(local)

    def request_cancel(self, handle, notify):
        executor, local = handle
        executor.request_cancel(local, notify)

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
        self.provider._target_prepared.pop(local.lease.lease_id, None)

    def abort(self, command, prepared, completion):
        lease_id = prepared.physical_lease_id
        self.provider.context.authority.request_release(lease_id)
        self.provider.context.authority.commit_release(
            lease_id, reason="target_attempt_aborted"
        )
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
        self._host_results: dict[GenerationKey, HostDescriptorSet] = {}
        self._policy: Optional[D2PPolicyActor] = None
        self._p2d_policy: Optional[P2DPolicyActor] = None
        self._paths_ready = False
        self._closers: list[Any] = []

    def state_allocators(self, scheduler):
        return hybrid_state_allocator(scheduler)

    def default_state_slot_counts(self, scheduler):
        return tuple(0 for _ in hybrid_state_allocator(scheduler))

    def install_submitter(self, submitter):
        self._submitter = submitter
        if self.role == "prefill" and int(self.scheduler.tp_rank) == 0:
            self._policy = D2PPolicyActor(
                direct_window_seconds=float(
                    os.getenv("SGLANG_AGENTIC_MULTINODE_DIRECT_WINDOW_SECONDS", "1")
                ),
                submit=submitter,
                make_direct=self._make_d2p_direct,
                make_host_store=self._make_d2p_host_store,
                make_host_restore=self._make_d2p_host_restore,
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
            TransferPath.D2P_DIRECT: NixlDirectIOExecutor(max_workers=4),
            TransferPath.P2D_DIRECT: NixlDirectIOExecutor(max_workers=4),
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
            max_workers=4,
        )
        self._host_load_executor = RemoteHostLoadExecutor(
            self._target_host_worker, max_workers=4
        )
        self._host_restore_source = make_host_restore_source_handler(
            self._source_host_worker,
            source_group=self.config.endpoint_group,
            target_group=self.config.peer_group,
            release_host_snapshot=self._host_arena.release_snapshot,
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
        return {path: 4 for path in TransferPath}

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
            bridge.wait_forward_fence(lease.lease_id)
            values = _values(command)
            token_count = int(values["token_count"])
            state = snapshot.state_indices or None
            shard = NixlDirectShard(
                bootstrap_addr=str(values["bootstrap_addr"]),
                room=int(values["room"]),
                page_indices=self._page_prefix(snapshot, token_count),
                state_indices=state,
                prefill_dp_rank=0,
                destination_tp_ranks=(int(self.scheduler.tp_rank),),
                pp_rank=int(self.scheduler.pp_rank),
            )
            operation = NixlDirectOperation(
                self._direct_source_runtime, DirectEndpoint.SOURCE, shard
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
        raw = lease.parent_indices[:token_count].detach().cpu().numpy()
        pages = tuple(
            int(value)
            for value in kv_to_page_indices(raw, lease.page_size).tolist()
        )
        shard = NixlDirectShard(
            bootstrap_addr=str(values["bootstrap_addr"]),
            room=int(values["room"]),
            page_indices=pages,
            state_indices=self._direct_target_state(lease),
            prefill_dp_rank=0,
            destination_tp_ranks=(int(self.scheduler.tp_rank),),
            pp_rank=int(self.scheduler.pp_rank),
        )
        operation = NixlDirectOperation(
            self._direct_target_runtime, DirectEndpoint.TARGET, shard
        )
        return operation

    def _host_target_payload(self, command, lease):
        values = _values(command)
        descriptors = HostDescriptorSet.from_payload(values["host_shards"])
        shard = descriptors.shards[int(self.scheduler.tp_rank)]
        state = self._direct_target_state(lease)
        payload = RemoteHostLoadPayload(
            shard=shard.to_dict(),
            device_indices=lease.parent_indices[: shard.token_count],
            state_indices=state,
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
        if bridge is not None:
            bridge.wait_forward_fence(lease.lease_id)
        values = _values(command)
        token_count = int(values["token_count"])
        state = None
        if snapshot.state_indices:
            state = tuple(snapshot.state_indices)
        return SourceHostStorePayload(
            extent_id=0,
            source_indices=snapshot.token_indices[:token_count],
            state_indices=state,
            # The registered-extent backend can issue pure copy-engine DMA
            # only when it has the physical token addresses on CPU.  Freeze
            # this small immutable mirror during PREPARE; omitting it falls
            # back to an SM gather kernel from a background controller thread.
            source_indices_host=tuple(
                int(value)
                for value in snapshot.token_indices[:token_count]
                .detach()
                .to(device="cpu", dtype=torch.int64)
                .tolist()
            ),
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
            "state_slots": len(snapshot.state_indices) or 1,
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
            )
        if self.role == "prefill" and record.parent_key is not None:
            if self._policy is not None:
                self._policy.child_arrived(record.parent_key, record)
        return None

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
                "target_generation": record.key.generation,
                "prompt_tokens": snapshot.token_count,
                "decode_growth_tokens": growth,
                "sampled_token_id": int(item.req.output_ids[-1]),
                "bootstrap_addr": runtime.bootstrap_addr,
                "room": _room(record.key, TransferPath.P2D_DIRECT),
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
        )
        if self._p2d_policy is not None:
            self._p2d_policy.direct_submitted(plan)
        return plan

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
        )

    def _child_values(self, candidate, child_record):
        values = dict(candidate.payload)
        hashes = tuple(values["page_chain_hashes"])
        req = child_record.req
        common = common_page_prefix_tokens(
            hashes,
            req.origin_input_ids,
            int(self.context.authority.page_size),
        )
        if self.adapter.hybrid and common != int(values["token_count"]):
            # The transferred Mamba checkpoint is valid only at the frozen
            # pd_mamba boundary.  Never pair it with a shorter Attention prefix.
            raise RuntimeError("child diverges before the frozen Mamba checkpoint")
        if common <= 0:
            raise RuntimeError("D parent and P child share no complete KV page")
        values.update(
            {
                "kind": "d2p",
                "target_generation": child_record.key.generation,
                "parent_tokens": common,
                "prompt_tokens": len(req.origin_input_ids),
                "token_count": common,
            }
        )
        return values

    def _make_d2p_direct(self, candidate, child_record):
        return GroupTransferPlan(
            key=candidate.key,
            path=TransferPath.D2P_DIRECT,
            operation=TransferOperation.DIRECT,
            source_owner=Owner.D_GPU,
            target_owner=Owner.P_GPU,
            lease_id=candidate.lease_id,
            payload=self._child_values(candidate, child_record),
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
        )

    def _make_d2p_host_restore(self, candidate, child_record, descriptors):
        values = self._child_values(candidate, child_record)
        values["kind"] = "d2p_host_restore"
        return build_host_restore_plan(
            self._make_d2p_host_store(candidate),
            descriptors,
            lease_id=f"h2p:{candidate.key.snapshot_id}",
            target_owner=Owner.P_GPU,
            target_transfer=values,
        )

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
        )

    def _make_p2d_host_restore(self, direct, descriptors):
        values = dict(direct.payload)
        values["kind"] = "p2d_host_restore"
        return build_host_restore_plan(
            self._make_p2d_host_store(direct),
            descriptors,
            lease_id=f"h2d:{direct.key.snapshot_id}",
            target_owner=Owner.D_GPU,
            target_transfer=values,
        )

    def decide_intent(self, context, intent: LinkIntent, candidate):
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

    def on_committed(self, context, plan, attempt):
        if (
            self._policy is not None
            and plan.operation is TransferOperation.DIRECT
            and plan.path is TransferPath.D2P_DIRECT
        ):
            self._policy.committed(plan.key)
        elif (
            self._policy is not None
            and plan.operation is TransferOperation.HOST_RESTORE
            and plan.path is TransferPath.D2P_HOST
        ):
            self._policy.committed(plan.key)
            # This group commit follows all target READ receipts and all
            # source-shard Host releases, so it is the authoritative TP-wide
            # capacity epoch (not a rank-local allocator callback).
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
            source_group = self.config.peer_group
        elif plan.path is TransferPath.P2D_HOST:
            if self._p2d_policy is None:
                return
            source_group = self.config.endpoint_group
        else:
            return
        descriptors = HostDescriptorSet.from_dma_results(
            results, source_group=source_group
        )
        self._host_results[plan.key] = descriptors
        if plan.path is TransferPath.D2P_HOST:
            self._policy.host_durable(plan.key, descriptors)
        else:
            self._p2d_policy.host_durable(plan.key, descriptors)

    def on_aborted(self, context, plan, attempt, reason):
        if self._policy is not None and plan.path is TransferPath.D2P_DIRECT:
            self._policy.direct_rejected(plan.key)
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
            elif plan.path is TransferPath.P2D_HOST and self._p2d_policy is not None:
                self._p2d_policy.host_store_rejected(plan.key)

    def memory_available(self, *, remote_role: str) -> None:
        """Consume a TCP capacity edge; never called from a scheduler poll."""

        if remote_role == "decode" and self._p2d_policy is not None:
            self._p2d_policy.memory_available()
        elif remote_role == "prefill" and self._policy is not None:
            self._policy.memory_available()

    def close(self):
        if self._policy is not None:
            self._policy.close()
        for value in reversed(self._closers):
            close = getattr(value, "close", None)
            if close is not None:
                close()
        if self._paths_ready:
            self._host_arena.close()


def create_default_physical_provider(scheduler, config):
    return AgenticDefaultPhysicalProvider(scheduler, config)


__all__ = [
    "AgenticDefaultPhysicalProvider",
    "create_default_physical_provider",
]
