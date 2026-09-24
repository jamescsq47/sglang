from __future__ import annotations

import json
import mmap
import queue
import time

import pytest

from sglang.srt.disaggregation.agentic_group_protocol import (
    CommandKind,
    GenerationKey,
    GroupCommand,
    LinkLifecycleCoordinator,
    LinkParticipant,
    Owner,
)
from sglang.srt.disaggregation.agentic_group_transfer import (
    CallbackPathHandler,
    GroupTransferPlan,
    PreparedRankTransfer,
    RankLocalCommandExecutor,
    RankZeroLinkOrchestrator,
    RemoteHostLoadExecutor,
    TargetLeaseKind,
    TransferOperation,
)
from sglang.srt.disaggregation.agentic_host_transfer import (
    HostDescriptorSet,
    HostRestoreAdmission,
    HostStoreRestoreChainer,
    build_host_restore_plan,
    make_host_restore_source_handler,
    make_host_restore_target_handler,
    make_no_io_host_endpoint_handler,
)
from sglang.srt.disaggregation.agentic_memory_authority import (
    AgenticMemoryAuthority,
)
from sglang.srt.disaggregation.agentic_remote_host import (
    HostShard,
    ReadReceipt,
    group_ack,
    layout_fingerprint,
)
from sglang.srt.disaggregation.agentic_source_host import (
    HostDirection,
    SourceHostStorePayload,
    SourceLocalHostArena,
    make_source_host_store_path,
)
from sglang.srt.disaggregation.agentic_transfer_queues import (
    AgenticTransferQueues,
    FenceKind,
    PhysicalProgress,
    PhysicalState,
    TransferPath,
)


class ListAllocator:
    page_size = 4

    def __init__(self, size=256):
        self.free_values = list(range(size))
        self.live = set()

    def available_size(self):
        return len(self.free_values)

    def alloc(self, count):
        if count > len(self.free_values):
            return None
        values = self.free_values[:count]
        del self.free_values[:count]
        self.live.update(values)
        return values

    def free(self, values):
        values = list(values)
        assert set(values).issubset(self.live)
        self.live.difference_update(values)
        self.free_values.extend(values)


class ImmediateExecutor:
    def submit(self, attempt, notify):
        notify()
        return attempt

    def progress(self, _handle):
        return PhysicalProgress(PhysicalState.SUCCEEDED, FenceKind.DMA_COMPLETE)

    def request_cancel(self, _handle, notify):
        notify()


class DescriptorExecutor(ImmediateExecutor):
    def progress(self, handle):
        shard = HostShard.from_dict(handle.payload)
        return PhysicalProgress(
            PhysicalState.SUCCEEDED,
            FenceKind.DMA_COMPLETE,
            result={"host_shard": shard.to_dict()},
        )


class DType:
    itemsize = 2


class MHAPool:
    layer_num = 1
    head_num = 1
    head_dim = 2
    store_dtype = DType()


class FakeSnapshot:
    def __init__(self, **values):
        self.__dict__.update(values)
        self.closed = False
        self.populated = False

    def mark_populated(self):
        self.populated = True

    def close(self, *, unlink=False):
        assert not unlink
        self.closed = True


class ImmediateCopy:
    def copy(self, extent, payload, cancel):
        assert not cancel.is_set()
        assert extent.extent_id == payload.extent_id


class FakeSourceWorker:
    def __init__(self, rank):
        self.rank = rank
        self.released = []
        self.reader_id = None
        self.cancelled = []

    def claim_export(self, snapshot_id, attempt_id):
        self.reader_id = str(attempt_id)
        return self.shard

    def release_export(
        self, snapshot_id, shards, receipts, *, committed_attempt
    ):
        group_ack(shards, receipts)
        assert shards[self.rank].snapshot_id == snapshot_id
        assert committed_attempt == self.reader_id
        self.released.append((snapshot_id, committed_attempt))
        return True

    def cancel_export(self, snapshot_id, shards, receipts):
        group_ack(shards, receipts, cancelled=True)
        assert all(receipt.read_id == self.reader_id for receipt in receipts)
        self.cancelled.append(snapshot_id)
        self.reader_id = None
        return True


class FakeExportWorker(FakeSourceWorker):
    def __init__(self, rank, size):
        super().__init__(rank)
        self.size = size

    def export_snapshot(self, snapshot_id, snapshot):
        assert snapshot.populated
        self.shard = HostShard(
            snapshot_id,
            f"export-{self.rank}",
            self.rank,
            self.size,
            layout_fingerprint({"kind": "fake-mha", "tp": self.size}),
            snapshot.token_count,
            4096 + self.rank * 4096,
            snapshot.byte_size,
            "",
            f"source-{self.rank}",
        )
        return self.shard

    def discard_unclaimed_export(self, _snapshot_id):
        return True


class FakeTargetWorker:
    def __init__(self, rank):
        self.rank = rank
        self.loaded = []

    def load(
        self,
        shard_value,
        *,
        attempt_id,
        device_indices,
        cancel_check,
        **_kwargs,
    ):
        assert not cancel_check()
        shard = HostShard.from_dict(shard_value)
        assert shard.tp_rank == self.rank
        assert len(device_indices) == shard.token_count
        self.loaded.append(shard.snapshot_id)
        return ReadReceipt(
            shard.snapshot_id,
            shard.export_id,
            shard.tp_rank,
            shard.tp_size,
            shard.layout,
            shard.token_count,
            str(attempt_id),
        )


def make_queues(path, executor):
    executors = {candidate: ImmediateExecutor() for candidate in TransferPath}
    executors[path] = executor
    return AgenticTransferQueues(
        executors,
        lanes={candidate: 2 for candidate in TransferPath},
        pending_capacity={candidate: 8 for candidate in TransferPath},
    )


def descriptors(key, tp_size):
    layout = layout_fingerprint({"kind": "fake-mha", "tp": tp_size})
    return tuple(
        HostShard(
            key.snapshot_id,
            f"export-{rank}",
            rank,
            tp_size,
            layout,
            3,
            4096 + rank * 4096,
            96,
            "",
            f"source-{rank}",
        )
        for rank in range(tp_size)
    )


@pytest.mark.parametrize(
    "path,host_owner,target_owner,lease_kind,target_transfer,ready_queue",
    [
        (
            TransferPath.D2P_HOST,
            Owner.D_HOST,
            Owner.PREFILL_READY,
            TargetLeaseKind.PREFILL_WORKSET,
            {"parent_tokens": 3, "prompt_tokens": 5},
            "prefill-ready",
        ),
        (
            TransferPath.P2D_HOST,
            Owner.P_HOST,
            Owner.DECODE_READY,
            TargetLeaseKind.DECODE_RESERVATION,
            {"prompt_tokens": 3, "decode_growth_tokens": 5},
            "decode-ready",
        ),
    ],
)
def test_source_local_host_descriptor_crosses_json_and_restores_full_tp(
    path,
    host_owner,
    target_owner,
    lease_kind,
    target_transfer,
    ready_queue,
):
    tp_size = 2
    key = GenerationKey("run", f"{path.value}-request", 1)
    source = [LinkParticipant("source", "source-engine", rank) for rank in range(2)]
    target = [LinkParticipant("target", "target-engine", rank) for rank in range(2)]
    shard_values = descriptors(key, tp_size)
    store_results = {
        participant: {"host_shard": shard_values[participant.rank].to_dict()}
        for participant in source
    }
    store_results.update({participant: {} for participant in target})
    descriptor_set = HostDescriptorSet.from_dma_results(
        store_results, source_group="source-engine"
    )
    store_plan = GroupTransferPlan(
        key,
        path,
        TransferOperation.HOST_STORE,
        Owner.D_GPU if path is TransferPath.D2P_HOST else Owner.P_GPU,
        host_owner,
        "store-lease",
        {"token_count": 3},
    )
    restore_plan = build_host_restore_plan(
        store_plan,
        descriptor_set,
        lease_id="restore-lease",
        target_owner=target_owner,
        target_transfer=target_transfer,
    )

    # This is exactly the TCP JSON boundary: no HostShard, tensor or lease
    # object is retained in the command payload.
    encoded = json.dumps(restore_plan.command_payload())
    decoded = json.loads(encoded)
    assert decoded["transfer"]["host_shards"][1]["tp_rank"] == 1

    events = queue.Queue()
    participants = tuple(source + target)
    coordinator = LinkLifecycleCoordinator("run", path.value, participants)
    committed = []
    orchestrator = RankZeroLinkOrchestrator(
        coordinator,
        broadcast=lambda command: [
            events.put(("command", participant, GroupCommand.from_dict(
                json.loads(json.dumps(command.to_dict()))
            )))
            for participant in participants
        ],
        on_committed=lambda _plan, attempt: committed.append(attempt),
    )

    ranks = {}
    queues = []
    load_executors = []
    source_workers = []
    released_extents = []
    target_workers = []
    authorities = []
    no_io = make_no_io_host_endpoint_handler()
    for participant in source:
        worker = FakeSourceWorker(participant.rank)
        worker.shard = shard_values[participant.rank]
        source_workers.append(worker)
        transfer_queues = make_queues(path, ImmediateExecutor())
        queues.append(transfer_queues)
        source_handler = make_host_restore_source_handler(
            worker,
            source_group="source-engine",
            target_group="target-engine",
            release_host_snapshot=lambda snapshot_id, rank=participant.rank: (
                released_extents.append((rank, snapshot_id)) or True
            ),
        )
        handlers = {candidate: no_io for candidate in TransferPath}
        handlers[path] = source_handler
        ranks[participant] = RankLocalCommandExecutor(
            participant=participant,
            queues=transfer_queues,
            handlers=handlers,
            emit_ack=lambda ack: events.put(("ack", ack)),
            report_failure=lambda failure: events.put(("failure", failure)),
        )
    for participant in target:
        worker = FakeTargetWorker(participant.rank)
        target_workers.append(worker)
        load_executor = RemoteHostLoadExecutor(worker, max_workers=1)
        load_executors.append(load_executor)
        transfer_queues = make_queues(path, load_executor)
        queues.append(transfer_queues)
        authority = AgenticMemoryAuthority(ListAllocator())
        authorities.append(authority)
        target_handler = make_host_restore_target_handler(
            authority,
            rank=participant.rank,
            lease_kind=lease_kind,
            owner=path.value,
            ready_queue=ready_queue,
        )
        handlers = {candidate: no_io for candidate in TransferPath}
        handlers[path] = target_handler
        ranks[participant] = RankLocalCommandExecutor(
            participant=participant,
            queues=transfer_queues,
            handlers=handlers,
            emit_ack=lambda ack: events.put(("ack", ack)),
            report_failure=lambda failure: events.put(("failure", failure)),
        )

    orchestrator.begin(restore_plan)
    deadline = time.monotonic() + 3
    while not committed and time.monotonic() < deadline:
        kind, *values = events.get(timeout=1)
        if kind == "command":
            participant, command = values
            ranks[participant].handle(command)
            if (
                command.kind is CommandKind.ISSUE_ACTIVATION_TICKET
                and command.target_owner in {
                    Owner.P_GPU,
                    Owner.PREFILL_READY,
                    Owner.D_GPU,
                    Owner.DECODE_READY,
                }
                and participant == target[-1]
            ):
                for target_participant in target:
                    ranks[target_participant].activate_staged(
                        command.key, command.attempt, command.lease_id
                    )
                    ranks[target_participant].confirm_scheduler_adopted(
                        command.key, command.attempt, command.lease_id
                    )
        elif kind == "ack":
            orchestrator.on_ack(values[0])
        else:
            raise AssertionError(values[0])

    assert committed == [1]
    assert all(worker.loaded == [key.snapshot_id] for worker in target_workers)
    assert all(worker.released == [(key.snapshot_id, "1")] for worker in source_workers)
    assert sorted(released_extents) == [
        (rank, key.snapshot_id) for rank in range(tp_size)
    ]
    assert all(len(authority.take_ready(ready_queue, timeout=0.1)) == 1 for authority in authorities)
    assert all(rank.wait_idle(1) for rank in ranks.values())
    for transfer_queues in queues:
        transfer_queues.close()
    for executor in load_executors:
        executor.close()


def test_hybrid_restore_payload_carries_complete_parent_and_reserved_state():
    key = GenerationKey("run", "hybrid", 2)
    shard = HostShard(
        key.snapshot_id,
        "hybrid-export",
        0,
        1,
        layout_fingerprint({"kind": "qwen35-attention-gdn-v1"}),
        3,
        4096,
        8192,
        "",
        "source",
    )
    authority = AgenticMemoryAuthority(
        ListAllocator(), state_allocators=(ListAllocator(16),)
    )
    handler = make_host_restore_target_handler(
        authority,
        rank=0,
        lease_kind=TargetLeaseKind.PREFILL_WORKSET,
        owner="hybrid-d2p",
        ready_queue="prefill-ready",
    )
    command = GroupCommand(
        key,
        1,
        1,
        CommandKind.PREPARE,
        Owner.D_HOST,
        Owner.PREFILL_READY,
        "hybrid-lease",
        {
            "agentic_data_plane": {
                "version": 1,
                "path": TransferPath.D2P_HOST.value,
                "operation": TransferOperation.HOST_RESTORE.value,
            },
            "transfer": {
                "parent_tokens": 3,
                "prompt_tokens": 5,
                "state_slot_counts": [2],
                "host_shards": [shard.to_dict()],
            },
        },
    )
    prepared = handler.prepare(command)
    assert len(prepared.transfer_payload.device_indices) == shard.token_count
    assert len(prepared.transfer_payload.state_indices) == 2
    handler.abort(command, prepared, None)
    assert authority.active_lease_count() == 0


def test_host_restore_abort_unclaims_only_with_complete_target_drain_barrier():
    key = GenerationKey("run", "cancel-host-restore", 0)
    shard_values = descriptors(key, 2)
    worker = FakeSourceWorker(0)
    worker.shard = shard_values[0]
    released = []
    handler = make_host_restore_source_handler(
        worker,
        source_group="d",
        target_group="p",
        release_host_snapshot=lambda snapshot_id: released.append(snapshot_id),
    )
    prepare = GroupCommand(
        key,
        1,
        1,
        CommandKind.PREPARE,
        Owner.D_HOST,
        Owner.PREFILL_READY,
        "restore",
        {
            "agentic_data_plane": {
                "version": 1,
                "path": TransferPath.D2P_HOST.value,
                "operation": TransferOperation.HOST_RESTORE.value,
            },
            "transfer": {
                "parent_tokens": 3,
                "prompt_tokens": 5,
                "host_shards": [shard.to_dict() for shard in shard_values],
            },
        },
    )
    prepared = handler.prepare(prepare)
    assert worker.reader_id == "1"
    drained = ReadReceipt(
        shard_values[1].snapshot_id,
        shard_values[1].export_id,
        1,
        2,
        shard_values[1].layout,
        shard_values[1].token_count,
        "1",
        "drained",
    )
    finalize = GroupCommand(
        key,
        1,
        4,
        CommandKind.ABORT_FINALIZE,
        Owner.D_HOST,
        Owner.PREFILL_READY,
        "restore",
        {
            "rank_results": [
                {
                    "participant": {
                        "role": "target",
                        "endpoint_group": "p",
                        "rank": 0,
                    },
                    "result": {"fence": "not_posted"},
                },
                {
                    "participant": {
                        "role": "target",
                        "endpoint_group": "p",
                        "rank": 1,
                    },
                    "result": {
                        "fence": "cancel_drained",
                        "read_receipt": drained.to_dict(),
                    },
                },
            ]
        },
    )
    handler.abort(finalize, prepared, None)
    assert worker.cancelled == [key.snapshot_id]
    assert worker.reader_id is None
    assert released == []


@pytest.mark.parametrize(
    "path,direction,source_owner,host_owner,target_owner,lease_kind,target_transfer,ready_queue",
    [
        (
            TransferPath.D2P_HOST,
            HostDirection.D2P,
            Owner.D_GPU,
            Owner.D_HOST,
            Owner.PREFILL_READY,
            TargetLeaseKind.PREFILL_WORKSET,
            {"parent_tokens": 3, "prompt_tokens": 5},
            "prefill-ready",
        ),
        (
            TransferPath.P2D_HOST,
            HostDirection.P2D,
            Owner.P_GPU,
            Owner.P_HOST,
            Owner.DECODE_READY,
            TargetLeaseKind.DECODE_RESERVATION,
            {"prompt_tokens": 3, "decode_growth_tokens": 5},
            "decode-ready",
        ),
    ],
)
def test_host_store_results_automatically_drive_remote_restore_attempt(
    path,
    direction,
    source_owner,
    host_owner,
    target_owner,
    lease_kind,
    target_transfer,
    ready_queue,
):
    key = GenerationKey("run", f"{path.value}-store-then-restore", 0)
    tp_size = 2
    source_group = "d" if direction is HostDirection.D2P else "p"
    target_group = "p" if direction is HostDirection.D2P else "d"
    source = [
        LinkParticipant("source", source_group, rank) for rank in range(tp_size)
    ]
    target = [
        LinkParticipant("target", target_group, rank) for rank in range(tp_size)
    ]
    participants = tuple(source + target)
    store_plan = GroupTransferPlan(
        key,
        path,
        TransferOperation.HOST_STORE,
        source_owner,
        host_owner,
        "store",
        {"token_count": 3},
    )

    events = queue.Queue()
    ranks = {}
    queues = []
    load_executors = []
    source_workers = []
    source_arenas = []
    store_executors = []
    authorities = []
    source_hbm_released = []
    host_extents_released = []
    final = []
    no_io = make_no_io_host_endpoint_handler()

    coordinator = LinkLifecycleCoordinator("run", "host-chain", participants)
    orchestrator = None
    chainer = None

    def committed(plan, _attempt, results):
        if plan.operation is TransferOperation.HOST_STORE:
            restore = chainer.on_store_committed(plan, _attempt, results)
            assert restore is not None
            orchestrator.begin(restore)
        else:
            final.append(plan)

    orchestrator = RankZeroLinkOrchestrator(
        coordinator,
        broadcast=lambda command: [
            events.put(
                (
                    "command",
                    participant,
                    GroupCommand.from_dict(
                        json.loads(json.dumps(command.to_dict()))
                    ),
                )
            )
            for participant in participants
        ],
        on_committed_results=committed,
    )
    chainer = HostStoreRestoreChainer(
        source_group=source_group,
        admit_restore=lambda _plan: HostRestoreAdmission(
            "restore", target_owner, target_transfer
        ),
    )

    for participant in source:
        worker = FakeExportWorker(participant.rank, tp_size)
        source_workers.append(worker)
        arena = SourceLocalHostArena(
            direction=direction,
            device_pool=MHAPool(),
            capacity_bytes=2 * mmap.ALLOCATIONGRANULARITY,
            snapshot_factory=lambda **values: FakeSnapshot(**values),
        )
        source_arenas.append(arena)
        bundle = make_source_host_store_path(
            arena,
            worker,
            ImmediateCopy(),
            descriptor=lambda _command: SourceHostStorePayload(
                0, tuple(range(3))
            ),
            source_hbm_release=lambda command, _extent, rank=participant.rank: (
                source_hbm_released.append((rank, command.key.snapshot_id))
            ),
        )
        store_executors.append(bundle.executor)
        transfer_queues = make_queues(path, bundle.executor)
        queues.append(transfer_queues)
        restore_handler = make_host_restore_source_handler(
            worker,
            source_group=source_group,
            target_group=target_group,
            release_host_snapshot=lambda snapshot_id, rank=participant.rank, arena=arena: (
                host_extents_released.append((rank, snapshot_id))
                or arena.release_snapshot(snapshot_id)
            ),
        )
        handlers = {candidate: no_io for candidate in TransferPath}
        handlers[(path, TransferOperation.HOST_STORE)] = bundle.handler
        handlers[(path, TransferOperation.HOST_RESTORE)] = restore_handler
        ranks[participant] = RankLocalCommandExecutor(
            participant=participant,
            queues=transfer_queues,
            handlers=handlers,
            emit_ack=lambda ack: events.put(("ack", ack)),
            report_failure=lambda failure: events.put(("failure", failure)),
        )

    for participant in target:
        worker = FakeTargetWorker(participant.rank)
        load_executor = RemoteHostLoadExecutor(worker, max_workers=1)
        load_executors.append(load_executor)
        transfer_queues = make_queues(path, load_executor)
        queues.append(transfer_queues)
        authority = AgenticMemoryAuthority(ListAllocator())
        authorities.append(authority)
        restore_handler = make_host_restore_target_handler(
            authority,
            rank=participant.rank,
            lease_kind=lease_kind,
            owner=path.value,
            ready_queue=ready_queue,
        )
        handlers = {candidate: no_io for candidate in TransferPath}
        handlers[(path, TransferOperation.HOST_STORE)] = no_io
        handlers[(path, TransferOperation.HOST_RESTORE)] = restore_handler
        ranks[participant] = RankLocalCommandExecutor(
            participant=participant,
            queues=transfer_queues,
            handlers=handlers,
            emit_ack=lambda ack: events.put(("ack", ack)),
            report_failure=lambda failure: events.put(("failure", failure)),
        )

    orchestrator.begin(store_plan)
    deadline = time.monotonic() + 3
    while not final and time.monotonic() < deadline:
        kind, *values = events.get(timeout=1)
        if kind == "command":
            participant, command = values
            ranks[participant].handle(command)
            if (
                command.kind is CommandKind.ISSUE_ACTIVATION_TICKET
                and command.target_owner in {
                    Owner.P_GPU,
                    Owner.PREFILL_READY,
                    Owner.D_GPU,
                    Owner.DECODE_READY,
                }
                and participant == target[-1]
            ):
                for target_participant in target:
                    ranks[target_participant].activate_staged(
                        command.key, command.attempt, command.lease_id
                    )
                    ranks[target_participant].confirm_scheduler_adopted(
                        command.key, command.attempt, command.lease_id
                    )
        elif kind == "ack":
            orchestrator.on_ack(values[0])
        else:
            raise AssertionError(values[0])

    assert len(final) == 1
    assert final[0].operation is TransferOperation.HOST_RESTORE
    expected = [(rank, key.snapshot_id) for rank in range(tp_size)]
    assert sorted(source_hbm_released) == expected
    assert sorted(host_extents_released) == expected
    assert all(worker.released == [(key.snapshot_id, "2")] for worker in source_workers)
    assert all(
        len(authority.take_ready(ready_queue, timeout=0.1)) == 1
        for authority in authorities
    )
    for transfer_queues in queues:
        transfer_queues.close()
    for executor in load_executors:
        executor.close()
    for executor in store_executors:
        executor.close()
    for arena in source_arenas:
        assert arena.used_bytes == 0
        arena.close()
