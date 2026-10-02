"""JSON control descriptors for the two source-local Host transfer paths.

The physical bytes stay in the source node's memfd arena and move through a
NIXL READ.  Only :class:`HostShard` and :class:`ReadReceipt` dictionaries cross
the TCP control link.  This module contains no routing, filesystem discovery,
or scheduler work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from sglang.srt.disaggregation.agentic_group_protocol import (
    GroupCommand,
    LinkParticipant,
    Owner,
)
from sglang.srt.disaggregation.agentic_group_transfer import (
    AuthorityPathHandler,
    CallbackPathHandler,
    GroupTransferPlan,
    PreparedRankTransfer,
    RemoteHostLoadPayload,
    TargetLeaseKind,
    TransferOperation,
)
from sglang.srt.disaggregation.agentic_memory_authority import (
    AgenticMemoryAuthority,
    PhysicalMemoryLease,
)
from sglang.srt.disaggregation.agentic_remote_host import HostShard, ReadReceipt
from sglang.srt.disaggregation.agentic_transfer_queues import TransferPath


@dataclass(frozen=True, slots=True)
class HostDescriptorSet:
    """One complete TP generation's immutable source-Host descriptors."""

    shards: tuple[HostShard, ...]

    def __post_init__(self) -> None:
        if not self.shards:
            raise ValueError("Host descriptor set is empty")
        ordered = tuple(sorted(self.shards, key=lambda shard: shard.tp_rank))
        for shard in ordered:
            shard.validate()
        first = ordered[0]
        expected = (
            first.snapshot_id,
            first.tp_size,
            first.layout,
            first.token_count,
        )
        if tuple(shard.tp_rank for shard in ordered) != tuple(range(first.tp_size)):
            raise ValueError("Host descriptor set does not cover every TP rank")
        if any(
            (shard.snapshot_id, shard.tp_size, shard.layout, shard.token_count)
            != expected
            for shard in ordered
        ):
            raise ValueError("Host descriptor set mixes snapshots or layouts")
        object.__setattr__(self, "shards", ordered)

    def to_payload(self) -> list[dict[str, Any]]:
        return [shard.to_dict() for shard in self.shards]

    @classmethod
    def from_payload(cls, value: Any) -> "HostDescriptorSet":
        if not isinstance(value, list):
            raise ValueError("host_shards must be a JSON list")
        return cls(tuple(HostShard.from_dict(item) for item in value))

    @classmethod
    def from_dma_results(
        cls,
        results: Mapping[LinkParticipant, Mapping[str, Any]],
        *,
        source_group: str,
    ) -> "HostDescriptorSet":
        shards = []
        for participant, result in results.items():
            if participant.endpoint_group != source_group:
                continue
            value = result.get("host_shard")
            if value is None:
                raise ValueError(
                    f"source rank {participant.rank} omitted its Host descriptor"
                )
            shard = HostShard.from_dict(value)
            if shard.tp_rank != participant.rank:
                raise ValueError("Host descriptor rank does not match ACK identity")
            shards.append(shard)
        return cls(tuple(shards))


@dataclass(frozen=True, slots=True)
class HostRestoreAdmission:
    """Routing/admission result injected by the owning controller."""

    lease_id: str
    target_owner: Owner
    target_transfer: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.lease_id:
            raise ValueError("restore lease ID is required")
        # GroupTransferPlan will deep-copy and JSON-validate this again.  Keep
        # this object policy-only: it contains no pages, tensors or workers.
        object.__setattr__(self, "target_transfer", dict(self.target_transfer))


class HostStoreRestoreChainer:
    """Purely construct the remote READ plan from a Host-store result.

    This object deliberately does not submit the returned plan.  The low-level
    commit callback runs on the P0 control thread; synchronously submitting
    there would wait on the same thread and deadlock.  A composite controller
    must first enqueue the committed-results event, call this object on its own
    controller thread, then submit the returned plan.
    """

    def __init__(
        self,
        *,
        source_group: str,
        admit_restore: Callable[[GroupTransferPlan], HostRestoreAdmission],
    ) -> None:
        if not source_group:
            raise ValueError("source endpoint group is required")
        self.source_group = str(source_group)
        self._admit_restore = admit_restore

    def on_store_committed(
        self,
        plan: GroupTransferPlan,
        _attempt: int,
        results: Mapping[LinkParticipant, Mapping[str, Any]],
    ) -> Optional[GroupTransferPlan]:
        if plan.operation is not TransferOperation.HOST_STORE:
            return None
        descriptors = HostDescriptorSet.from_dma_results(
            results, source_group=self.source_group
        )
        admission = self._admit_restore(plan)
        if not isinstance(admission, HostRestoreAdmission):
            raise TypeError("restore admission returned an invalid result")
        return build_host_restore_plan(
            plan,
            descriptors,
            lease_id=admission.lease_id,
            target_owner=admission.target_owner,
            target_transfer=admission.target_transfer,
        )


def build_host_restore_plan(
    store_plan: GroupTransferPlan,
    descriptors: HostDescriptorSet,
    *,
    lease_id: str,
    target_owner: Owner,
    target_transfer: Mapping[str, Any],
) -> GroupTransferPlan:
    """Create the second Host leg from a committed source-store result."""

    if store_plan.operation is not TransferOperation.HOST_STORE:
        raise ValueError("restore requires a committed Host-store plan")
    if store_plan.path not in {TransferPath.D2P_HOST, TransferPath.P2D_HOST}:
        raise ValueError("restore requires a Host path")
    if descriptors.shards[0].snapshot_id != store_plan.key.snapshot_id:
        raise ValueError("Host descriptor belongs to another generation")
    transfer = dict(target_transfer)
    transfer["host_shards"] = descriptors.to_payload()
    return GroupTransferPlan(
        key=store_plan.key,
        path=store_plan.path,
        operation=TransferOperation.HOST_RESTORE,
        source_owner=store_plan.target_owner,
        target_owner=target_owner,
        lease_id=lease_id,
        payload=transfer,
        source_group=store_plan.source_group,
        target_group=str(transfer.get("target_group") or store_plan.target_group),
    )


def _transfer_values(command: GroupCommand) -> Mapping[str, Any]:
    value = command.payload.get("transfer", {})
    if not isinstance(value, Mapping):
        raise ValueError("transfer payload must be a mapping")
    return value


def _rank_results(command: GroupCommand) -> tuple[tuple[LinkParticipant, Mapping], ...]:
    values = command.payload.get("rank_results")
    if not isinstance(values, list):
        raise ValueError("RELEASE lacks endpoint-qualified rank results")
    parsed = []
    for value in values:
        identity = value.get("participant", {})
        result = value.get("result", {})
        if not isinstance(identity, Mapping) or not isinstance(result, Mapping):
            raise ValueError("invalid rank result")
        parsed.append(
            (
                LinkParticipant(
                    str(identity["role"]),
                    str(identity["endpoint_group"]),
                    int(identity["rank"]),
                ),
                result,
            )
        )
    return tuple(parsed)


def _host_descriptors(command: GroupCommand) -> HostDescriptorSet:
    return HostDescriptorSet.from_payload(_transfer_values(command)["host_shards"])


def make_host_restore_target_handler(
    authority: AgenticMemoryAuthority,
    *,
    rank: int,
    lease_kind: TargetLeaseKind,
    owner: str,
    ready_queue: str,
) -> AuthorityPathHandler:
    """Reserve a complete workset and READ this rank's Host shard into it."""

    rank = int(rank)

    def payload_builder(
        command: GroupCommand, lease: Optional[PhysicalMemoryLease]
    ) -> RemoteHostLoadPayload:
        if lease is None:
            raise RuntimeError("Host restore requires a destination lease")
        descriptors = _host_descriptors(command)
        if not 0 <= rank < len(descriptors.shards):
            raise ValueError("target rank is outside the Host descriptor set")
        shard = descriptors.shards[rank]
        if shard.tp_rank != rank:
            raise ValueError("target rank received another rank's Host shard")
        device_indices = lease.parent_indices[: shard.token_count]
        state_indices = None
        if lease.state_indices:
            if len(lease.state_indices) != 1:
                raise ValueError("hybrid Host restore expects one state allocator")
            state_indices = lease.state_indices[0]
        return RemoteHostLoadPayload(
            shard=shard.to_dict(),
            device_indices=device_indices,
            state_indices=state_indices,
        )

    return AuthorityPathHandler(
        authority,
        lease_kind=lease_kind,
        owner=owner,
        payload_builder=payload_builder,
        ready_queue=ready_queue,
    )


@dataclass(frozen=True, slots=True)
class _HostRestoreSourceState:
    descriptors: HostDescriptorSet
    target_group: str


def make_host_restore_source_handler(
    worker,
    *,
    source_group: str,
    target_group: str = "",
    release_host_snapshot: Callable[[str], bool],
) -> CallbackPathHandler:
    """Join restore without I/O and free Host only after every target fence."""

    def prepare(command: GroupCommand) -> PreparedRankTransfer:
        descriptors = _host_descriptors(command)
        selected_target = str(
            command.payload.get("agentic_data_plane", {}).get("target_group")
            or target_group
        )
        if not selected_target:
            raise ValueError("Host restore command has no target endpoint group")
        if worker.rank >= len(descriptors.shards):
            raise ValueError("source rank is outside the Host descriptor set")
        claimed = worker.claim_export(
            command.key.snapshot_id, str(command.attempt)
        )
        if claimed != descriptors.shards[worker.rank]:
            raise ValueError("claimed Host export differs from restore descriptor")
        return PreparedRankTransfer(
            _HostRestoreSourceState(descriptors, selected_target), requires_io=False
        )

    def commit(command, prepared, _completion) -> None:
        state = prepared.transfer_payload
        selected_target = state.target_group
        receipts = []
        observed = set()
        for participant, result in _rank_results(command):
            if participant.endpoint_group != selected_target:
                continue
            value = result.get("read_receipt")
            if value is None:
                raise ValueError(
                    f"target rank {participant.rank} omitted its READ receipt"
                )
            receipt = ReadReceipt.from_dict(value)
            if receipt.tp_rank != participant.rank:
                raise ValueError("READ receipt rank does not match ACK identity")
            observed.add(participant.rank)
            receipts.append(receipt)
        expected = set(range(len(state.descriptors.shards)))
        if observed != expected:
            raise ValueError("Host restore lacks a complete target TP fence")
        if not worker.release_export(
            command.key.snapshot_id,
            state.descriptors.shards,
            receipts,
            committed_attempt=str(command.attempt),
        ):
            raise RuntimeError("source Host export was not releasable")
        if not release_host_snapshot(command.key.snapshot_id):
            raise RuntimeError("source Host arena extent was not releasable")

    def abort(command, prepared, _completion) -> None:
        state = prepared.transfer_payload
        selected_target = state.target_group
        receipts = []
        observed = set()
        for participant, result in _rank_results(command):
            if participant.endpoint_group != selected_target:
                continue
            if participant.rank in observed:
                raise ValueError("duplicate target abort result")
            observed.add(participant.rank)
            value = result.get("read_receipt")
            if value is not None:
                receipt = ReadReceipt.from_dict(value)
                if receipt.tp_rank != participant.rank:
                    raise ValueError(
                        "cancel receipt rank does not match ACK identity"
                    )
                receipts.append(receipt)
                continue
            if result.get("fence") != "not_posted":
                raise ValueError(
                    "posted Host READ lacks an authoritative drained receipt"
                )
            shard = state.descriptors.shards[participant.rank]
            receipts.append(
                ReadReceipt(
                    shard.snapshot_id,
                    shard.export_id,
                    shard.tp_rank,
                    shard.tp_size,
                    shard.layout,
                    shard.token_count,
                    str(command.attempt),
                    "drained",
                )
            )
        expected = set(range(len(state.descriptors.shards)))
        if observed != expected:
            raise ValueError("Host abort lacks a complete target TP drain fence")
        if not worker.cancel_export(
            command.key.snapshot_id,
            state.descriptors.shards,
            receipts,
        ):
            raise RuntimeError("source Host export claim was not cancellable")

    return CallbackPathHandler(prepare, commit=commit, abort=abort)


def make_no_io_host_endpoint_handler() -> CallbackPathHandler:
    """Participant-side barrier hook for the endpoint inactive on a Host leg."""

    return CallbackPathHandler(
        lambda _command: PreparedRankTransfer(None, requires_io=False)
    )
