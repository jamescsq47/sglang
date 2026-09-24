from types import SimpleNamespace

from sglang.srt.disaggregation.agentic_default_physical_provider import (
    _TargetHandler,
    _TargetPrepared,
)
from sglang.srt.disaggregation.agentic_group_protocol import (
    CommandKind,
    GenerationKey,
    GroupCommand,
    Owner,
)
from sglang.srt.disaggregation.agentic_memory_authority import (
    LeaseKind,
    PhysicalMemoryLease,
    RequestGenerationAttempt,
)


def command():
    return GroupCommand(
        key=GenerationKey("run", "req", 0),
        attempt=1,
        command_seq=1,
        kind=CommandKind.PREPARE,
        source_owner=Owner.D_GPU,
        target_owner=Owner.P_GPU,
        lease_id="wire",
        payload={
            "agentic_data_plane": {
                "version": 1,
                "path": "d2p_direct",
                "operation": "direct",
            },
            "transfer": {"kind": "d2p"},
        },
    )


def lease():
    return PhysicalMemoryLease(
        lease_id=7,
        key=RequestGenerationAttempt("req", 1, 1),
        owner="d2p",
        kind=LeaseKind.PREFILL_WORKSET,
        page_size=1,
        parent_tokens=1,
        parent_allocated_tokens=1,
        prompt_tokens=2,
        prompt_allocated_tokens=2,
        growth_reserved_tokens=0,
        device_indices=(11, 12),
    )


class Authority:
    def __init__(self):
        self.begun = []

    def begin_io(self, lease_id, attempt):
        self.begun.append((lease_id, attempt))
        return True


def test_target_prepare_keeps_local_state_out_of_transport_payload():
    physical = lease()
    req = object()
    io_payload = object()
    authority = Authority()
    provider = SimpleNamespace(
        context=SimpleNamespace(authority=authority),
        _target_prepared={},
        _reserve_target=lambda _command: (physical, req),
        _direct_target_payload=lambda _command, _lease: io_payload,
    )
    handler = _TargetHandler(provider, direct=True)

    prepared = handler.prepare(command())

    assert prepared.transfer_payload is io_payload
    assert prepared.physical_lease_id == physical.lease_id
    assert provider._target_prepared[physical.lease_id] == _TargetPrepared(
        physical, req, io_payload
    )
    handler.begin_io(command(), prepared)
    assert authority.begun == [(physical.lease_id, "1")]
