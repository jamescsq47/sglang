from types import SimpleNamespace

import pytest
import torch

from sglang.srt.disaggregation.agentic_default_physical_provider import (
    AgenticDefaultPhysicalProvider,
    _TargetHandler,
    _TargetPrepared,
    _reverse_bootstrap_port,
)
from sglang.srt.disaggregation.agentic_group_protocol import (
    CommandKind,
    GenerationKey,
    GroupCommand,
    Owner,
)
from sglang.srt.disaggregation.agentic_group_transfer import (
    GroupTransferPlan,
    TransferOperation,
)
from sglang.srt.disaggregation.agentic_memory_authority import (
    LeaseKind,
    PhysicalMemoryLease,
    RequestGenerationAttempt,
)
from sglang.srt.disaggregation.agentic_transfer_queues import TransferPath


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


def test_target_prepare_rejects_bad_checkpoint_before_reserving_memory():
    cmd = command()
    cmd.payload["transfer"]["mamba_checkpoint_tokens"] = "not-an-integer"
    reserved = []
    provider = SimpleNamespace(
        _reserve_target=lambda _command: reserved.append(_command),
    )

    with pytest.raises((TypeError, ValueError)):
        _TargetHandler(provider, direct=True).prepare(cmd)

    assert reserved == []


def test_reverse_direct_uses_dedicated_bootstrap_port(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_DIRECT_BOOTSTRAP_PORT", "62000")
    assert _reverse_bootstrap_port() == 62000


def test_reverse_direct_rejects_missing_dedicated_port(monkeypatch):
    monkeypatch.delenv("SGLANG_AGENTIC_KV_DIRECT_BOOTSTRAP_PORT", raising=False)
    try:
        _reverse_bootstrap_port()
    except RuntimeError as exc:
        assert "dedicated" in str(exc)
    else:
        raise AssertionError("missing reverse bootstrap port must fail closed")


def test_decode_bind_installs_prefill_sampled_token_before_publish():
    physical = PhysicalMemoryLease(
        lease_id=9,
        key=RequestGenerationAttempt("req", 0, 1),
        owner="p2d",
        kind=LeaseKind.DECODE_RESERVATION,
        page_size=1,
        parent_tokens=1,
        parent_allocated_tokens=1,
        prompt_tokens=1,
        prompt_allocated_tokens=1,
        growth_reserved_tokens=1,
        device_indices=(11, 12),
    )
    req = SimpleNamespace(output_ids=[])

    class Bridge:
        def bind_decode(self, _lease, bound_req, **_kwargs):
            assert bound_req.output_ids == [42]

    provider = AgenticDefaultPhysicalProvider.__new__(
        AgenticDefaultPhysicalProvider
    )
    provider.context = SimpleNamespace(d_memory_bridge=Bridge())
    provider.adapter = SimpleNamespace(
        bind_decode_prompt=None,
        release_bound=None,
        release_decode_unadopted=None,
    )

    provider._bind_target(_TargetPrepared(physical, req, object(), 42))

    assert req.output_ids == [42]


def test_host_source_payload_freezes_registered_dma_index_mirror():
    provider = AgenticDefaultPhysicalProvider.__new__(
        AgenticDefaultPhysicalProvider
    )
    snapshot = SimpleNamespace(
        token_indices=torch.tensor([7, 8, 19, 20], dtype=torch.int32),
        state_indices=(3,),
    )
    provider._source_entry = lambda _command: (object(), object(), snapshot)
    cmd = command()
    cmd.payload["transfer"]["token_count"] = 3

    payload = provider._host_source_payload(cmd)

    assert payload.source_indices.tolist() == [7, 8, 19]
    assert payload.source_indices_host == (7, 8, 19)
    assert payload.state_indices == (3,)


def test_group_host_restore_commit_advances_capacity_epoch():
    calls = []

    class Policy:
        def committed(self, key):
            calls.append(("committed", key))

        def host_memory_available(self):
            calls.append(("capacity", None))

    key = GenerationKey("run", "host", 2)
    provider = AgenticDefaultPhysicalProvider.__new__(
        AgenticDefaultPhysicalProvider
    )
    provider._policy = Policy()
    provider._p2d_policy = None
    provider._pending_candidates = {}
    provider._host_results = {}
    group_restore = GroupTransferPlan(
        key=key,
        path=TransferPath.D2P_HOST,
        operation=TransferOperation.HOST_RESTORE,
        source_owner=Owner.D_HOST,
        target_owner=Owner.PREFILL_READY,
        lease_id="restore",
        payload={"kind": "d2p_host_restore"},
    )

    provider.on_committed(None, group_restore, 4)

    assert calls == [("committed", key), ("capacity", None)]


def test_host_store_capacity_abort_waits_but_other_failure_is_fatal():
    calls = []
    policy = SimpleNamespace(
        host_store_rejected=lambda key: calls.append(key),
    )
    key = GenerationKey("run", "host-full", 3)
    provider = AgenticDefaultPhysicalProvider.__new__(
        AgenticDefaultPhysicalProvider
    )
    provider._policy = policy
    provider._p2d_policy = None
    host_store = GroupTransferPlan(
        key=key,
        path=TransferPath.D2P_HOST,
        operation=TransferOperation.HOST_STORE,
        source_owner=Owner.D_GPU,
        target_owner=Owner.D_HOST,
        lease_id="store",
        payload={"kind": "d2p_host_store"},
    )

    provider.on_aborted(
        None, host_store, 5, "decode:d0:r3: source-local Host arena is full"
    )
    assert calls == [key]
    with pytest.raises(RuntimeError, match="non-capacity Host-store failure"):
        provider.on_aborted(None, host_store, 6, "CUDA copy failed")
