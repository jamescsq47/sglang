from types import SimpleNamespace
import threading

import pytest
import torch

from sglang.srt.disaggregation.agentic_default_physical_provider import (
    AgenticDefaultPhysicalProvider,
    _D2PHostEntry,
    _P2D_CHILD_ARRIVED,
    _DispatchHandler,
    _TargetHandler,
    _TargetPrepared,
    _reverse_bootstrap_port,
)
from sglang.srt.disaggregation.agentic_group_protocol import (
    CommandKind,
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
    TransferOperation,
)
from sglang.srt.disaggregation.agentic_memory_authority import (
    LeaseKind,
    PhysicalMemoryLease,
    RequestGenerationAttempt,
)
from sglang.srt.disaggregation.agentic_remote_host import HostShard
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


def test_direct_lane_count_is_configurable_without_initializing_paths(
    monkeypatch,
):
    monkeypatch.setenv("SGLANG_AGENTIC_MULTINODE_DIRECT_LANES", "8")
    monkeypatch.setenv("SGLANG_AGENTIC_MULTINODE_D2P_DIRECT_LANES", "4")
    monkeypatch.setenv("SGLANG_AGENTIC_MULTINODE_P2D_DIRECT_LANES", "8")
    monkeypatch.setenv("SGLANG_AGENTIC_MULTINODE_HOST_LANES", "6")
    provider = AgenticDefaultPhysicalProvider(
        SimpleNamespace(tp_rank=1), SimpleNamespace(role="decode")
    )

    lanes = provider.lanes(None)

    assert lanes[TransferPath.D2P_DIRECT] == 4
    assert lanes[TransferPath.P2D_DIRECT] == 8
    assert lanes[TransferPath.D2P_HOST] == 6
    assert lanes[TransferPath.P2D_HOST] == 6


class Authority:
    def __init__(self):
        self.begun = []
        self.released = []

    def begin_io(self, lease_id, attempt):
        self.begun.append((lease_id, attempt))
        return True

    def request_release(self, lease_id):
        self.released.append(("request", lease_id))
        return True

    def commit_release(self, lease_id, *, reason):
        self.released.append(("commit", lease_id, reason))
        return True


class CleanupPayload:
    def __init__(self):
        self.cleanup_calls = 0

    def cleanup(self):
        self.cleanup_calls += 1


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


def test_direct_target_success_cleans_transport_after_publish():
    physical = lease()
    payload = CleanupPayload()
    published = []
    provider = SimpleNamespace(
        context=SimpleNamespace(authority=Authority()),
        _target_prepared={
            physical.lease_id: _TargetPrepared(physical, object(), payload)
        },
        _publish_target=lambda local: published.append(local.lease.lease_id),
    )
    prepared = PreparedRankTransfer(payload, physical_lease_id=physical.lease_id)

    _TargetHandler(provider, direct=True).commit(command(), prepared, object())

    assert published == [physical.lease_id]
    assert payload.cleanup_calls == 1
    assert provider._target_prepared == {}


def test_direct_target_posted_cancel_cleans_after_workset_release():
    physical = lease()
    payload = CleanupPayload()
    authority = Authority()
    provider = SimpleNamespace(
        context=SimpleNamespace(authority=authority),
        _target_prepared={
            physical.lease_id: _TargetPrepared(physical, object(), payload)
        },
    )
    prepared = PreparedRankTransfer(payload, physical_lease_id=physical.lease_id)

    _TargetHandler(provider, direct=True).abort(command(), prepared, object())

    assert authority.released == [
        ("request", physical.lease_id),
        ("commit", physical.lease_id, "target_attempt_aborted"),
    ]
    assert payload.cleanup_calls == 1
    assert provider._target_prepared == {}


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


def test_host_source_payload_defers_index_mirror_to_io_worker():
    provider = AgenticDefaultPhysicalProvider.__new__(
        AgenticDefaultPhysicalProvider
    )
    snapshot = SimpleNamespace(
        token_indices=torch.tensor([7, 8, 19, 20], dtype=torch.int32),
        state_indices=(3,),
        state_slot_count=1,
    )
    provider._source_entry = lambda _command: (object(), object(), snapshot)
    cmd = command()
    cmd.payload["transfer"]["token_count"] = 3

    payload = provider._host_source_payload(cmd)

    assert payload.source_indices.tolist() == [7, 8, 19]
    assert callable(payload.source_indices_host)
    assert payload.source_indices_host() == (7, 8, 19)
    assert callable(payload.state_indices)
    assert payload.state_indices() == (3,)


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


def test_application_final_evicts_durable_host_snapshot_without_recompute():
    key = GenerationKey("run", "terminal-host", 5)
    submitted = []
    begun = []
    provider = AgenticDefaultPhysicalProvider.__new__(
        AgenticDefaultPhysicalProvider
    )
    provider._policy = SimpleNamespace(
        begin_eviction=lambda value: begun.append(value) or True,
        eviction_rejected=lambda _value: None,
    )
    provider._submitter = submitted.append
    provider._d2p_host_lock = threading.RLock()
    provider._terminal_d2p = set()
    provider._d2p_host_entries = {
        key: _D2PHostEntry(key, "d0", 128, (1024, 1024), 1.0)
    }

    provider.application_final(None, key)

    assert begun == [key]
    assert len(submitted) == 1
    assert submitted[0].operation is TransferOperation.HOST_EVICT
    assert submitted[0].payload["reason"] == "application_final"


def test_application_final_wins_race_with_submitted_pressure_eviction():
    key = GenerationKey("run", "terminal-evict-race", 6)
    calls = []
    provider = AgenticDefaultPhysicalProvider.__new__(
        AgenticDefaultPhysicalProvider
    )
    provider._policy = SimpleNamespace(
        cancelled=lambda value: calls.append(("cancelled", value)),
        evicted=lambda value: calls.append(("evicted", value)),
        host_memory_available=lambda: calls.append(("capacity", None)),
    )
    provider._p2d_policy = None
    provider._d2p_host_lock = threading.RLock()
    provider._terminal_d2p = {key}
    provider._d2p_host_entries = {
        key: _D2PHostEntry(key, "d0", 128, (1024, 1024), 1.0, evicting=True)
    }
    provider._d2p_host_evictions = 0
    provider._pending_candidates = {}
    provider._host_results = {}
    plan = GroupTransferPlan(
        key=key,
        path=TransferPath.D2P_HOST,
        operation=TransferOperation.HOST_EVICT,
        source_owner=Owner.D_HOST,
        target_owner=Owner.NONE,
        lease_id="pressure-evict",
        payload={"reason": "d2p_host_high_watermark"},
        source_group="d0",
    )

    provider.on_committed(None, plan, 1)

    assert calls == [("cancelled", key), ("capacity", None)]
    assert key not in provider._d2p_host_entries
    assert key not in provider._terminal_d2p


def test_application_final_after_eviction_commit_retires_tombstone():
    key = GenerationKey("run", "terminal-after-evict", 7)
    retired = []
    provider = AgenticDefaultPhysicalProvider.__new__(
        AgenticDefaultPhysicalProvider
    )
    provider._policy = SimpleNamespace(
        application_final=lambda value: retired.append(value) or True,
    )
    provider._submitter = lambda _plan: None
    provider._d2p_host_lock = threading.RLock()
    provider._terminal_d2p = set()
    provider._d2p_host_entries = {}

    provider.application_final(None, key)

    assert retired == [key]
    assert key not in provider._terminal_d2p


def test_p2d_prefill_candidate_waits_for_late_bound_decode_group():
    key = GenerationKey("run", "late-p2d", 3)
    candidate = GroupTransferPlan(
        key=key,
        path=TransferPath.P2D_DIRECT,
        operation=TransferOperation.DIRECT,
        source_owner=Owner.P_GPU,
        target_owner=Owner.D_GPU,
        lease_id="p2d",
        payload={"kind": "p2d", "target_generation": 3},
        source_group="p2",
    )
    registered = []
    provider = AgenticDefaultPhysicalProvider.__new__(
        AgenticDefaultPhysicalProvider
    )
    provider._policy = None
    provider._p2d_policy = SimpleNamespace(
        direct_submitted=lambda plan: registered.append(plan),
        bind_target=lambda plan: plan,
        direct_rejected=lambda _key: None,
    )
    provider._pending_p2d_candidates = {}
    provider._early_p2d_targets = {}
    provider._p2d_spill_timers = {}
    provider._p2d_route_lock = __import__("threading").RLock()
    provider.role = "prefill"
    provider.scheduler = SimpleNamespace(tp_rank=0)

    assert provider.decide_intent(
        None,
        LinkIntent(
            "fabric",
            LinkParticipant("prefill", "p2", 0),
            key,
            1,
            "plan",
        ),
        candidate,
    ) is None
    selected = provider.decide_intent(
        None,
        LinkIntent(
            "fabric",
            LinkParticipant("decode", "d3", 0),
            key,
            1,
            _P2D_CHILD_ARRIVED,
            {"target_group": "d3", "target_generation": 3},
        ),
        None,
    )

    assert selected.target_group == "d3"
    assert selected.payload["target_group"] == "d3"
    assert registered == [candidate]


def test_d2p_child_intent_hashes_only_complete_pages():
    provider = AgenticDefaultPhysicalProvider.__new__(
        AgenticDefaultPhysicalProvider
    )
    provider.role = "prefill"
    provider.config = SimpleNamespace(endpoint_group="p0")
    parent = GenerationKey("run", "unaligned", 0)
    record = SimpleNamespace(
        key=GenerationKey("run", "unaligned", 1),
        parent_key=parent,
        req=SimpleNamespace(origin_input_ids=list(range(70))),
    )

    key, event, payload = provider.request_control_intent(
        SimpleNamespace(authority=SimpleNamespace(page_size=64)), record
    )

    assert key == parent
    assert event != ""
    assert payload["prompt_tokens"] == 70
    assert len(payload["page_chain_hashes"]) == 1


def test_p2d_child_intent_carries_router_decode_reservation():
    provider = AgenticDefaultPhysicalProvider.__new__(
        AgenticDefaultPhysicalProvider
    )
    provider.role = "decode"
    provider.config = SimpleNamespace(endpoint_group="d2")
    record = SimpleNamespace(
        key=GenerationKey("run", "late-bound", 3),
        req=SimpleNamespace(
            sampling_params=SimpleNamespace(
                custom_params={
                    "agentic_decode_reservation_id": "reservation-7"
                }
            )
        ),
    )

    _key, event, payload = provider.request_control_intent(None, record)

    assert event == _P2D_CHILD_ARRIVED
    assert payload["target_group"] == "d2"
    assert payload["decode_reservation_id"] == "reservation-7"


def test_p2d_materialized_schedules_nonblocking_router_ack(monkeypatch):
    monkeypatch.setenv(
        "SGLANG_AGENTIC_DECODE_RESERVATION_CALLBACK_URL",
        "http://router/dualpd/decode_materialized",
    )
    submissions = []
    provider = AgenticDefaultPhysicalProvider.__new__(
        AgenticDefaultPhysicalProvider
    )
    provider._route_notifier = SimpleNamespace(submit=submissions.append)
    plan = GroupTransferPlan(
        key=GenerationKey("run", "prepared", 1),
        path=TransferPath.P2D_DIRECT,
        operation=TransferOperation.DIRECT,
        source_owner=Owner.P_GPU,
        target_owner=Owner.D_GPU,
        lease_id="prepared",
        payload={"decode_reservation_id": "reservation-9"},
        source_group="p0",
        target_group="d0",
    )

    provider.on_materialized(None, plan, 4)

    assert len(submissions) == 1


def test_host_target_payload_defers_destination_indices_to_io_thread():
    provider = AgenticDefaultPhysicalProvider.__new__(
        AgenticDefaultPhysicalProvider
    )
    provider.scheduler = SimpleNamespace(tp_rank=0)
    physical = PhysicalMemoryLease(
        lease_id=8,
        key=RequestGenerationAttempt("req", 2, 1),
        owner="p2d",
        kind=LeaseKind.DECODE_RESERVATION,
        page_size=64,
        parent_tokens=3,
        parent_allocated_tokens=64,
        prompt_tokens=3,
        prompt_allocated_tokens=64,
        growth_reserved_tokens=64,
        device_indices=torch.arange(64, 128, device="cpu"),
    )
    shard = HostShard(
        "run:req:2",
        "export",
        0,
        1,
        "a" * 64,
        3,
        4096,
        8192,
        "YQ==",
        "source",
    )
    restore = GroupCommand(
        key=GenerationKey("run", "req", 2),
        attempt=1,
        command_seq=1,
        kind=CommandKind.PREPARE,
        source_owner=Owner.P_HOST,
        target_owner=Owner.D_GPU,
        lease_id="h2d",
        payload={
            "agentic_data_plane": {
                "version": 1,
                "path": "p2d_host",
                "operation": "host_restore",
            },
            "transfer": {"host_shards": [shard.to_dict()]},
        },
    )

    payload = provider._host_target_payload(restore, physical)

    assert callable(payload.device_indices)
    assert payload.device_indices() == (64, 65, 66)


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


def test_dispatch_handler_retires_no_io_participant_state():
    selected = CallbackPathHandler(
        lambda _command: PreparedRankTransfer(None, requires_io=False)
    )
    dispatch = _DispatchHandler(lambda _command: selected)
    prepare = command()
    dispatch.prepare(prepare)
    assert dispatch.active_count == 1
    finalize = GroupCommand(
        key=prepare.key,
        attempt=prepare.attempt,
        command_seq=prepare.command_seq + 1,
        kind=CommandKind.FINALIZE,
        source_owner=prepare.source_owner,
        target_owner=prepare.target_owner,
        lease_id=prepare.lease_id,
        payload=prepare.payload,
    )
    dispatch.retire(finalize)
    assert dispatch.active_count == 0


def test_d2p_host_pressure_selects_shortest_generation_to_low_watermark(
    monkeypatch,
):
    monkeypatch.setenv("SGLANG_AGENTIC_MULTINODE_D2P_HOST_GIB", "0.000001")
    monkeypatch.setenv(
        "SGLANG_AGENTIC_MULTINODE_D2P_HOST_HIGH_WATERMARK", "0.90"
    )
    monkeypatch.setenv(
        "SGLANG_AGENTIC_MULTINODE_D2P_HOST_LOW_WATERMARK", "0.75"
    )
    selected = []
    submitted = []

    class Policy:
        def begin_eviction(self, key):
            selected.append(key)
            return True

        def eviction_rejected(self, _key):
            raise AssertionError("eviction submission should not fail")

    provider = AgenticDefaultPhysicalProvider.__new__(
        AgenticDefaultPhysicalProvider
    )
    provider._policy = Policy()
    provider._submitter = submitted.append
    provider._d2p_host_lock = __import__("threading").RLock()
    provider._d2p_host_entries = {}
    now = 10.0
    for request_id, tokens, age in (
        ("long", 900, 0),
        ("short-old", 100, 2),
        ("short-new", 100, 1),
    ):
        key = GenerationKey("run", request_id, 1)
        provider._d2p_host_entries[key] = _D2PHostEntry(
            key, "d0", tokens, (400, 400), now - age
        )

    assert provider._schedule_d2p_host_evictions("d0") == 1
    assert [key.request_id for key in selected] == ["short-old"]
    assert submitted[0].operation is TransferOperation.HOST_EVICT
    assert submitted[0].source_owner is Owner.D_HOST
    assert submitted[0].target_owner is Owner.NONE
