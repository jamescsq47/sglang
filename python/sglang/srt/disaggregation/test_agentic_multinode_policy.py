import threading
import time

from sglang.srt.disaggregation.agentic_group_protocol import GenerationKey, Owner
from sglang.srt.disaggregation.agentic_group_transfer import (
    GroupTransferPlan,
    TransferOperation,
)
from sglang.srt.disaggregation.agentic_multinode_policy import (
    D2PPolicyActor,
    P2DPolicyActor,
)
from sglang.srt.disaggregation.agentic_transfer_queues import TransferPath


def plan(
    key,
    kind,
    *,
    prompt_tokens=0,
    decode_growth_tokens=0,
    source_group="",
    target_group="",
):
    return GroupTransferPlan(
        key=key,
        path=TransferPath.D2P_DIRECT,
        operation=TransferOperation.DIRECT,
        source_owner=Owner.D_GPU,
        target_owner=Owner.PREFILL_READY,
        lease_id=kind,
        payload={
            "kind": kind,
            "prompt_tokens": prompt_tokens,
            "decode_growth_tokens": decode_growth_tokens,
        },
        source_group=source_group,
        target_group=target_group,
    )


def wait_count(values, count):
    deadline = time.monotonic() + 1
    while len(values) < count and time.monotonic() < deadline:
        time.sleep(0.002)
    assert len(values) >= count


def test_child_before_parent_submits_direct_once():
    key = GenerationKey("run", "r", 1)
    submitted = []
    actor = D2PPolicyActor(
        direct_window_seconds=1,
        submit=submitted.append,
        make_direct=lambda candidate, child: plan(key, "direct-" + child),
        make_host_store=lambda candidate: plan(key, "store"),
        make_host_restore=lambda candidate, child, desc: plan(key, "restore"),
    )
    try:
        actor.child_arrived(key, "child")
        actor.offer_parent(plan(key, "candidate"))
        wait_count(submitted, 1)
        assert [value.lease_id for value in submitted] == ["direct-child"]
    finally:
        actor.close()


def test_cancelled_generation_does_not_turn_racing_submit_rejection_fatal():
    key = GenerationKey("run", "cancel-race", 1)
    entered = threading.Event()
    release = threading.Event()

    def submit(_plan):
        entered.set()
        assert release.wait(1)
        raise RuntimeError("runtime is closing")

    actor = D2PPolicyActor(
        direct_window_seconds=1,
        submit=submit,
        make_direct=lambda candidate, child: plan(key, "direct"),
        make_host_store=lambda candidate: plan(key, "store"),
        make_host_restore=lambda candidate, child, desc: plan(key, "restore"),
    )
    try:
        actor.offer_parent(plan(key, "candidate"))
        actor.child_arrived(key, "child")
        assert entered.wait(1)
        actor.cancelled(key)
        release.set()
        time.sleep(0.02)
        # A fresh generation remains admissible; the expected CLOSING reject
        # from the cancelled generation did not poison the actor.
        next_key = GenerationKey("run", "after-cancel", 1)
        actor.offer_parent(plan(next_key, "candidate"))
        actor.cancelled(next_key)
    finally:
        release.set()
        actor.close()


def test_timeout_store_then_durable_waits_for_child():
    key = GenerationKey("run", "r", 2)
    submitted = []
    actor = D2PPolicyActor(
        direct_window_seconds=0.01,
        submit=submitted.append,
        make_direct=lambda candidate, child: plan(key, "direct"),
        make_host_store=lambda candidate: plan(key, "store"),
        make_host_restore=lambda candidate, child, desc: plan(
            key, "restore-" + desc
        ),
    )
    try:
        actor.offer_parent(plan(key, "candidate"))
        deadline = time.monotonic() + 1
        while not submitted and time.monotonic() < deadline:
            time.sleep(0.002)
        assert [value.lease_id for value in submitted] == ["store"]
        actor.host_durable(key, "host")
        assert len(submitted) == 1
        actor.child_arrived(key, object())
        wait_count(submitted, 2)
        assert [value.lease_id for value in submitted] == ["store", "restore-host"]
    finally:
        actor.close()


def test_durable_host_eviction_routes_late_child_to_full_recompute():
    key = GenerationKey("run", "evicted", 2)
    child_key = GenerationKey("run", "evicted", 3)
    submitted = []
    actor = D2PPolicyActor(
        direct_window_seconds=0.001,
        submit=submitted.append,
        make_direct=lambda candidate, child: plan(key, "direct"),
        make_host_store=lambda candidate: plan(key, "store"),
        make_host_restore=lambda candidate, child, desc: plan(key, "restore"),
        make_recompute=lambda parent, child: plan(child_key, "recompute"),
    )
    try:
        actor.offer_parent(plan(key, "candidate"))
        wait_count(submitted, 1)
        actor.host_durable(key, "host")
        assert actor.begin_eviction(key)
        actor.evicted(key)
        actor.child_arrived(key, {"target_generation": 3})
        wait_count(submitted, 2)
        assert submitted[-1].key == child_key
        assert submitted[-1].lease_id == "recompute"
        assert actor.phase_counts()["evicted_wait_child"] == 1
    finally:
        actor.close()


def test_application_final_retires_already_committed_eviction():
    key = GenerationKey("run", "evicted-final", 2)
    submitted = []
    actor = D2PPolicyActor(
        direct_window_seconds=0.001,
        submit=submitted.append,
        make_direct=lambda candidate, child: plan(key, "direct"),
        make_host_store=lambda candidate: plan(key, "store"),
        make_host_restore=lambda candidate, child, desc: plan(key, "restore"),
        make_recompute=lambda parent, child: plan(key, "recompute"),
    )
    try:
        actor.offer_parent(plan(key, "candidate"))
        wait_count(submitted, 1)
        actor.host_durable(key, "host")
        assert actor.begin_eviction(key)
        actor.evicted(key)
        assert actor.phase_counts() == {"evicted_wait_child": 1}

        assert actor.application_final(key)

        assert actor.phase_counts() == {}
    finally:
        actor.close()


def test_early_application_final_waits_for_parent_host_fence():
    key = GenerationKey("run", "early-final", 2)
    submitted = []
    actor = D2PPolicyActor(
        direct_window_seconds=0.001,
        submit=submitted.append,
        make_direct=lambda candidate, child: plan(key, "direct"),
        make_host_store=lambda candidate: plan(key, "store"),
        make_host_restore=lambda candidate, child, desc: plan(key, "restore"),
        make_recompute=lambda parent, child: plan(key, "recompute"),
    )
    try:
        # The TCP final edge can overtake the asynchronous D parent intent.
        # Unknown must remain pending rather than being mistaken for an
        # already-retired generation.
        assert not actor.application_final(key)
        actor.offer_parent(plan(key, "candidate"))
        wait_count(submitted, 1)
        actor.host_durable(key, "host")
        assert actor.begin_eviction(key)
        actor.evicted(key)
        assert actor.application_final(key)
        assert actor.phase_counts() == {}
    finally:
        actor.close()


def test_direct_capacity_failure_falls_back_and_restore_retries_on_edge():
    key = GenerationKey("run", "r", 3)
    submitted = []
    actor = D2PPolicyActor(
        direct_window_seconds=1,
        submit=submitted.append,
        make_direct=lambda candidate, child: plan(key, "direct"),
        make_host_store=lambda candidate: plan(key, "store"),
        make_host_restore=lambda candidate, child, desc: plan(key, "restore"),
    )
    try:
        actor.offer_parent(plan(key, "candidate"))
        actor.child_arrived(key, "child")
        actor.direct_rejected(key)
        actor.host_durable(key, "host")
        wait_count(submitted, 3)
        assert [value.lease_id for value in submitted] == [
            "direct",
            "store",
            "restore",
        ]
        actor.restore_capacity_rejected(key)
        actor.memory_available()
        wait_count(submitted, 4)
        assert [value.lease_id for value in submitted][-1] == "restore"
        assert len(submitted) == 4
    finally:
        actor.close()


def test_prefill_capacity_edge_only_wakes_matching_d2p_target_group():
    submitted = []

    def restore(candidate, child, _desc):
        return plan(
            candidate.key,
            "d2p-restore",
            source_group=candidate.source_group,
            target_group=child["target_group"],
        )

    actor = D2PPolicyActor(
        direct_window_seconds=1,
        submit=submitted.append,
        make_direct=lambda candidate, child: plan(
            candidate.key,
            "d2p-direct",
            source_group=candidate.source_group,
            target_group=child["target_group"],
        ),
        make_host_store=lambda candidate: plan(
            candidate.key, "d2p-store", source_group=candidate.source_group
        ),
        make_host_restore=restore,
    )
    try:
        for group in ("p0", "p1"):
            key = GenerationKey("run", f"to-{group}", 1)
            actor.offer_parent(plan(key, "candidate", source_group="d0"))
            actor.child_arrived(key, {"target_group": group})
            actor.direct_rejected(key)
            actor.host_durable(key, "host")
            actor.restore_capacity_rejected(key)
        wait_count(submitted, 6)
        before = len(submitted)

        actor.memory_available(1024, endpoint_group="p0")
        wait_count(submitted, before + 1)

        restored = submitted[before:]
        assert [
            (value.key.request_id, value.target_group) for value in restored
        ] == [("to-p0", "p0")]
    finally:
        actor.close()


def test_p2d_host_restore_retries_only_on_capacity_edge():
    key = GenerationKey("run", "p", 4)
    submitted = []
    direct = plan(key, "p2d-direct")
    actor = P2DPolicyActor(
        submit=submitted.append,
        make_host_store=lambda value: plan(value.key, "p2d-store"),
        make_host_restore=lambda value, _desc: plan(value.key, "p2d-restore"),
    )
    actor.direct_submitted(direct)
    actor.direct_rejected(key)
    actor.host_durable(key, "host")
    assert [value.lease_id for value in submitted] == [
        "p2d-store",
        "p2d-restore",
    ]
    actor.restore_capacity_rejected(key)
    assert len(submitted) == 2
    actor.memory_available()
    assert [value.lease_id for value in submitted][-1] == "p2d-restore"


def test_p2d_can_spill_before_decode_target_is_late_bound():
    key = GenerationKey("run", "p-late", 4)
    submitted = []
    candidate = plan(key, "p2d-direct")
    candidate = GroupTransferPlan(
        key=candidate.key,
        path=candidate.path,
        operation=candidate.operation,
        source_owner=candidate.source_owner,
        target_owner=candidate.target_owner,
        lease_id=candidate.lease_id,
        payload={**candidate.payload, "late_bind_pending": True},
        source_group="p0",
    )
    actor = P2DPolicyActor(
        submit=submitted.append,
        make_host_store=lambda value: plan(value.key, "p2d-store"),
        make_host_restore=lambda value, _desc: GroupTransferPlan(
            key=value.key,
            path=value.path,
            operation=value.operation,
            source_owner=value.source_owner,
            target_owner=value.target_owner,
            lease_id="p2d-restore",
            payload=value.payload,
            source_group=value.source_group,
            target_group=value.target_group,
        ),
    )
    actor.direct_submitted(candidate)
    actor.direct_rejected(key)
    actor.host_durable(key, "host")
    assert [value.lease_id for value in submitted] == ["p2d-store"]

    selected = GroupTransferPlan(
        key=candidate.key,
        path=candidate.path,
        operation=candidate.operation,
        source_owner=candidate.source_owner,
        target_owner=candidate.target_owner,
        lease_id=candidate.lease_id,
        payload={"kind": "p2d"},
        source_group="p0",
        target_group="d3",
    )
    restore = actor.bind_target(selected)
    assert restore.lease_id == "p2d-restore"
    assert restore.target_group == "d3"


def test_p2d_late_target_and_spill_are_mutually_exclusive():
    key = GenerationKey("run", "p-race", 1)
    submitted = []
    candidate = GroupTransferPlan(
        key=key,
        path=TransferPath.P2D_DIRECT,
        operation=TransferOperation.DIRECT,
        source_owner=Owner.P_GPU,
        target_owner=Owner.D_GPU,
        lease_id="candidate",
        payload={"late_bind_pending": True},
        source_group="p0",
    )
    actor = P2DPolicyActor(
        submit=submitted.append,
        make_host_store=lambda value: GroupTransferPlan(
            key=value.key,
            path=TransferPath.P2D_HOST,
            operation=TransferOperation.HOST_STORE,
            source_owner=Owner.P_GPU,
            target_owner=Owner.P_HOST,
            lease_id="store",
            payload=value.payload,
            source_group=value.source_group,
        ),
        make_host_restore=lambda value, _desc: value,
    )
    actor.direct_submitted(candidate)
    selected = GroupTransferPlan(
        key=key,
        path=candidate.path,
        operation=candidate.operation,
        source_owner=candidate.source_owner,
        target_owner=candidate.target_owner,
        lease_id=candidate.lease_id,
        payload={},
        source_group="p0",
        target_group="d0",
    )
    assert actor.bind_target(selected) == selected
    assert actor.spill_if_unbound(key) is False
    assert submitted == []


def test_p2d_late_target_bypasses_rejected_full_host_store():
    key = GenerationKey("run", "p-host-full", 1)
    submitted = []
    candidate = GroupTransferPlan(
        key=key,
        path=TransferPath.P2D_DIRECT,
        operation=TransferOperation.DIRECT,
        source_owner=Owner.P_GPU,
        target_owner=Owner.D_GPU,
        lease_id="candidate",
        payload={"late_bind_pending": True},
        source_group="p0",
    )
    actor = P2DPolicyActor(
        submit=submitted.append,
        make_host_store=lambda value: GroupTransferPlan(
            key=value.key,
            path=TransferPath.P2D_HOST,
            operation=TransferOperation.HOST_STORE,
            source_owner=Owner.P_GPU,
            target_owner=Owner.P_HOST,
            lease_id="store",
            payload=value.payload,
            source_group=value.source_group,
        ),
        make_host_restore=lambda value, _desc: value,
    )
    actor.direct_submitted(candidate)
    assert actor.spill_if_unbound(key) is True
    actor.host_store_rejected(key)
    selected = GroupTransferPlan(
        key=key,
        path=candidate.path,
        operation=candidate.operation,
        source_owner=candidate.source_owner,
        target_owner=candidate.target_owner,
        lease_id=candidate.lease_id,
        payload={},
        source_group="p0",
        target_group="d2",
    )

    assert actor.bind_target(selected) == selected
    before = len(submitted)
    actor.direct_rejected(key)
    actor.host_store_rejected(key)
    # A real failure of the newly selected Direct path returns to passive
    # Host-capacity wait; it must not spin Direct without a new edge.
    assert len(submitted) == before + 1


def test_one_capacity_edge_wakes_only_one_restore():
    submitted = []
    actor = P2DPolicyActor(
        submit=submitted.append,
        make_host_store=lambda value: plan(value.key, "p2d-store"),
        make_host_restore=lambda value, _desc: plan(value.key, "p2d-restore"),
    )
    keys = [GenerationKey("run", f"p-{index}", 1) for index in range(3)]
    for key in keys:
        actor.direct_submitted(plan(key, "p2d-direct"))
        actor.direct_rejected(key)
        actor.host_durable(key, "host")
        actor.restore_capacity_rejected(key)
    before = len(submitted)

    actor.memory_available()

    assert len(submitted) == before + 1


def test_capacity_snapshot_admits_all_feasible_and_skips_large_head():
    submitted = []
    actor = P2DPolicyActor(
        submit=submitted.append,
        make_host_store=lambda value: plan(value.key, "p2d-store"),
        make_host_restore=lambda value, _desc: plan(
            value.key, "p2d-restore"
        ),
    )
    entries = [
        (GenerationKey("run", "large", 1), 900),
        (GenerationKey("run", "small-a", 1), 300),
        (GenerationKey("run", "small-b", 1), 300),
    ]
    for key, tokens in entries:
        actor.direct_submitted(
            plan(key, "p2d-direct", prompt_tokens=tokens)
        )
        actor.direct_rejected(key)
        actor.host_durable(key, "host")
        actor.restore_capacity_rejected(key)
    before = len(submitted)

    actor.memory_available(700)

    restored = submitted[before:]
    assert [value.key.request_id for value in restored] == ["small-a", "small-b"]


def test_decode_capacity_edge_only_wakes_matching_p2d_target_group():
    submitted = []
    actor = P2DPolicyActor(
        submit=submitted.append,
        make_host_store=lambda value: plan(
            value.key, "p2d-store", source_group=value.source_group
        ),
        make_host_restore=lambda value, _desc: plan(
            value.key,
            "p2d-restore",
            source_group=value.source_group,
            target_group=value.target_group,
        ),
    )
    for group in ("d0", "d1"):
        key = GenerationKey("run", f"to-{group}", 1)
        actor.direct_submitted(
            plan(
                key,
                "p2d-direct",
                prompt_tokens=128,
                source_group="p0",
                target_group=group,
            )
        )
        actor.direct_rejected(key)
        actor.host_durable(key, "host")
        actor.restore_capacity_rejected(key)
    before = len(submitted)

    actor.memory_available(1024, endpoint_group="d0")

    restored = submitted[before:]
    assert [(value.key.request_id, value.target_group) for value in restored] == [
        ("to-d0", "d0")
    ]


def test_d2p_host_store_retries_only_after_source_arena_release():
    key = GenerationKey("run", "d2p-host-full", 6)
    submitted = []
    actor = D2PPolicyActor(
        direct_window_seconds=1,
        submit=submitted.append,
        make_direct=lambda candidate, child: plan(key, "direct"),
        make_host_store=lambda candidate: plan(key, "store"),
        make_host_restore=lambda candidate, child, desc: plan(key, "restore"),
    )
    try:
        actor.offer_parent(plan(key, "candidate"))
        actor.child_arrived(key, "child")
        actor.direct_rejected(key)
        wait_count(submitted, 2)
        actor.host_store_rejected(key)
        time.sleep(0.01)
        assert [value.lease_id for value in submitted] == ["direct", "store"]
        actor.host_memory_available()
        wait_count(submitted, 3)
        assert submitted[-1].lease_id == "store"
    finally:
        actor.close()


def test_p2d_host_store_retries_only_after_source_arena_release():
    key = GenerationKey("run", "p2d-host-full", 7)
    submitted = []
    actor = P2DPolicyActor(
        submit=submitted.append,
        make_host_store=lambda value: plan(value.key, "p2d-store"),
        make_host_restore=lambda value, _desc: plan(value.key, "p2d-restore"),
    )
    actor.direct_submitted(plan(key, "p2d-direct"))
    actor.direct_rejected(key)
    actor.host_store_rejected(key)
    assert [value.lease_id for value in submitted] == ["p2d-store"]
    actor.host_memory_available()
    assert [value.lease_id for value in submitted] == [
        "p2d-store",
        "p2d-store",
    ]


def test_host_capacity_epoch_is_not_lost_before_abort_finalize():
    d_key = GenerationKey("run", "d-early-edge", 8)
    d_submitted = []
    d2p = D2PPolicyActor(
        direct_window_seconds=1,
        submit=d_submitted.append,
        make_direct=lambda candidate, child: plan(d_key, "direct"),
        make_host_store=lambda candidate: plan(d_key, "store"),
        make_host_restore=lambda candidate, child, desc: plan(d_key, "restore"),
    )
    p_key = GenerationKey("run", "p-early-edge", 9)
    p_submitted = []
    p2d = P2DPolicyActor(
        submit=p_submitted.append,
        make_host_store=lambda value: plan(value.key, "p2d-store"),
        make_host_restore=lambda value, _desc: plan(value.key, "p2d-restore"),
    )
    try:
        d2p.offer_parent(plan(d_key, "candidate"))
        d2p.child_arrived(d_key, "child")
        d2p.direct_rejected(d_key)
        wait_count(d_submitted, 2)
        d2p.host_memory_available()
        d2p.host_store_rejected(d_key)
        wait_count(d_submitted, 3)
        assert [value.lease_id for value in d_submitted][-2:] == ["store", "store"]

        p2d.direct_submitted(plan(p_key, "p2d-direct"))
        p2d.direct_rejected(p_key)
        p2d.host_memory_available()
        p2d.host_store_rejected(p_key)
        assert [value.lease_id for value in p_submitted] == [
            "p2d-store",
            "p2d-store",
        ]
    finally:
        d2p.close()


def test_unknown_or_terminal_direct_rejection_is_idempotent():
    key = GenerationKey("run", "unknown", 5)
    submitted = []
    d2p = D2PPolicyActor(
        direct_window_seconds=1,
        submit=submitted.append,
        make_direct=lambda candidate, child: plan(key, "direct"),
        make_host_store=lambda candidate: plan(key, "store"),
        make_host_restore=lambda candidate, child, desc: plan(key, "restore"),
    )
    p2d = P2DPolicyActor(
        submit=submitted.append,
        make_host_store=lambda value: plan(value.key, "p2d-store"),
        make_host_restore=lambda value, _desc: plan(value.key, "p2d-restore"),
    )
    try:
        d2p.direct_rejected(key)
        p2d.direct_rejected(key)
        assert submitted == []
    finally:
        d2p.close()
