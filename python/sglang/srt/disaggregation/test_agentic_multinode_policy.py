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


def plan(key, kind):
    return GroupTransferPlan(
        key=key,
        path=TransferPath.D2P_DIRECT,
        operation=TransferOperation.DIRECT,
        source_owner=Owner.D_GPU,
        target_owner=Owner.PREFILL_READY,
        lease_id=kind,
        payload={"kind": kind},
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
