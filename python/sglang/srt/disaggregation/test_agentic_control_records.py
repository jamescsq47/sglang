import builtins
from concurrent.futures import ThreadPoolExecutor

import pytest

from sglang.srt.disaggregation.agentic_control_records import (
    ControlCapacityError, ControlRecordError, ControlRecords, ControlReplayError,
    Mutation, RecordKey,
)


KEY = RecordKey("host-export", "request:0", "attempt-1")


def create(value=None, owner="source"):
    return Mutation(KEY, 0, None, owner, {"rank": 0} if value is None else value)


def test_create_owner_cas_and_no_aba():
    core = ControlRecords("run")
    first = core.mutate("run", "source", 1, create())
    assert first.applied
    assert not core.mutate("run", "other", 1, create()).applied
    bad = Mutation(KEY, first.record.revision, "wrong", "receiver", {})
    assert not core.mutate("run", "other", 2, bad).applied
    moved = core.mutate("run", "source", 2, Mutation(
        KEY, first.record.revision, "source", "receiver", {"loaded": True}))
    assert moved.applied
    stale_delete = Mutation(KEY, first.record.revision, "source", "source", delete=True)
    assert not core.mutate("run", "source", 3, stale_delete).applied
    deleted = core.mutate("run", "receiver", 1, Mutation(
        KEY, moved.record.revision, "receiver", "receiver", delete=True))
    assert deleted.applied and deleted.record.deleted
    assert not core.mutate("run", "source", 4, create()).applied


def test_lost_ack_returns_exact_original_result_after_other_owner_changes():
    core = ControlRecords("run")
    op = create()
    first = core.mutate("run", "source", 1, op)
    core.mutate("run", "receiver", 1, Mutation(
        KEY, first.record.revision, "source", "receiver", delete=True))
    # Replay receipt survives deletion: a lost reply cannot create a fake
    # failure or resurrect an already released descriptor/claim.
    assert core.mutate("run", "source", 1, op) == first
    with pytest.raises(ControlReplayError):
        core.mutate("run", "source", 1, create({"different": True}))
    core.mutate("run", "source", 2, create())
    with pytest.raises(ControlReplayError):
        core.mutate("run", "source", 1, op)
    with pytest.raises(ControlReplayError):
        core.mutate("run", "source", 4, op)


@pytest.mark.parametrize("ranks", [1, 2, 8])
def test_concurrent_create_has_one_owner(ranks):
    core = ControlRecords("run")
    with ThreadPoolExecutor(max_workers=ranks) as pool:
        results = list(pool.map(lambda rank: core.mutate(
            "run", str(rank), 1, create(owner=str(rank))), range(ranks)))
    assert sum(result.applied for result in results) == 1
    assert len({result.record.owner for result in results}) == 1


def test_no_files_and_values_cannot_mutate_committed_state(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("control state must not touch files")
    monkeypatch.setattr(builtins, "open", forbidden)
    core = ControlRecords("run")
    value = {"ranks": [0]}
    result = core.mutate("run", "c", 1, create(value))
    value["ranks"].append(1)
    result.record.value["ranks"].append(2)
    got = core.get("run", KEY)
    assert got.value == {"ranks": [0]}
    got.value["ranks"].append(3)
    cursor, records = core.snapshot("run")
    records[KEY].value["ranks"].append(4)
    events = core.changes("run", 0, timeout=0)
    events[0].record.value["ranks"].append(5)
    assert core.get("run", KEY).value == {"ranks": [0]}
    assert core.changes("run", cursor, timeout=0) == ()


def test_commit_notifies_waiter_and_overflow_is_not_silent():
    core = ControlRecords("run", event_capacity=1)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(core.changes, "run", 0, timeout=2)
        first = core.mutate("run", "c", 1, create())
        assert future.result(timeout=2)[0].record == first.record
    core.mutate("run", "c", 2, Mutation(
        KEY, first.record.revision, "source", "source", {"phase": "ready"}))
    assert core.changes("run", 0, timeout=0) is None
    cursor, records = core.snapshot("run")
    assert records[KEY].value == {"phase": "ready"}
    assert core.changes("run", cursor, timeout=0) == ()


def test_budget_exhaustion_never_drops_owner_or_receipt():
    core = ControlRecords("run", max_records=1, max_clients=2)
    original = core.mutate("run", "c", 1, create())
    with pytest.raises(ControlCapacityError):
        core.mutate("run", "c", 2, Mutation(
            RecordKey("host-export", "request:1", "attempt-1"), 0, None, "c", {}))
    assert core.get("run", KEY) == original.record
    assert core.mutate("run", "c", 1, create()) == original
    core.mutate("run", "other", 1, create())
    with pytest.raises(ControlCapacityError):
        core.mutate("run", "third", 1, create())


def test_bad_epoch_and_shutdown_cannot_grant_release():
    core = ControlRecords("run")
    with pytest.raises(ControlRecordError):
        core.mutate("old-run", "c", 1, create())
    core.mutate("run", "c", 1, create())
    with ThreadPoolExecutor(max_workers=1) as pool:
        waiter = pool.submit(core.changes, "run", 1)
        core.close()
        with pytest.raises(ControlRecordError):
            waiter.result(timeout=2)
    with pytest.raises(ControlRecordError):
        core.get("run", KEY)


@pytest.mark.parametrize("key", [("", "s", "a"), ("ns", "", "a"), ("ns", "s", "")])
def test_explicit_identity_required(key):
    with pytest.raises(ValueError):
        RecordKey(*key)


def test_non_json_and_unowned_update_rejected_without_commit():
    core = ControlRecords("run")
    with pytest.raises(ValueError):
        core.mutate("run", "c", 1, create({"bad": float("nan")}))
    first = core.mutate("run", "c", 1, create())
    with pytest.raises(ValueError):
        core.mutate("run", "c", 2, Mutation(KEY, first.record.revision, None, "c"))
    assert core.get("run", KEY) == first.record


def test_generation_owner_excludes_other_attempt_and_path():
    core = ControlRecords("run")
    key = RecordKey.ownership("request:0")
    direct = core.mutate("run", "p", 1, Mutation(
        key, 0, None, "direct-attempt-1", {"path": "direct"}))
    assert direct.applied
    assert not core.mutate("run", "d", 1, Mutation(
        key, 0, None, "slow-attempt-2", {"path": "slow"})).applied
