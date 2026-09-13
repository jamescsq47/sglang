"""Foreign Host readiness must be discoverable before TP selects a restore.

No ownership, mmap, claim, allocator or DMA operation belongs in discovery.
The existing selected-request path revalidates the ledger before each claim.
"""
import copy
from types import SimpleNamespace

import pytest

from sglang.srt.disaggregation.agentic_host_staging import AgenticPHostStagingManager


def manager(rank=0):
    value = object.__new__(AgenticPHostStagingManager)
    value.owner = "p-group:prefill-1"
    value.arena_domain = 1
    value.tp_rank = rank
    value.tp_size = 2
    value.host_ready = {}
    value._ledger_entries_cache = {"req:3": {
        "snapshot_id": "req:3", "state": "host_ready",
        "p_owner": "p-group:prefill-0", "arena_domain": 0,
        "recovery_domain": 1, "tp_size": 2,
        "rank_grants": {"0": {"arena_path": "/not-opened/0"},
                        "1": {"arena_path": "/not-opened/1"}},
    }}

    def forbidden(*args, **kwargs):
        raise AssertionError("readiness must not access ledger I/O, mmap or allocation")

    value.ledger = SimpleNamespace(get=forbidden, snapshot_entries=forbidden)
    value._import_remote_host_record = forbidden
    return value


@pytest.mark.parametrize("rank", [0, 1])
def test_foreign_tp_snapshot_is_ready_before_local_import(rank):
    value = manager(rank)
    before = copy.deepcopy(value._ledger_entries_cache)
    parent = SimpleNamespace(snapshot_id="req:3")
    # This is the readiness predicate used by TP rank0's bounded Host selector.
    assert value.snapshot_ready(parent)
    assert not value.host_ready
    assert value._ledger_entries_cache == before


@pytest.mark.parametrize("change", [
    {"state": "host_writing"}, {"state": "host_reserved"},
    {"state": "evicting"}, {"state": "failed"}, {"state": "consumed"},
    {"recovery_domain": 0}, {"recovery_domain": None},
    {"tp_size": 1}, {"rank_grants": {"0": {}}},
    {"recovery_owner": "p-group:unrelated"},
])
def test_foreign_readiness_rejects_incomplete_or_other_assignment(change):
    value = manager()
    value._ledger_entries_cache["req:3"].update(change)
    assert not value.snapshot_ready(SimpleNamespace(snapshot_id="req:3"))
    assert not value.host_ready


def test_missing_cached_entry_is_false_without_synchronous_ledger_scan():
    value = manager()
    value._ledger_entries_cache.clear()
    assert not value.snapshot_ready(SimpleNamespace(snapshot_id="req:3"))


@pytest.mark.parametrize("tp_size", [1, 2])
def test_existing_local_readiness_is_unchanged(tp_size):
    value = manager()
    value.tp_size = tp_size
    value.host_ready["req:3"] = {"existing": True}
    value._ledger_entries_cache.clear()
    assert value.snapshot_ready(SimpleNamespace(snapshot_id="req:3"))


def test_tp1_keeps_existing_discovery_path():
    value = manager()
    value.tp_size = 1
    entry = value._ledger_entries_cache["req:3"]
    entry["tp_size"] = 1
    entry["rank_grants"] = {"0": {}}
    assert not value.snapshot_ready(SimpleNamespace(snapshot_id="req:3"))


@pytest.mark.parametrize("rank", [0, 1])
def test_local_storage_record_does_not_override_foreign_recovery_assignment(rank):
    value = manager(rank)
    value.host_ready["req:3"] = {"existing": True}
    entry = value._ledger_entries_cache["req:3"]
    entry.update(p_owner=value.owner, recovery_domain=0)
    assert not value.snapshot_ready(SimpleNamespace(snapshot_id="req:3"))
    entry["recovery_domain"] = 1
    assert value.snapshot_ready(SimpleNamespace(snapshot_id="req:3"))
