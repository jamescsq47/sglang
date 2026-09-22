"""CPU-only contract tests for the authoritative in-memory Host ledger."""

import ast
from concurrent.futures import ThreadPoolExecutor
import inspect

import pytest

from sglang.srt.disaggregation.agentic_host_control import InMemoryHostStagingLedger
from sglang.srt.disaggregation.agentic_host_staging import (
    HostStageState,
    SharedHostStagingLedger,
)


SID = "request:0"
OWNER = "source-host:d0"


def offer(ledger, rank=0, size=1):
    return ledger.offer({
        "snapshot_id": SID, "request_id": "request", "generation": 0,
        "token_count": 16, "token_digest": "digest", "byte_size": 128,
        "tp_rank": rank, "tp_size": size, "d_pid": 100 + rank,
        "arena_domain": 0, "source_host_node": "decode-node",
        "source_host_engine": "decode0",
    })


def ready(ledger, size=1):
    with ThreadPoolExecutor(max_workers=size) as executor:
        list(executor.map(lambda rank: offer(ledger, rank, size), range(size)))
    for rank in range(size):
        assert ledger.claim_rank(SID, OWNER, tp_rank=rank, tp_size=size)
        assert ledger.publish_rank_grant(SID, OWNER, {
            "kind": "shared_host_extent", "byte_size": 128, "offset": 128 * rank,
        }, tp_rank=rank, tp_size=size)
    for rank in range(size):
        assert ledger.complete_host_write(SID, 100 + rank, tp_rank=rank, tp_size=size)
        if rank + 1 < size:
            assert ledger.get(SID)["state"] == "host_writing"
    assert ledger.get(SID)["state"] == "host_ready"


def claim(ledger, rank, size, claim_id="attempt-1"):
    kwargs = dict(tp_rank=rank, tp_size=size, claim_id=claim_id)
    assert ledger.claim_d2p_recovery_rank(SID, OWNER, recovery_domain=0, **kwargs)
    assert ledger.attach_d2p_recovery_lease_rank(
        SID, OWNER, lease_id=1000 + rank, **kwargs,
    )
    return dict(kwargs, lease_id=1000 + rank)


def age(ledger):
    def callback(entries):
        entries[SID]["updated_at"] = 0
        return None, True
    ledger._mutate(callback, event_snapshot_id=SID)


@pytest.mark.parametrize("size", [1, 2, 8])
def test_eviction_receipt_survives_prune_without_extents_or_generation_reuse(size):
    ledger = InMemoryHostStagingLedger(max_generations=1)
    ready(ledger, size)
    assert ledger.begin_host_eviction(SID, OWNER, tp_size=size, reason="source_d_host_pressure")
    for rank in range(size):
        assert ledger.complete_host_eviction_rank(SID, OWNER, tp_rank=rank, tp_size=size)
        if rank + 1 < size:
            assert ledger.get(SID)["state"] == "evicting"
            ledger.prune(0, 0)
            assert SID in ledger._entries
    assert ledger.get(SID)["state"] == "recompute_required"
    # Even a terminal receipt cannot erase physical source release tracking.
    ledger.prune(0, 0)
    assert SID in ledger._entries
    for rank in range(size):
        assert ledger.complete_source_host_release_rank(SID, OWNER, tp_rank=rank, tp_size=size)
    before = ledger.get(SID)["_event_revision"]
    ledger.prune(0, 0)
    receipt = ledger.get(SID)
    assert receipt["state"] == "recompute_required"
    assert receipt["reason"] == "source_d_host_pressure"
    assert receipt["terminal_receipt"] and receipt["_event_revision"] > before
    assert not {"grants", "offers", "recovery_claims", "arena_path", "byte_size"} & receipt.keys()
    assert SID not in ledger._entries and ledger._retired == {SID}
    assert ledger.snapshot_entries()[SID] == receipt
    assert ledger.drain_changes()[-1]["entry"] == receipt
    receipt["reason"] = "caller mutation"
    assert ledger.get(SID)["reason"] == "source_d_host_pressure"
    ledger.prune(0, 0)
    assert ledger.get(SID)["terminal_receipt"]
    assert ledger.drain_changes() == []
    with pytest.raises(ValueError, match="retired request-generation"):
        offer(ledger, 0, size)
    with pytest.raises(RuntimeError, match="generation capacity"):
        ledger.offer({"snapshot_id": "new:0", "tp_rank": 0, "tp_size": 1,
                      "token_count": 16, "byte_size": 128})

    # A P request submitted before the eviction still reads the terminal
    # receipt, rather than waiting forever for a now-pruned Host snapshot.
    from types import SimpleNamespace
    from sglang.srt.disaggregation.agentic_host_staging import AgenticPHostStagingManager
    from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration
    manager = object.__new__(AgenticPHostStagingManager)
    manager.ledger, manager.host_ready = ledger, {}
    request = SimpleNamespace(rid="already-submitted")
    assert manager._prepare_host_restore(request, RequestGeneration("request", 0)) is False
    assert request._agentic_kv_fallback == "shared_host_evicted"


@pytest.mark.parametrize("size", [1, 2, 8])
def test_all_rank_ownership_and_delayed_handoff_retained(size):
    ledger = InMemoryHostStagingLedger()
    ready(ledger, size)
    claims = [claim(ledger, rank, size) for rank in range(size)]
    for kwargs in claims:
        assert ledger.mark_d2p_recovery_phase_rank(SID, OWNER, phase="io_inflight", **kwargs)
    for rank in range(size):
        assert ledger.complete_d2p_host_load_rank(SID, OWNER, tp_rank=rank, tp_size=size)
        if rank + 1 < size:
            assert ledger.get(SID)["state"] == "h2d_loading"
    for rank in range(size):
        assert ledger.complete_host_bind_rank(SID, OWNER, tp_rank=rank, tp_size=size)
        if rank + 1 < size:
            assert ledger.get(SID)["state"] == "hbm_ready"
    for rank in range(size):
        assert ledger.complete_source_host_release_rank(
            SID, OWNER, tp_rank=rank, tp_size=size,
        )
    age(ledger)
    ledger.prune(0, 0)
    assert ledger.get(SID)["state"] == "consumed"
    for rank, kwargs in enumerate(claims):
        assert ledger.mark_d2p_recovery_phase_rank(SID, OWNER, phase="handed", **kwargs)
        # Duplicate reports are idempotent; stale claim/lease cannot advance.
        assert ledger.mark_d2p_recovery_phase_rank(SID, OWNER, phase="handed", **kwargs)
        assert not ledger.mark_d2p_recovery_phase_rank(
            SID, OWNER, phase="handed", **dict(kwargs, claim_id="old-attempt"),
        )
        assert not ledger.mark_d2p_recovery_phase_rank(
            SID, OWNER, phase="handed", **dict(kwargs, lease_id=9999),
        )
        age(ledger)
        last_revision = ledger.get(SID)["_event_revision"]
        ledger.prune(0, 0)
        if rank + 1 < size:
            assert ledger.get(SID) is not None
    assert ledger.get(SID) is None
    tombstone = ledger.drain_changes()[-1]
    assert tombstone["entry"] is None
    assert tombstone["revision"] > last_revision
    # Delayed rank offers must not resurrect a consumed/freed generation,
    # even after its detailed terminal metadata has been pruned.
    for rank in range(size):
        with pytest.raises(ValueError, match="retired request-generation"):
            offer(ledger, rank, size)
    assert ledger.get(SID) is None
    assert ledger.drain_changes() == []


@pytest.mark.parametrize("size", [1, 2, 8])
def test_pin_cancellation_and_eviction_are_mutually_exclusive(size):
    ledger = InMemoryHostStagingLedger()
    ready(ledger, size)
    kwargs = [claim(ledger, rank, size) for rank in range(size)]
    assert not ledger.begin_host_eviction(SID, OWNER, tp_size=size, reason="pressure")
    assert not ledger.cancel_d2p_recovery_rank(
        SID, OWNER, **dict(kwargs[0], claim_id="stale"),
    )
    for rank, item in enumerate(kwargs):
        assert ledger.cancel_d2p_recovery_rank(SID, OWNER, **item)
        if rank + 1 < size:
            assert ledger.get(SID)["state"] == "h2d_loading"
    assert ledger.get(SID)["state"] == "host_ready"
    assert ledger.begin_host_eviction(SID, OWNER, tp_size=size, reason="pressure")
    for rank in range(size):
        assert not ledger.claim_d2p_recovery_rank(
            SID, OWNER, tp_rank=rank, tp_size=size, claim_id="attempt-2", recovery_domain=0,
        )
        assert ledger.complete_host_eviction_rank(SID, OWNER, tp_rank=rank, tp_size=size)
        if rank + 1 < size:
            assert ledger.get(SID)["state"] == "evicting"
    assert ledger.get(SID)["state"] == "recompute_required"


def test_inflight_lease_is_never_cancelled_or_aged_out():
    ledger = InMemoryHostStagingLedger()
    ready(ledger)
    kwargs = claim(ledger, 0, 1)
    assert ledger.mark_d2p_recovery_phase_rank(SID, OWNER, phase="io_inflight", **kwargs)
    assert not ledger.cancel_d2p_recovery_rank(SID, OWNER, **kwargs)
    age(ledger)
    ledger.prune(0, 0)
    assert ledger.get(SID)["recovery_claims"]["0"]["phase"] == "io_inflight"


@pytest.mark.parametrize("size", [1, 2, 8])
def test_duplicate_attach_does_not_downgrade_inflight_or_enable_cancel(size):
    ledger = InMemoryHostStagingLedger()
    ready(ledger, size)
    for rank in range(size):
        kwargs = claim(ledger, rank, size)
        assert ledger.mark_d2p_recovery_phase_rank(SID, OWNER, phase="io_inflight", **kwargs)
        previous = ledger.get(SID)
        assert ledger.attach_d2p_recovery_lease_rank(SID, OWNER, **kwargs)
        assert ledger.get(SID) == previous
        assert not ledger.cancel_d2p_recovery_rank(SID, OWNER, **kwargs)
        assert ledger.get(SID)["recovery_claims"][str(rank)]["phase"] == "io_inflight"


@pytest.mark.parametrize("size", [1, 2, 8])
def test_guarded_recovery_events_validate_order_and_all_rank_completion(size):
    ledger = InMemoryHostStagingLedger()
    ready(ledger, size)
    commands = [dict(claim(ledger, rank, size), remote_read_epoch=1) for rank in range(size)]
    for command in commands:
        assert not ledger.apply_recovery_event(SID, OWNER, event="loaded", **command)
        assert ledger.mark_d2p_recovery_phase_rank(
            SID, OWNER, phase="io_inflight", **{k: v for k, v in command.items() if k != "remote_read_epoch"},
        )
        assert not ledger.apply_recovery_event(SID, OWNER, event="bound", **command)
        assert not ledger.apply_recovery_event(SID, OWNER, event="handed", **command)
        assert ledger.apply_recovery_event(SID, OWNER, event="loaded", **command)
    assert ledger.get(SID)["state"] == "hbm_ready"
    for command in commands:
        assert ledger.apply_recovery_event(SID, OWNER, event="bound", **command)
    assert ledger.get(SID)["state"] == "consumed"
    for command in commands:
        assert ledger.apply_recovery_event(SID, OWNER, event="handed", **command)
        before = ledger.get(SID)
        for event in ("loaded", "bound", "handed"):
            assert ledger.apply_recovery_event(SID, OWNER, event=event, **command)
        assert ledger.get(SID) == before


@pytest.mark.parametrize("size", [1, 2, 8])
def test_queued_old_attempt_same_p_cannot_complete_new_recovery(size):
    ledger = InMemoryHostStagingLedger()
    ready(ledger, size)
    old = [claim(ledger, rank, size) for rank in range(size)]
    for command in old:
        assert ledger.mark_d2p_recovery_phase_rank(SID, OWNER, phase="io_inflight", **command)
    assert ledger.request_d2p_retry(SID, OWNER, reason="physical read fenced, workset retired")
    # Simulate the existing worker's all-rank drained/lease-retired receipts.
    for rank in range(size):
        assert ledger.complete_d2p_retry_rank(
            SID, OWNER, tp_rank=rank, tp_size=size, remote_read_epoch=1,
        )
    for rank in range(size):
        current = claim(ledger, rank, size, claim_id="attempt-2")
        assert ledger.mark_d2p_recovery_phase_rank(SID, OWNER, phase="io_inflight", **current)
    assert ledger.get(SID)["remote_read_epoch"] == 2
    before = ledger.get(SID)
    for command in old:
        for event in ("loaded", "bound", "handed"):
            assert not ledger.apply_recovery_event(
                SID, OWNER, remote_read_epoch=1, event=event, **command,
            )
            # Even if a delayed transport mistakes the new claim ID, the old
            # remote epoch cannot acknowledge the new allocation attempt.
            assert not ledger.apply_recovery_event(
                SID, OWNER, remote_read_epoch=1, event=event,
                **dict(command, claim_id="attempt-2"),
            )
    assert ledger.get(SID) == before


@pytest.mark.parametrize("replacement", [
    {"tp_rank": -1}, {"tp_rank": 1}, {"tp_size": 2}, {"tp_rank": True},
    {"lease_id": 99}, {"lease_id": None}, {"claim_id": "stale"},
    {"remote_read_epoch": 0}, {"remote_read_epoch": 2},
])
def test_guarded_recovery_event_rejects_wrong_identity(replacement):
    ledger = InMemoryHostStagingLedger()
    ready(ledger)
    command = claim(ledger, 0, 1)
    assert ledger.mark_d2p_recovery_phase_rank(SID, OWNER, phase="io_inflight", **command)
    command["remote_read_epoch"] = 1
    before = ledger.get(SID)
    assert not ledger.apply_recovery_event(
        SID, OWNER, event="loaded", **dict(command, **replacement),
    )
    assert not ledger.apply_recovery_event(SID, "another-P", event="loaded", **command)
    assert ledger.get(SID) == before


def test_consumed_requires_all_source_release_receipts():
    ledger = InMemoryHostStagingLedger()
    ready(ledger, 2)
    assert ledger.complete_host_load_rank(SID, OWNER, tp_rank=0, tp_size=2)
    assert ledger.complete_host_load_rank(SID, OWNER, tp_rank=1, tp_size=2)
    assert ledger.complete_source_host_release_rank(SID, OWNER, tp_rank=0, tp_size=2)
    age(ledger)
    ledger.prune(0, 0)
    assert ledger.get(SID)["state"] == "consumed"
    assert ledger.complete_source_host_release_rank(SID, OWNER, tp_rank=1, tp_size=2)
    age(ledger)
    ledger.prune(0, 0)
    assert ledger.get(SID) is None


def test_failure_record_with_lease_is_retained_even_after_host_release():
    ledger = InMemoryHostStagingLedger()
    ready(ledger)
    claim(ledger, 0, 1)
    # Fault injection: a terminal record must not erase still-attached HBM
    # ownership; explicit cancellation/drain is required, never a TTL.
    def failed(entries):
        entries[SID].update(state="failed", updated_at=0, source_host_released_ranks=[0])
        return None, True
    ledger._mutate(failed, event_snapshot_id=SID)
    ledger.prune(0, 0)
    assert ledger.get(SID)["recovery_claims"]["0"]["lease_id"] == 1000


@pytest.mark.parametrize("size", [1, 2, 8])
def test_failed_receipt_survives_only_after_all_source_fences(size):
    ledger = InMemoryHostStagingLedger()
    ready(ledger, size)
    def failed(entries):
        entries[SID].update(state="failed", updated_at=0, reason="read_failed")
        return None, True
    ledger._mutate(failed, event_snapshot_id=SID)
    ledger.prune(0, 0)
    assert SID in ledger._entries
    for rank in range(size):
        assert ledger.complete_source_host_release_rank(SID, OWNER, tp_rank=rank, tp_size=size)
    ledger.prune(0, 0)
    receipt = ledger.get(SID)
    assert SID not in ledger._entries
    assert receipt["state"] == "failed" and receipt["terminal_receipt"]
    assert receipt["reason"] == "read_failed"
    assert "rank_grants" not in receipt and "recovery_claims" not in receipt
    assert ledger.drain_changes()[-1]["entry"] == receipt


def test_input_output_and_change_records_are_detached():
    ledger = InMemoryHostStagingLedger()
    offered = offer(ledger)
    offered["rank_offers"]["0"]["byte_size"] = 777
    assert ledger.get(SID)["rank_offers"]["0"]["byte_size"] == 128
    snapshot = ledger.snapshot_entries()
    snapshot[SID]["rank_offers"].clear()
    fetched = ledger.get(SID)
    fetched["rank_offers"].clear()
    changes = ledger.drain_changes()
    changes[0]["entry"]["rank_offers"].clear()
    assert len(ledger.get(SID)["rank_offers"]) == 1
    assert ledger.claim(SID, OWNER)
    grant = {"kind": "shared_host_extent", "metadata": {"offsets": [5]}}
    ledger.publish_rank_grant(SID, OWNER, grant, tp_rank=0, tp_size=1)
    grant["metadata"]["offsets"][0] = 99
    assert ledger.get(SID)["rank_grants"]["0"]["metadata"]["offsets"] == [5]


def test_failed_and_rejected_mutations_do_not_leak_or_notify():
    ledger = InMemoryHostStagingLedger()
    offer(ledger)
    before = ledger.get(SID)
    ledger.drain_changes()
    def fail(entries):
        entries[SID]["rank_offers"].clear()
        raise RuntimeError("fault before commit")
    with pytest.raises(RuntimeError, match="fault before commit"):
        ledger._mutate(fail, event_snapshot_id=SID)
    def reject(entries):
        entries[SID]["rank_offers"].clear()
        return False, False
    assert ledger._mutate(reject, event_snapshot_id=SID) is False
    assert ledger.get(SID) == before
    assert ledger.drain_changes() == []
    with pytest.raises(ValueError, match="requires snapshot"):
        ledger._mutate(reject)


def test_document_transaction_failure_is_atomic():
    ledger = InMemoryHostStagingLedger()
    offer(ledger)
    before = ledger.get(SID)
    def fail(data):
        data["relays"]["bad"] = {"x": 1}
        data["entries"][SID]["rank_offers"].clear()
        raise RuntimeError("failed registry transaction")
    with pytest.raises(RuntimeError):
        ledger._mutate_document(fail, event_snapshot_id=SID)
    assert ledger.get(SID) == before
    assert ledger._relays == {}


def test_generation_limit_never_evicts_tombstones_or_blocks_live_progress():
    ledger = InMemoryHostStagingLedger(max_generations=1)
    ready(ledger)
    new_offer = dict(ledger.get(SID), snapshot_id="other:0", request_id="other")
    with pytest.raises(RuntimeError, match="generation capacity exhausted"):
        ledger.offer(new_offer)
    # Reaching the bound must not block a live record's completion/release.
    assert ledger.complete_host_load_rank(SID, OWNER, tp_rank=0, tp_size=1)
    assert ledger.complete_source_host_release_rank(SID, OWNER, tp_rank=0, tp_size=1)
    age(ledger)
    ledger.prune(0, 0)
    assert ledger.get(SID) is None
    with pytest.raises(ValueError, match="retired request-generation"):
        offer(ledger)
    with pytest.raises(RuntimeError, match="generation capacity exhausted"):
        ledger.offer(new_offer)
    assert ledger.get("other:0") is None
    assert ledger._retired == {SID}


def test_document_transaction_cannot_resurrect_retired_generation():
    ledger = InMemoryHostStagingLedger()
    ready(ledger)
    saved = ledger.get(SID)
    assert ledger.complete_host_load_rank(SID, OWNER, tp_rank=0, tp_size=1)
    assert ledger.complete_source_host_release_rank(SID, OWNER, tp_rank=0, tp_size=1)
    age(ledger)
    ledger.prune(0, 0)
    ledger.drain_changes()
    def resurrect(data):
        data["relays"]["should-not-commit"] = {}
        data["entries"][SID] = saved
        return True, True
    with pytest.raises(ValueError, match="retired request-generation"):
        ledger._mutate_document(resurrect)
    assert ledger.get(SID) is None
    assert ledger._relays == {}
    assert ledger.drain_changes() == []


def test_notifications_are_committed_coalesced_and_event_driven():
    ledger = InMemoryHostStagingLedger()
    offered = offer(ledger)
    assert offered["state"] == "offered"
    assert ledger.claim(SID, OWNER)
    events = ledger.drain_changes()
    assert len(events) == 1
    assert events[0]["entry"]["state"] == "host_reserved"
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(ledger.drain_changes, timeout=2)
        ledger.publish_rank_grant(SID, OWNER, {"kind": "shared_host_extent"}, tp_rank=0, tp_size=1)
        assert future.result(timeout=3)[0]["entry"] == ledger.get(SID)
    assert ledger.drain_changes(timeout=0) == []


def test_no_runtime_filesystem_operations(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Host control touched a filesystem")
    import os
    import fcntl
    with monkeypatch.context() as patch:
        patch.setattr("builtins.open", forbidden)
        patch.setattr(fcntl, "flock", forbidden)
        for name in ("open", "fdopen", "stat", "scandir", "mkdir", "makedirs", "unlink", "replace"):
            patch.setattr(os, name, forbidden)
        ledger = InMemoryHostStagingLedger()
        ready(ledger, 8)
        assert len(ledger.snapshot_entries()) == 1
        assert ledger.list_state(HostStageState.HOST_READY)
        assert ledger.drain_changes()
        ledger.prune(0, 0)
        assert ledger.get(SID) is not None
        with pytest.raises(NotImplementedError):
            ledger.claim_relay_job("legacy", 1)
        with pytest.raises(NotImplementedError):
            ledger.read_entry_event("not-a-file")


def test_every_file_backed_base_method_is_overridden():
    # Catch additions to the superclass before they accidentally introduce
    # runtime NFS into this backend. Parsing happens outside the no-FS test.
    source = ast.parse(inspect.getsource(SharedHostStagingLedger))
    filesystem_attributes = {
        "open", "fdopen", "scandir", "unlink", "replace", "makedirs", "flock",
    }
    for method in source.body[0].body:
        if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        filesystem = any(
            isinstance(node, ast.Call) and (
                isinstance(node.func, ast.Name) and node.func.id == "open"
                or isinstance(node.func, ast.Attribute) and node.func.attr in filesystem_attributes
            ) for node in ast.walk(method)
        )
        if filesystem:
            assert method.name in InMemoryHostStagingLedger.__dict__, method.name
