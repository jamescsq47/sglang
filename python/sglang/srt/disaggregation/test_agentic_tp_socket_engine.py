"""CPU regressions for endpoint-mode coordinator/executor engine boundaries."""

from types import SimpleNamespace

import pytest

from sglang.srt.disaggregation.agentic_kv_lifecycle import AgenticRequestMetadata
from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager as Manager,
)


def test_finished_shard_does_not_elect_on_decode_scheduler(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "enabled")
    metadata = AgenticRequestMetadata(
        request_id="generation", generation=1, tool_suffix_token_ids=((9,),)
    )
    retained = []

    def forbidden(*args, **kwargs):
        raise AssertionError("scheduler/follower must not perform producer RPC")

    for rank in range(8):
        manager = SimpleNamespace(
            tp_world_size=8,
            tp_rank=rank,
            page_size=4,
            agentic_early_claim_store=SimpleNamespace(
                claim_generation_producer=forbidden
            ),
            agentic_direct_runtime=object(),
            agentic_hostless=True,
            _publish_agentic_direct_candidate=lambda req, meta, tokens: retained.append(
                req.rid
            )
            or True,
        )
        req = SimpleNamespace(
            rid=f"shard{rank}",
            origin_input_ids=[1, 2, 3, 4],
            output_ids=[5, 9],
            tokenizer=None,
            finished_reason=None,
            finished=lambda: True,
        )
        assert Manager._offload_agentic_finished_snapshot(manager, req, metadata)
    assert len(retained) == 8


def test_producer_election_approval_and_duplicate_release_are_rank_zero_only():
    calls, releases = [], []
    request = SimpleNamespace(snapshot_id="generation:1")

    def make(rank, decision):
        manager = SimpleNamespace(
            tp_rank=rank,
            agentic_early_claim_store=SimpleNamespace(
                claim_generation_producer=lambda *a, **kw: calls.append(rank)
                or decision
            ),
            _retire_candidate_for_release=lambda *args: releases.append(args),
        )
        candidate = {
            "producer_approved": False,
            "req": SimpleNamespace(rid="rid"),
            "metadata": SimpleNamespace(current=request),
        }
        return manager, candidate

    follower, local = make(7, True)
    assert not Manager._agentic_resolve_generation_producer(follower, local)
    assert not calls and not releases
    leader, candidate = make(0, True)
    assert Manager._agentic_resolve_generation_producer(leader, candidate)
    assert candidate["producer_approved"] and calls == [0]
    assert Manager._agentic_resolve_generation_producer(leader, candidate)
    assert calls == [0]  # no repeated RPC on each progress step
    leader, duplicate = make(0, False)
    assert not Manager._agentic_resolve_generation_producer(leader, duplicate)
    assert len(releases) == 1 and not duplicate["producer_approved"]


def test_uncertain_producer_result_never_releases_or_starts_sender():
    releases = []

    def unavailable(*args, **kwargs):
        raise OSError("committed reply lost")

    manager = SimpleNamespace(
        tp_rank=0,
        agentic_early_claim_store=SimpleNamespace(
            claim_generation_producer=unavailable
        ),
        _retire_candidate_for_release=lambda *args: releases.append(args),
    )
    candidate = {
        "producer_approved": False,
        "req": SimpleNamespace(rid="rid"),
        "metadata": SimpleNamespace(current=object()),
    }
    with pytest.raises(OSError):
        Manager._agentic_resolve_generation_producer(manager, candidate)
    assert not releases and not candidate["producer_approved"]
    assert not Manager._progress_agentic_direct_candidate_setup(manager, candidate, 0)


def test_follower_abort_consumes_coordinator_receipt_not_peer_reports():
    reads, decisions = [], []

    def local_status(key, rank=None):
        reads.append(rank)
        assert rank is None, "follower cannot inspect other physical ranks"
        return 0

    mailbox = SimpleNamespace(
        bind_identity=lambda *args: None,
        local_status=local_status,
        receipt=lambda key: 1,
    )
    assert Manager._agentic_tp_abort_requested(
        SimpleNamespace(tp_rank=7, tp_world_size=8), mailbox, "gen"
    )
    assert reads == [None]
    mailbox.local_status = lambda key, rank: 2 if rank == 7 else 0
    mailbox.publish_receipt = lambda *args: decisions.append(args)
    assert Manager._agentic_tp_abort_requested(
        SimpleNamespace(tp_rank=0, tp_world_size=8), mailbox, "gen"
    )
    assert decisions == [("gen", 1)]


def test_p_abort_follower_reports_own_fence_then_waits_authoritative_claim_return():
    import time
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.disaggregation.agentic_kv_lifecycle import (
        RequestGeneration,
        SnapshotState,
    )
    from sglang.srt.disaggregation.base import KVPoll

    reports, drops = [], []
    request = RequestGeneration("abort", 1)
    loading = SimpleNamespace(state=SnapshotState.DIRECT_LOADING, claim_id="claim")
    ready = SimpleNamespace(state=SnapshotState.DIRECT_READY, claim_id=None)

    def peer_read(*args):
        raise AssertionError("follower cannot reduce peer fences")

    current = [loading]
    scheduler = SimpleNamespace(
        tp_size=8,
        tp_rank=7,
        agentic_tp_direct_abort_mailbox=SimpleNamespace(
            publish_local_progress=lambda *args: reports.append(args),
            local_status=peer_read,
        ),
        _agentic_drop_early_direct_receive=lambda *args, **kwargs: drops.append(args),
    )
    entry = SimpleNamespace(
        request=request,
        claim_id="claim",
        io_quiesced=True,
        direct_abort_marker_seen=True,
        direct_abort_fence_kind="unstarted",
        started_at=time.monotonic(),
        transport_poll=KVPoll.WaitingForInput,
    )
    store = SimpleNamespace(load=lambda *args, **kwargs: current[0])
    assert Scheduler._agentic_handle_unstarted_direct_abort(
        scheduler, entry, store, object(), KVPoll.WaitingForInput, 2
    )
    assert reports == [(request.snapshot_id, 1)] and not drops
    current[0] = ready  # only rank0's all-shard CAS publishes this
    assert Scheduler._agentic_handle_unstarted_direct_abort(
        scheduler, entry, store, object(), KVPoll.WaitingForInput, 2
    )
    assert len(drops) == 1
