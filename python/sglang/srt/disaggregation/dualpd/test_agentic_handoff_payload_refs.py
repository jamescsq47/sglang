"""Attempt-scoped wire compaction must not change physical handoff semantics."""

from dataclasses import replace

import pytest

from sglang.srt.disaggregation.agentic_group_protocol import (
    CommandKind, GenerationKey, Owner, RankPhase, LinkParticipant,
)
from sglang.srt.disaggregation.agentic_group_transfer import (
    CallbackPathHandler, PreparedRankTransfer, RankLocalCommandExecutor,
)
from sglang.srt.disaggregation.agentic_transfer_queues import (
    FenceKind, PhysicalProgress, PhysicalState, TransferPath,
)
from sglang.srt.disaggregation.test_agentic_group_transfer import (
    command, make_queues, wait_phase,
)


TAIL = (
    CommandKind.HANDOFF, CommandKind.ACTIVATE,
    CommandKind.ISSUE_ACTIVATION_TICKET, CommandKind.FINALIZE,
)
PAYLOAD = {
    "transfer": {"target_generation": 4, "page_hashes": ["a", "b"]},
    "rank_results": [{"result": {"receipt": [1, 2]}}],
}


@pytest.fixture
def setup_rank(request):
    queues, executors = make_queues()
    acks, failures, commits = [], [], []
    handler = CallbackPathHandler(
        lambda _: PreparedRankTransfer("payload"),
        commit=lambda cmd, _, completion: commits.append(
            (cmd.payload, completion.fence)
        ),
    )
    rank = RankLocalCommandExecutor(
        participant=LinkParticipant("source", "d", 0),
        queues=queues, handlers={path: handler for path in TransferPath},
        emit_ack=acks.append, report_failure=failures.append,
        compact_handoff_payload=True,
    )
    source_owner = getattr(request, "param", Owner.D_GPU)
    rank.handle(replace(command(CommandKind.PREPARE, 1), source_owner=source_owner))
    rank.handle(replace(command(CommandKind.START, 2), source_owner=source_owner))
    handle = executors[TransferPath.D2P_DIRECT].wait_handle()
    assert commits == []
    executors[TransferPath.D2P_DIRECT].finish(
        handle, PhysicalProgress(PhysicalState.SUCCEEDED, FenceKind.DMA_COMPLETE)
    )
    wait_phase(acks, RankPhase.DMA_DONE)
    release = replace(command(CommandKind.RELEASE, 3), payload=PAYLOAD,
                      source_owner=source_owner)
    yield rank, release, acks, failures, commits, handler
    queues.close()


def ref(release, kind, seq):
    return replace(
        release, kind=kind, command_seq=seq,
        payload={"handoff_payload_ref": release.command_seq},
    )


@pytest.mark.parametrize("setup_rank", [Owner.D_GPU, Owner.D_HOST], indirect=True)
def test_refs_preserve_payload_isolation_fence_and_scheduler_release(setup_rank):
    rank, release, acks, failures, commits, handler = setup_rank
    # A callback may mutate nested legacy data: it must never poison the
    # private RELEASE baseline or a later command's independent payload.
    handler._prepare_handoff = lambda cmd, *_: cmd.payload["transfer"][
        "page_hashes"
    ].append("callback mutation")
    rank.handle(release)
    release.payload["rank_results"][0]["result"]["receipt"].append(999)
    for seq, kind in enumerate(TAIL, 4):
        wire = ref(release, kind, seq)
        full = rank.handle(wire)
        assert dict(full.payload) == PAYLOAD
        assert dict(wire.payload) == {"handoff_payload_ref": 3}
        if kind is CommandKind.HANDOFF and release.source_owner is Owner.D_HOST:
            # Host release remains immediate after the all-rank BOUND fence;
            # it must NOT wait for the native scheduler's activation ticket.
            assert len(commits) == 1
            assert commits[0][1] is FenceKind.DMA_COMPLETE
        if kind is CommandKind.ISSUE_ACTIVATION_TICKET and release.source_owner is Owner.D_GPU:
            assert commits == []
            rank.activate_staged(release.key, 1, release.lease_id, action="release")
            rank.confirm_scheduler_adopted(release.key, 1, release.lease_id)
            assert len(commits) == 1
            assert commits[0][1] is FenceKind.DMA_COMPLETE
        full.payload["transfer"]["page_hashes"].append("consumer mutation")
    assert rank.active_count == 0
    assert failures == []
    wait_phase(acks, RankPhase.FINALIZED)
    with pytest.raises(RuntimeError):
        rank.handle(ref(release, CommandKind.HANDOFF, 4))


@pytest.mark.parametrize("change", [
    {"payload": {"handoff_payload_ref": True}},
    {"payload": {"handoff_payload_ref": "3"}},
    {"payload": {"handoff_payload_ref": 2}},
    {"payload": {"handoff_payload_ref": 3, "transfer": {}}},
    {"attempt": 2}, {"lease_id": "another-lease"},
    {"key": GenerationKey("run", "request", 4)},
    {"key": GenerationKey("other-run", "request", 3)},
    {"source_owner": Owner.P_GPU}, {"target_owner": Owner.D_GPU},
    {"endpoint_groups": ("other",)}, {"command_seq": 5},
    {"kind": CommandKind.ACTIVATE}, {"kind": CommandKind.FINALIZE},
    {"kind": CommandKind.CANCEL},
])
def test_bad_reference_fails_without_release_or_ack(setup_rank, change):
    rank, release, acks, _, commits, _ = setup_rank
    rank.handle(release)
    previous = len(acks)
    with pytest.raises((ValueError, RuntimeError)):
        rank.handle(replace(ref(release, CommandKind.HANDOFF, 4), **change))
    assert len(acks) == previous
    assert commits == []
    # Rejection must not consume the valid reference or advance its sequence.
    assert rank.handle(ref(release, CommandKind.HANDOFF, 4)).payload == PAYLOAD


def test_missing_duplicate_and_cancelled_reference(setup_rank):
    rank, release, acks, _, commits, _ = setup_rank
    with pytest.raises(RuntimeError):
        rank.handle(ref(release, CommandKind.HANDOFF, 4))
    rank.handle(release)
    with pytest.raises(RuntimeError):
        rank.handle(release)
    rank.handle(ref(release, CommandKind.HANDOFF, 4))
    with pytest.raises(RuntimeError):
        rank.handle(ref(release, CommandKind.HANDOFF, 4))
    rank.handle(replace(release, kind=CommandKind.CANCEL, command_seq=5,
                        payload={"reason": "cancelled"}))
    wait_phase(acks, RankPhase.FAILED_DRAINED)
    with pytest.raises(RuntimeError):
        rank.handle(ref(release, CommandKind.ACTIVATE, 5))
    rank.handle(replace(release, kind=CommandKind.ABORT_FINALIZE,
                        command_seq=6, payload={}))
    assert rank.active_count == 0
    assert commits == []


def test_disabled_receiver_rejects_reference(setup_rank):
    rank, release, _, _, _, _ = setup_rank
    rank._compact_handoff_payload = False
    rank.handle(release)
    with pytest.raises(RuntimeError):
        rank.handle(ref(release, CommandKind.HANDOFF, 4))
