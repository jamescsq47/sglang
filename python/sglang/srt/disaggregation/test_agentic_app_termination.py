"""Application-authoritative termination must not discard a continuing parent."""

import pytest
from types import SimpleNamespace

from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    AgenticOutputKind,
    AgenticRequestMetadata,
    SnapshotState,
)


def metadata():
    return AgenticRequestMetadata(
        request_id="swe-terminal-regression", generation=26,
        tool_type="shell", tool_suffix_strings=("```",),
        terminal_marker_strings=("TASK_COMPLETE",),
        tool_suffix_token_ids=((10,),), terminal_marker_token_ids=((7, 8),),
    )


class Tokenizer:
    def __init__(self, text):
        self.text = text

    def decode(self, *args, **kwargs):
        return self.text


@pytest.mark.parametrize("text", [
    'The fix works. </think>\n```bash\necho "TASK_COMPLETE"\n```',
    'TASK_COMPLETE is what I will say later. </think>\n```bash\npytest\n```',
    '<think>TASK_COMPLETE',
    '</think>\nTASK_COMPLETE',
    'TASK_COMPLETE',
    '```bash\nls\n```',
    'No command',
])
def test_app_decision_never_short_circuited_by_engine_markers(monkeypatch, text):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_APP_OWNS_TERMINATION", "true")
    # Both token and text markers must be deferred, including a genuine final:
    # only the existing final ACK may make that lifecycle decision in this mode.
    assert metadata().classify_output([7, 8, 10], Tokenizer(text)) is AgenticOutputKind.UNKNOWN


def test_app_mode_does_not_require_decoding_or_tokenizer(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_APP_OWNS_TERMINATION", "1")
    assert metadata().classify_output([7, 8]) is AgenticOutputKind.UNKNOWN


@pytest.mark.parametrize("setting", [None, "false", "0"])
def test_native_marker_behavior_is_unchanged_when_opted_out(monkeypatch, setting):
    monkeypatch.delenv("SGLANG_AGENTIC_KV_APP_OWNS_TERMINATION", raising=False)
    if setting is not None:
        monkeypatch.setenv("SGLANG_AGENTIC_KV_APP_OWNS_TERMINATION", setting)
    assert metadata().classify_output([7, 8]) is AgenticOutputKind.TERMINAL
    assert metadata().classify_output([10]) is AgenticOutputKind.TOOL
    assert metadata().classify_output([1]) is AgenticOutputKind.UNKNOWN


@pytest.mark.parametrize(("enabled", "finish", "enters_snapshot_path"), [
    (True, "stop", True), (False, "stop", False),
    (True, "length", False), (True, "abort", False),
])
def test_real_offload_gate_preserves_continuations_but_not_truncation(
    monkeypatch, enabled, finish, enters_snapshot_path
):
    from sglang.srt.disaggregation.decode_kvcache_offload_manager import DecodeKVCacheOffloadManager

    monkeypatch.setenv("SGLANG_AGENTIC_KV_APP_OWNS_TERMINATION", str(enabled).lower())
    entered = []
    req = SimpleNamespace(
        finished=lambda: True, output_ids=[7, 8, 10], origin_input_ids=[],
        tokenizer=Tokenizer('</think>\n```bash\necho "TASK_COMPLETE"\n```'),
        finished_reason=SimpleNamespace(to_json=lambda: {"type": finish}),
    )
    # An intentionally sub-page fixture stops at snapshot construction. This
    # exercises the real production gate without GPU allocation or fake DMA.
    manager = SimpleNamespace(
        page_size=64,
        _publish_agentic_failure=lambda meta, reason: entered.append(reason),
    )
    assert not DecodeKVCacheOffloadManager._offload_agentic_finished_snapshot(manager, req, metadata())
    assert entered == (["empty_aligned_snapshot"] if enters_snapshot_path else [])


@pytest.mark.parametrize(("state", "staging", "sent", "releases"), [
    (SnapshotState.DIRECT_READY, False, False, True),
    (SnapshotState.DIRECT_LOADING, False, False, False),
    (SnapshotState.DIRECT_READY, True, False, False),
    (SnapshotState.DIRECT_READY, False, True, False),
    (SnapshotState.CONSUMED, False, False, False),
])
def test_final_ack_uses_existing_claim_and_transfer_guards(state, staging, sent, releases):
    from sglang.srt.disaggregation.decode_kvcache_offload_manager import DecodeKVCacheOffloadManager

    events = []
    meta = metadata()
    manifest = SimpleNamespace(snapshot_id=meta.current.snapshot_id, state=state)
    candidate = dict(metadata=meta, manifest=manifest, req=object(),
                     staging=staging, sent=sent, created_at=1.0)
    manager = SimpleNamespace(
        _agentic_direct_manifest=lambda *a, **kw: manifest,
        agentic_snapshot_store=SimpleNamespace(finalize_direct_offer=lambda *a, **kw: events.append("CAS") or manifest),
        _cleanup_agentic_direct_sender=lambda *a: events.append("sender"),
        _agentic_release_early_claim=lambda *a: events.append("claim"),
        _agentic_release_final_confirmation=lambda *a: events.append("ACK"),
        _retire_candidate_for_release=lambda *a: events.append("retire"),
    )
    assert DecodeKVCacheOffloadManager._agentic_complete_final_candidate(manager, candidate, 2.0) is releases
    assert events == (["CAS", "sender", "claim", "ACK", "retire"] if releases else [])
