import hashlib

import pytest

from sglang.srt.disaggregation.agentic_kv_lifecycle import token_ids_digest


def test_default_digest_unchanged(monkeypatch):
    monkeypatch.delenv("SGLANG_AGENTIC_KV_TOKEN_CONTENT_HASH", raising=False)
    tokens = [1, 42, 0, -1]
    expected = hashlib.sha256(b"".join(t.to_bytes(8, "little", signed=True) for t in tokens)).hexdigest()
    assert token_ids_digest(tokens) == expected
    assert token_ids_digest([1, 42]) != token_ids_digest([1, 43])


@pytest.mark.parametrize("setting", ["0", "false", "NO", " off "])
def test_disabled_never_reads_token_content(monkeypatch, setting):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_TOKEN_CONTENT_HASH", setting)

    class UnreadableTokens:
        def __iter__(self):
            raise AssertionError("disabled mode must not iterate tokens")

    def forbidden_hash(*args, **kwargs):
        raise AssertionError("disabled mode must not hash")

    monkeypatch.setattr(hashlib, "sha256", forbidden_hash)
    assert token_ids_digest(UnreadableTokens()) == "unchecked-token-content-v1"


def test_mixed_modes_do_not_compare_equal(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_TOKEN_CONTENT_HASH", "true")
    checked = token_ids_digest([11, 12])
    monkeypatch.setenv("SGLANG_AGENTIC_KV_TOKEN_CONTENT_HASH", "false")
    unchecked = token_ids_digest([11, 12])
    assert checked != unchecked
    monkeypatch.setenv("SGLANG_AGENTIC_KV_TOKEN_CONTENT_HASH", "true")
    assert token_ids_digest([11, 12]) != unchecked


def test_unknown_setting_does_not_silently_disable(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_TOKEN_CONTENT_HASH", "flase")
    assert token_ids_digest([1]) != token_ids_digest([2])


def test_hash_off_host_admission_still_rejects_short_parent(monkeypatch):
    from types import SimpleNamespace as NS
    from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration
    from sglang.srt.disaggregation.test_agentic_h2d_decoupling import manager
    from sglang.srt.managers.scheduler import AgenticPWorksetLeaseBroker

    monkeypatch.setenv("SGLANG_AGENTIC_KV_TOKEN_CONTENT_HASH", "false")
    m = manager(lanes=4)
    m.active, m.aborting = {}, {}
    m.tp_rank, m.tp_size, m.arena_domain, m.owner = 0, 1, 0, "p:test"
    m.workset_broker = AgenticPWorksetLeaseBroker(page_size=4)
    parent = RequestGeneration("short-parent", 0)
    record = dict(snapshot=NS(_materialized=object()), loading=False,
                  offer=dict(token_count=2, token_digest=token_ids_digest([11, 22]), byte_size=128))
    m.host_ready = {parent.snapshot_id: record}
    m._ledger_entries_cache = {parent.snapshot_id: dict(state="host_ready", p_owner=m.owner)}
    failures, released = [], []
    def fail(*args, **kwargs):
        failures.append(kwargs["reason"])
        return True
    m.ledger = NS(get=lambda sid: m._ledger_entries_cache.get(sid), transition=fail)
    m._release_record = lambda item: released.append(item)
    req = NS(rid="rid", origin_input_ids=[11], _agentic_kv_queue_class="slow")

    # Executes the production gate, not a duplicate predicate. False here
    # means continue via explicitly marked recompute, never bind short KV.
    assert m.gate_request(req, parent) is False
    assert failures == ["permanent_parent_digest_mismatch"]
    assert req._agentic_kv_fallback == "permanent_parent_digest_mismatch"
    assert released == [record]
    assert not m.loads and not m.host_ready
    assert m.h2d_physical_occupancy() == 0
    assert m.workset_broker.get(parent.snapshot_id) is None
