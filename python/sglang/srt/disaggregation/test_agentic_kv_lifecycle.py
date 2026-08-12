import base64
import json
import os
import tempfile
import time
from dataclasses import replace

import pytest

from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    AgenticOutputKind,
    AgenticRequestMetadata,
    MooncakeSnapshotStore,
    RequestGeneration,
    SharedSnapshotEvictionController,
    SnapshotIndex,
    SnapshotEvictionController,
    SnapshotLifecycleError,
    SnapshotManifest,
    SnapshotNotReadyError,
    SnapshotState,
    _discard_shared_ledger_snapshot,
    expand_mha_page_keys,
    namespace_page_keys,
    unpack_agentic_extra_key,
)


def test_agentic_request_metadata_parses_and_classifies_tool_output():
    metadata = AgenticRequestMetadata.from_custom_params(
        {
            "agentic_request_id": "trajectory/with:delimiters",
            "agentic_generation": 3,
            "agentic_parent_generation": 2,
            "agentic_tool_type": "search",
            "agentic_tool_suffix_token_ids": [[10, 11], [99]],
            "agentic_terminal_marker_token_ids": [[7, 8]],
        }
    )
    assert metadata is not None
    assert metadata.current.generation == 3
    assert metadata.parent.generation == 2
    assert metadata.classify_output([1, 10, 11]) is AgenticOutputKind.TOOL
    assert metadata.classify_output([1, 7, 8]) is AgenticOutputKind.TERMINAL
    assert metadata.classify_output([1, 10]) is AgenticOutputKind.UNKNOWN
    assert metadata.is_tool_output([1, 10, 11])
    assert metadata.is_tool_output([1, 99])
    assert not metadata.is_tool_output([1, 10])
    assert not metadata.is_tool_output([1, 7, 8, 10, 11])


class FixedDecodeTokenizer:
    def __init__(self, text):
        self.text = text

    def decode(self, output_ids, **kwargs):
        assert kwargs["skip_special_tokens"] is False
        assert kwargs["clean_up_tokenization_spaces"] is False
        return self.text


@pytest.mark.parametrize(
    ("decoded", "expected"),
    [
        ("任意工具结果 🔧</tool_call>", True),
        ("任意工具结果 🔧</tool_call>  \n", True),
        ("cafe\u0301", True),
        ("<answer>完成</answer> 🔧</tool_call>", False),
        ("普通最终回答", False),
    ],
)
def test_agentic_request_metadata_text_markers_are_tokenization_independent(
    decoded, expected
):
    metadata = AgenticRequestMetadata.from_custom_params(
        {
            "agentic_request_id": "unicode",
            "agentic_generation": 0,
            "agentic_tool_suffix_strings": ["🔧</tool_call>", "café"],
            "agentic_terminal_marker_strings": ["<answer>"],
            # Deliberately unrelated ids prove classification uses decoded text.
            "agentic_tool_suffix_token_ids": [[999]],
        }
    )
    assert metadata.is_tool_output([1, 2, 3], FixedDecodeTokenizer(decoded)) is expected


@pytest.mark.parametrize(
    ("decoded", "expected"),
    [
        ("<function=search>query</function>", AgenticOutputKind.TOOL),
        ("<function=finish>answer</function>", AgenticOutputKind.TERMINAL),
        ("No function call was detected", AgenticOutputKind.UNKNOWN),
    ],
)
def test_agentic_request_metadata_preserves_unknown_repair_turns(decoded, expected):
    metadata = AgenticRequestMetadata.from_custom_params(
        {
            "agentic_request_id": "repair",
            "agentic_generation": 1,
            "agentic_tool_suffix_strings": ["</function>"],
            "agentic_terminal_marker_strings": ["<function=finish>"],
        }
    )
    assert metadata.classify_output(
        [1, 2, 3], FixedDecodeTokenizer(decoded)
    ) is expected


def test_agentic_request_metadata_rejects_non_monotonic_parent():
    with pytest.raises(ValueError, match="must precede"):
        AgenticRequestMetadata.from_custom_params(
            {
                "agentic_request_id": "r",
                "agentic_generation": 1,
                "agentic_parent_generation": 1,
            }
        )


def test_unpack_agentic_extra_key_restores_stable_key_and_unicode_metadata():
    stable = "agentic-v1:请求/with:specials:g2"
    params = {
        "agentic_request_id": "请求/with:specials",
        "agentic_generation": 2,
        "agentic_parent_generation": 1,
        "agentic_tool_suffix_strings": ["🔧</tool_call>"],
    }

    def encode(value):
        raw = value if isinstance(value, bytes) else value.encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    envelope = "agentic-v1e:" + encode(stable) + ":" + encode(
        json.dumps(params, ensure_ascii=False, separators=(",", ":")).encode()
    )
    restored_key, restored_params = unpack_agentic_extra_key(envelope)
    assert restored_key == stable
    assert restored_params == params


@pytest.mark.parametrize(
    "envelope",
    [
        "agentic-v1e:not-base64:also-not-base64",
        "agentic-v1e:" + "a" * 33000,
    ],
)
def test_unpack_agentic_extra_key_rejects_malformed_input(envelope):
    with pytest.raises(ValueError, match="agentic"):
        unpack_agentic_extra_key(envelope)


class FakeMooncakeStore:
    def __init__(self):
        self.objects = {}
        self.leased = set()
        self.events = []

    def put(self, key, value):
        self.events.append(("put", key))
        if key in self.objects:
            return -1
        self.objects[key] = bytes(value)
        return 0

    def upsert(self, key, value):
        self.events.append(("upsert", key))
        self.objects[key] = bytes(value)
        return 0

    def get(self, key):
        return self.objects.get(key, b"")

    def is_exist(self, key):
        return int(key in self.objects)

    def batch_is_exist(self, keys):
        return [self.is_exist(key) for key in keys]

    def batch_remove(self, keys, force=False):
        self.events.append(("batch_remove", tuple(keys), force))
        result = []
        for key in keys:
            if key in self.leased and not force:
                result.append(-706)
            elif key in self.objects:
                del self.objects[key]
                result.append(0)
            else:
                result.append(-2)
        return result

    def remove(self, key, force=False):
        self.events.append(("remove", key, force))
        if key in self.leased and not force:
            return -706
        if key in self.objects:
            del self.objects[key]
            return 0
        return -2


def make_manifest(state=SnapshotState.MOONCAKE_READY, **kwargs):
    return SnapshotManifest(
        request=RequestGeneration(kwargs.pop("request_id", "req-a"), kwargs.pop("generation", 2)),
        page_keys=tuple(kwargs.pop("page_keys", ("page-k", "page-v"))),
        token_count=kwargs.pop("token_count", 128),
        byte_size=kwargs.pop("byte_size", 1024),
        state=state,
        created_at=kwargs.pop("created_at", 10.0),
        updated_at=kwargs.pop("updated_at", 10.0),
        tool_type=kwargs.pop("tool_type", "search"),
        tool_started_at=kwargs.pop("tool_started_at", 20.0),
        **kwargs,
    )


def test_manifest_roundtrip_and_transition_guards():
    manifest = make_manifest()
    assert SnapshotManifest.from_bytes(manifest.to_bytes()) == manifest
    loading = manifest.transition(SnapshotState.P_LOADING)
    assert loading.transition(SnapshotState.P_HOST).state is SnapshotState.P_HOST
    with pytest.raises(SnapshotLifecycleError):
        manifest.transition(SnapshotState.TO_DECODE)


def test_direct_ready_can_be_confirmed_final_by_application():
    manifest = make_manifest(
        state=SnapshotState.DIRECT_READY,
        direct_bootstrap_addr="127.0.0.1:45501",
        direct_room=123,
        token_digest="abc",
    )
    assert manifest.transition(SnapshotState.FINAL).state is SnapshotState.FINAL


def test_publish_uses_offloading_manifest_then_ready_commit_marker():
    raw = FakeMooncakeStore()
    snapshot_store = MooncakeSnapshotStore(raw)
    manifest = make_manifest(state=SnapshotState.OFFLOADING)
    snapshot_store.begin_publish(manifest)

    with pytest.raises(SnapshotLifecycleError, match="incomplete snapshot"):
        snapshot_store.commit_publish(manifest.request)

    for key in manifest.page_keys:
        raw.objects[key] = b"page"
    ready = snapshot_store.commit_publish(manifest.request)
    assert ready.state is SnapshotState.MOONCAKE_READY
    assert snapshot_store.load(manifest.request) == ready


def test_failed_publish_removes_partial_pages_but_keeps_failure_marker():
    raw = FakeMooncakeStore()
    snapshot_store = MooncakeSnapshotStore(raw)
    manifest = make_manifest(state=SnapshotState.OFFLOADING)
    snapshot_store.begin_publish(manifest)
    raw.objects[manifest.page_keys[0]] = b"partial"
    result = snapshot_store.fail_publish(manifest)
    assert result.removed
    assert raw.is_exist(manifest.page_keys[0]) == 0
    failed = snapshot_store.load(manifest.request, require_ready=False)
    assert failed.state is SnapshotState.FAILED


def test_failure_marker_has_no_fake_pages_and_is_idempotent():
    raw = FakeMooncakeStore()
    snapshot_store = MooncakeSnapshotStore(raw)
    request = RequestGeneration("no-space", 4)
    failed = snapshot_store.publish_failure(
        request, reason="d_host_allocation_failed", tool_type="search"
    )
    assert failed.page_keys == ()
    assert failed.failure_reason == "d_host_allocation_failed"
    assert snapshot_store.publish_failure(
        request, reason="retry", tool_type="search"
    ) == failed


def test_claim_hides_snapshot_from_evictors_and_requires_matching_ack():
    raw = FakeMooncakeStore()
    snapshot_store = MooncakeSnapshotStore(raw)
    offloading = make_manifest(state=SnapshotState.OFFLOADING)
    for key in offloading.page_keys:
        raw.objects[key] = b"page"
    snapshot_store.begin_publish(offloading)
    snapshot_store.commit_publish(offloading.request)

    loading = snapshot_store.claim_for_load(offloading.request, "p-worker-0")
    assert loading.state is SnapshotState.P_LOADING
    with pytest.raises(SnapshotNotReadyError, match="already claimed"):
        snapshot_store.claim_for_load(offloading.request, "p-worker-1")
    with pytest.raises(SnapshotNotReadyError, match="p_loading"):
        snapshot_store.load(offloading.request)
    with pytest.raises(SnapshotLifecycleError, match="invalid P host ACK"):
        snapshot_store.mark_p_host(loading, "wrong-worker")
    p_host = snapshot_store.mark_p_host(loading, "p-worker-0")
    p_gpu = snapshot_store.mark_p_gpu(p_host, "p-worker-0")
    assert p_gpu.state is SnapshotState.P_GPU


def test_direct_offer_claim_release_and_complete_are_single_consumer():
    raw = FakeMooncakeStore()
    store = MooncakeSnapshotStore(raw)
    request = RequestGeneration("direct", 1)
    offer = SnapshotManifest(
        request=request,
        page_keys=(),
        token_count=128,
        byte_size=0,
        state=SnapshotState.DIRECT_READY,
        token_digest="abc",
        direct_bootstrap_addr="127.0.0.1:45501",
        direct_room=123,
    )
    store.publish_direct_offer(offer)
    loading = store.claim_direct(request, "p0")
    with pytest.raises(SnapshotNotReadyError):
        store.claim_direct(request, "p1")
    ready = store.release_direct_claim(loading, "p0")
    assert ready.state is SnapshotState.DIRECT_READY
    loading = store.claim_direct(request, "p1")
    consumed = store.complete_direct(loading, "p1")
    assert consumed.state is SnapshotState.CONSUMED
    assert raw.is_exist(request.claim_key) == 0


def test_direct_fallback_does_not_overwrite_an_active_p_claim():
    raw = FakeMooncakeStore()
    store = MooncakeSnapshotStore(raw)
    request = RequestGeneration("direct-race", 3)
    offer = SnapshotManifest(
        request=request,
        page_keys=(),
        token_count=128,
        byte_size=0,
        state=SnapshotState.DIRECT_READY,
        token_digest="abc",
        direct_bootstrap_addr="127.0.0.1:45501",
        direct_room=456,
    )
    store.publish_direct_offer(offer)

    loading = store.claim_direct(request, "p0")
    assert store.begin_slow_fallback(offer, owner_id="d0") is None
    assert store.load(request, require_ready=False).state is SnapshotState.DIRECT_LOADING

    ready = store.release_direct_claim(loading, "p0")
    fallback = store.begin_slow_fallback(ready, owner_id="d0")
    assert fallback is not None
    assert fallback.state is SnapshotState.SLOW_FALLBACK
    assert raw.is_exist(request.claim_key) == 0
    with pytest.raises(SnapshotNotReadyError):
        store.claim_direct(request, "late-p")


def test_direct_fallback_tolerates_transient_manifest_read_miss(monkeypatch):
    raw = FakeMooncakeStore()
    store = MooncakeSnapshotStore(raw)
    request = RequestGeneration("direct-read-after-write", 0)
    offer = SnapshotManifest(
        request=request,
        page_keys=(),
        token_count=128,
        byte_size=0,
        state=SnapshotState.DIRECT_READY,
        token_digest="abc",
        direct_bootstrap_addr="127.0.0.1:45501",
        direct_room=789,
    )
    store.publish_direct_offer(offer)
    original_get = raw.get
    missed = False

    def miss_manifest_once(key):
        nonlocal missed
        if key == request.manifest_key and not missed:
            missed = True
            return b""
        return original_get(key)

    monkeypatch.setattr(raw, "get", miss_manifest_once)
    fallback = store.begin_slow_fallback(offer, owner_id="d0")
    assert fallback is not None
    assert fallback.state is SnapshotState.SLOW_FALLBACK
    assert store.load(request, require_ready=False).state is SnapshotState.SLOW_FALLBACK


def test_direct_claim_retries_transient_illegal_client_without_losing_claim():
    class TransientUpsertStore(FakeMooncakeStore):
        def __init__(self):
            super().__init__()
            self.failures_left = 2

        def upsert(self, key, value):
            self.events.append(("upsert", key))
            if self.failures_left:
                self.failures_left -= 1
                return -601
            self.objects[key] = bytes(value)
            return 0

    raw = TransientUpsertStore()
    store = MooncakeSnapshotStore(raw)
    offer = make_manifest(
        state=SnapshotState.DIRECT_READY,
        request_id="transient-claim",
        generation=0,
        page_keys=(),
        token_digest="digest",
        direct_bootstrap_addr="127.0.0.1:45501",
        direct_room=100,
    )
    store.publish_direct_offer(offer)
    claimed = store.claim_direct(offer.request, "p0")
    assert claimed.state is SnapshotState.DIRECT_LOADING
    assert claimed.claim_id == "p0"
    assert raw.failures_left == 0


def test_claimed_update_accepts_ambiguous_upsert_that_already_committed():
    class AmbiguousSuccessStore(FakeMooncakeStore):
        def __init__(self):
            super().__init__()
            self.once = True

        def upsert(self, key, value):
            self.events.append(("upsert", key))
            self.objects[key] = bytes(value)
            if self.once:
                self.once = False
                return -601
            return 0

    raw = AmbiguousSuccessStore()
    store = MooncakeSnapshotStore(raw)
    offer = make_manifest(
        state=SnapshotState.DIRECT_READY,
        request_id="ambiguous-claim",
        generation=0,
        page_keys=(),
        token_digest="digest",
        direct_bootstrap_addr="127.0.0.1:45501",
        direct_room=101,
    )
    store.publish_direct_offer(offer)
    claimed = store.claim_direct(offer.request, "p0")
    assert claimed.state is SnapshotState.DIRECT_LOADING
    assert sum(event[0] == "upsert" for event in raw.events) == 1


def test_claimed_update_never_overwrites_an_unexpected_advanced_state():
    class AdvancedStateStore(FakeMooncakeStore):
        def upsert(self, key, value):
            self.events.append(("upsert", key))
            desired = SnapshotManifest.from_bytes(value)
            advanced = replace(
                desired, claim_id=None, state=SnapshotState.SLOW_FALLBACK
            )
            self.objects[key] = advanced.to_bytes()
            return -601

    raw = AdvancedStateStore()
    store = MooncakeSnapshotStore(raw)
    offer = make_manifest(
        state=SnapshotState.DIRECT_READY,
        request_id="advanced-claim",
        generation=0,
        page_keys=(),
        token_digest="digest",
        direct_bootstrap_addr="127.0.0.1:45501",
        direct_room=102,
    )
    store.publish_direct_offer(offer)
    with pytest.raises(SnapshotNotReadyError, match="advanced to slow_fallback"):
        store.claim_direct(offer.request, "p0")
    assert store.load(offer.request, require_ready=False).state is SnapshotState.SLOW_FALLBACK
    assert raw.is_exist(offer.request.claim_key) == 0


def test_consumed_snapshot_force_removes_stale_get_leases():
    raw = FakeMooncakeStore()
    snapshot_store = MooncakeSnapshotStore(raw)
    manifest = make_manifest(state=SnapshotState.P_GPU, claim_id="p-worker-0")
    for key in manifest.page_keys:
        raw.objects[key] = b"page"
    raw.objects[manifest.manifest_key] = manifest.to_bytes()
    raw.leased.add(manifest.page_keys[-1])

    result = snapshot_store.delete_snapshot(
        manifest, final_state=SnapshotState.CONSUMED
    )
    assert result.removed
    assert ("batch_remove", manifest.page_keys, True) in raw.events
    terminal = snapshot_store.load(manifest.request, require_ready=False)
    assert terminal.state is SnapshotState.CONSUMED
    assert raw.is_exist(manifest.manifest_key) == 1
    assert all(raw.is_exist(key) == 0 for key in manifest.page_keys)
    assert not snapshot_store.gc_terminal_manifest(
        terminal, retention_seconds=10.0, now=terminal.terminal_at + 9.0
    )
    assert snapshot_store.gc_terminal_manifest(
        terminal, retention_seconds=10.0, now=terminal.terminal_at + 10.0
    )
    assert raw.is_exist(manifest.manifest_key) == 0


def test_terminal_snapshot_is_removed_from_shared_admission_ledger(monkeypatch):
    with tempfile.NamedTemporaryFile(
        mode="w+", dir="/dev/shm", prefix="agentic-ledger-test-", delete=True
    ) as ledger_file:
        json.dump(
            {
                "version": 1,
                "reservations": {"snapshot:0": {"byte_size": 1}},
                "residents": {"snapshot:0": "encoded"},
            },
            ledger_file,
        )
        ledger_file.flush()
        monkeypatch.setenv("SGLANG_AGENTIC_KV_LEDGER_PATH", ledger_file.name)
        _discard_shared_ledger_snapshot("snapshot:0")
        ledger_file.seek(0)
        ledger = json.load(ledger_file)
        assert ledger["reservations"] == {}
        assert ledger["residents"] == {}
        assert "snapshot:0" in ledger["terminals"]


def test_abort_can_abandon_a_fully_loaded_host_snapshot():
    raw = FakeMooncakeStore()
    store = MooncakeSnapshotStore(raw)
    manifest = make_manifest(state=SnapshotState.P_HOST, claim_id="p0")
    raw.objects[manifest.manifest_key] = manifest.to_bytes()
    raw.objects[manifest.request.claim_key] = b"load:p0"
    for key in manifest.page_keys:
        raw.objects[key] = b"page"

    result = store.abandon_load(manifest, "p0")
    assert result.removed
    assert store.load(manifest.request, require_ready=False).state is SnapshotState.EVICTED
    assert raw.is_exist(manifest.request.claim_key) == 0


def test_stale_loading_recovery_keeps_lease_safe_then_retries():
    raw = FakeMooncakeStore()
    store = MooncakeSnapshotStore(raw)
    manifest = make_manifest(state=SnapshotState.P_LOADING, claim_id="dead-p")
    raw.objects[manifest.manifest_key] = manifest.to_bytes()
    raw.objects[manifest.request.claim_key] = b"load:dead-p"
    for key in manifest.page_keys:
        raw.objects[key] = b"page"
    raw.leased.add(manifest.page_keys[0])

    first = store.recover_stale(manifest)
    assert not first.removed
    pending = store.load(manifest.request, require_ready=False)
    assert pending.state is SnapshotState.DELETE_PENDING
    raw.leased.clear()
    second = store.recover_stale(pending)
    assert second.removed
    assert store.load(manifest.request, require_ready=False).state is SnapshotState.EVICTED


def test_eviction_selects_complete_snapshots_by_requested_cost():
    index = SnapshotIndex()
    long_wait = make_manifest(
        request_id="long",
        byte_size=100,
        tool_type="browser",
        tool_started_at=90.0,
    )
    large_but_ready = make_manifest(
        request_id="large",
        byte_size=500,
        tool_type="code",
        tool_started_at=99.0,
    )
    p_host = make_manifest(
        request_id="not-evictable",
        state=SnapshotState.P_HOST,
        byte_size=10_000,
    )
    for manifest in (long_wait, large_but_ready, p_host):
        index.upsert(manifest)

    selected = index.select_evictions(
        50,
        now=100.0,
        expected_tool_seconds={"browser": 100.0, "code": 2.0},
    )
    assert [manifest.snapshot_id for manifest in selected] == [long_wait.snapshot_id]
    assert index.byte_size == 600


def test_eviction_controller_removes_whole_snapshot_before_reserving():
    raw = FakeMooncakeStore()
    store = MooncakeSnapshotStore(raw)
    controller = SnapshotEvictionController(
        store,
        capacity_bytes=1000,
        high_watermark=0.9,
        expected_tool_seconds={"search": 100.0},
    )
    old = make_manifest(request_id="old", byte_size=700)
    raw.objects[old.manifest_key] = old.to_bytes()
    for key in old.page_keys:
        raw.objects[key] = b"page"
    controller.index.upsert(old)

    assert controller.reserve(300)
    terminal = store.load(old.request, require_ready=False)
    assert terminal.state is SnapshotState.EVICTED
    assert all(raw.is_exist(key) == 0 for key in old.page_keys)


def test_eviction_controller_retries_delete_pending_after_lease_drains():
    raw = FakeMooncakeStore()
    store = MooncakeSnapshotStore(raw)
    controller = SnapshotEvictionController(
        store, capacity_bytes=1000, high_watermark=0.9
    )
    old = make_manifest(request_id="leased-old", byte_size=700)
    raw.objects[old.manifest_key] = old.to_bytes()
    for key in old.page_keys:
        raw.objects[key] = b"page"
    raw.leased.add(old.page_keys[-1])
    controller.index.upsert(old)

    assert not controller.reserve(300)
    assert store.load(old.request, require_ready=False).state is SnapshotState.DELETE_PENDING
    raw.leased.clear()
    assert controller.reserve(300)
    assert store.load(old.request, require_ready=False).state is SnapshotState.EVICTED


def test_shared_controller_accounts_reservations_across_d_processes():
    raw = FakeMooncakeStore()
    store = MooncakeSnapshotStore(raw)
    fd, ledger_path = tempfile.mkstemp(prefix="agentic-kv-test-", dir="/dev/shm")
    os.close(fd)
    try:
        d0 = SharedSnapshotEvictionController(
            store, ledger_path=ledger_path, capacity_bytes=1000, high_watermark=0.9
        )
        d1 = SharedSnapshotEvictionController(
            store, ledger_path=ledger_path, capacity_bytes=1000, high_watermark=0.9
        )
        first = make_manifest(
            request_id="d0-pending", state=SnapshotState.OFFLOADING, byte_size=600
        )
        second = make_manifest(
            request_id="d1-pending", state=SnapshotState.OFFLOADING, byte_size=400
        )
        assert d0.reserve(first)
        assert not d1.reserve(second)
        d0.cancel(first)
        assert d1.reserve(second)
    finally:
        os.unlink(ledger_path)


def test_shared_commit_does_not_resurrect_snapshot_consumed_by_fast_p(monkeypatch):
    raw = FakeMooncakeStore()
    store = MooncakeSnapshotStore(raw)
    fd, ledger_path = tempfile.mkstemp(prefix="agentic-kv-test-", dir="/dev/shm")
    os.close(fd)
    try:
        controller = SharedSnapshotEvictionController(
            store, ledger_path=ledger_path, capacity_bytes=1000
        )
        pending = make_manifest(
            request_id="commit-race",
            state=SnapshotState.OFFLOADING,
            byte_size=100,
        )
        ready = make_manifest(request_id="commit-race", byte_size=100)
        assert controller.reserve(pending)
        raw.objects[ready.manifest_key] = ready.to_bytes()
        monkeypatch.setenv("SGLANG_AGENTIC_KV_LEDGER_PATH", ledger_path)
        _discard_shared_ledger_snapshot(ready.snapshot_id)
        controller.commit(ready)
        with open(ledger_path, encoding="utf-8") as ledger_file:
            ledger = json.load(ledger_file)
        assert ledger["reservations"] == {}
        assert ledger["residents"] == {}
        assert ready.snapshot_id in ledger["terminals"]
    finally:
        os.unlink(ledger_path)


def test_shared_controller_evicts_whole_snapshot_owned_by_another_d(monkeypatch):
    raw = FakeMooncakeStore()
    store = MooncakeSnapshotStore(raw)
    fd, ledger_path = tempfile.mkstemp(prefix="agentic-kv-test-", dir="/dev/shm")
    os.close(fd)
    try:
        monkeypatch.setenv("SGLANG_AGENTIC_KV_LEDGER_PATH", ledger_path)
        d0 = SharedSnapshotEvictionController(
            store,
            ledger_path=ledger_path,
            capacity_bytes=1000,
            high_watermark=0.9,
            expected_tool_seconds={"search": 100.0},
        )
        d1 = SharedSnapshotEvictionController(
            store,
            ledger_path=ledger_path,
            capacity_bytes=1000,
            high_watermark=0.9,
            expected_tool_seconds={"search": 100.0},
        )
        old = make_manifest(request_id="d0-resident", byte_size=700)
        raw.objects[old.manifest_key] = old.to_bytes()
        for key in old.page_keys:
            raw.objects[key] = b"page"
        pending_old = make_manifest(
            request_id="d0-resident",
            state=SnapshotState.OFFLOADING,
            byte_size=700,
        )
        assert d0.reserve(pending_old)
        d0.commit(old)

        incoming = make_manifest(
            request_id="d1-new", state=SnapshotState.OFFLOADING, byte_size=300
        )
        assert d1.reserve(incoming)
        assert store.load(old.request, require_ready=False).state is SnapshotState.EVICTED
        assert all(raw.is_exist(key) == 0 for key in old.page_keys)
    finally:
        os.unlink(ledger_path)


def test_expand_mha_page_keys_is_complete_and_unique():
    assert expand_mha_page_keys(["p0", "p1"], ["0"]) == (
        "p0_0_k",
        "p0_0_v",
        "p1_0_k",
        "p1_0_v",
    )


def test_generation_namespace_prevents_old_delete_from_touching_new_pages():
    old = namespace_page_keys(RequestGeneration("req-a", 1), ["hash0", "hash1"])
    new = namespace_page_keys(RequestGeneration("req-a", 2), ["hash0", "hash1"])
    assert set(old).isdisjoint(new)
