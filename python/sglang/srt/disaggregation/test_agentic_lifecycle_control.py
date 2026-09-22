"""Real socket lifecycle fences and scheduler nonblocking completion tests."""

import ast
import builtins
import logging
import threading
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import __future__

import pytest

from sglang.srt.disaggregation.agentic_control_rpc import (
    ControlRPCClient,
    ControlRPCServer,
)
from sglang.srt.disaggregation.agentic_control_store import (
    BrokerRawStore,
    MemoryControlStore,
)
from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    MooncakeSnapshotStore,
    SnapshotState,
)
from sglang.srt.disaggregation.agentic_lifecycle_control import poll_lifecycle_call
from sglang.srt.disaggregation.test_agentic_kv_lifecycle import make_manifest


@pytest.fixture
def control(monkeypatch, tmp_path):
    server = ControlRPCServer("lifecycle", "secret")
    state = MemoryControlStore(server.publish)
    server.register_service("records", state.methods())
    client = ControlRPCClient(server.address, run_id="lifecycle", token="secret")
    client.wait_ready()
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "test")
    monkeypatch.setenv("SGLANG_PD_P_READY_DIR", str(tmp_path / "not-created"))
    monkeypatch.setattr(
        "sglang.srt.disaggregation.agentic_control_rpc.get_control_client",
        lambda: client,
    )
    try:
        yield state, BrokerRawStore(str(tmp_path / "metadata"))
    finally:
        client.close()
        server.close()


def offer(tp=1):
    return make_manifest(
        state=SnapshotState.DIRECT_READY,
        page_keys=(),
        direct_bootstrap_addr="10.0.0.1:45501",
        direct_room=123,
        token_digest="abc",
        tp_size=tp,
        kv_layout_hash="layout",
    )


@pytest.mark.parametrize("tp", [1, 2, 8])
def test_direct_lifecycle_and_tp_fences_without_files(control, monkeypatch, tp):
    _, raw = control
    store = MooncakeSnapshotStore(raw)
    manifest = offer(tp)
    # Initialize the push subscription before forbidding all file operations.
    from sglang.srt.disaggregation.agentic_lifecycle_control import lifecycle_records

    lifecycle_records()

    def forbidden(*a, **kw):
        raise AssertionError("runtime control touched a file")

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", forbidden)
        patch.setattr("os.open", forbidden)
        patch.setattr("os.unlink", forbidden)
        patch.setattr("os.makedirs", forbidden)
        patch.setattr("fcntl.flock", forbidden)
        store.publish_direct_offer(manifest)
        claimed = store.claim_direct(manifest.request, "P0:attempt")
        assert store.claim_direct(manifest.request, "P0:attempt") == claimed
        for rank in range(tp):
            received = store.complete_direct_rank(
                claimed, "P0:attempt", tp_rank=rank, tp_size=tp
            )
            assert received.state is (
                SnapshotState.P_RECEIVED
                if rank == tp - 1
                else SnapshotState.DIRECT_LOADING
            )
        result = store.commit_direct_bound(received, "P0:attempt")
        assert result.state is SnapshotState.CONSUMED
        assert store._local_claim_owner(manifest.request) is None


def test_old_owner_cannot_overwrite_new_claim_even_with_stale_push(
    control, monkeypatch
):
    _, raw = control
    store = MooncakeSnapshotStore(raw)
    manifest = offer()
    store.publish_direct_offer(manifest)
    old = store.claim_direct(manifest.request, "old")
    ready = store.release_direct_claim(old, "old")
    current = store.claim_direct(ready.request, "new")
    monkeypatch.setattr(store, "_require_local_claim_owner", lambda *a: None)
    with pytest.raises(RuntimeError, match="owner changed"):
        store._update_claimed_transition(
            old.transition(SnapshotState.P_RECEIVED),
            expected_states=(SnapshotState.DIRECT_LOADING,),
            owner_claim_id="old",
        )
    assert store.load(manifest.request, require_ready=False) == current
    store._release_local_claim(manifest.request, "old")
    assert store._local_claim_owner(manifest.request) == "new"


def test_raw_store_factory_selects_broker_without_directory_io(control, monkeypatch):
    from sglang.srt.disaggregation.agentic_host_staging import AgenticNodeLocalRawStore

    def forbidden(*args, **kwargs):
        raise AssertionError("metadata factory touched disk")

    monkeypatch.setattr("os.makedirs", forbidden)
    monkeypatch.setattr("os.open", forbidden)
    value = AgenticNodeLocalRawStore("/run/does-not-exist/metadata")
    assert isinstance(value, BrokerRawStore)
    assert value.put("key", b"payload") == 0
    assert value.get("key") == b"payload"


@pytest.mark.parametrize("tp", [1, 2, 8])
def test_slow_finalization_is_idempotent_and_file_free(control, monkeypatch, tp):
    _, raw = control
    store = MooncakeSnapshotStore(raw)
    manifest = offer(tp)
    store.publish_direct_offer(manifest)
    store.begin_slow_fallback(manifest, owner_id="D0")

    def forbidden(*a, **kw):
        raise AssertionError("slow finalization touched files")

    monkeypatch.setattr("os.open", forbidden)
    monkeypatch.setattr("os.path.exists", forbidden)
    assert store.complete_slow_fallback_group(manifest.request)
    assert store.complete_slow_fallback_group(manifest.request)
    assert (
        store.load(manifest.request, require_ready=False).state
        is SnapshotState.CONSUMED
    )


def test_background_lifecycle_retains_future_and_does_not_repeat():
    holder = SimpleNamespace()
    entered, release = threading.Event(), threading.Event()
    calls = []

    def operation():
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return "ack"

    try:
        start = time.monotonic()
        assert poll_lifecycle_call(holder, "phase", operation) == (False, None)
        assert time.monotonic() - start < 0.25
        assert entered.wait(1)
        for _ in range(10):
            assert poll_lifecycle_call(holder, "phase", operation) == (False, None)
        release.set()
        holder._agentic_lifecycle_futures["phase"].result(2)
        assert poll_lifecycle_call(holder, "phase", operation) == (True, "ack")
        assert calls == [1]
    finally:
        release.set()


def test_ambiguous_failure_keeps_phase_and_never_reexecutes():
    holder = SimpleNamespace()
    count = []

    def operation():
        count.append(1)
        raise ConnectionError("lost ACK")

    try:
        poll_lifecycle_call(holder, "phase", operation)
    except ConnectionError:
        pass
    with pytest.raises(ConnectionError):
        holder._agentic_lifecycle_futures["phase"].result(2)
    for _ in range(3):
        with pytest.raises(ConnectionError):
            poll_lifecycle_call(holder, "phase", operation)
    assert count == [1]


def test_scheduler_bind_waits_for_ack_not_only_pushed_consumed(monkeypatch):
    path = Path(__file__).parents[1] / "managers" / "scheduler.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_agentic_admit_early_direct_bind"
    )
    scope = {
        "SnapshotState": SnapshotState,
        "SnapshotNotReadyError": RuntimeError,
        "nullcontext": nullcontext,
        "time": time,
        "logger": logging.getLogger(__name__),
    }
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        scope,
    )
    callback = scope[method.name]
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "enabled")
    request = offer().request
    current = SimpleNamespace(state=SnapshotState.P_RECEIVED)
    entered, release = threading.Event(), threading.Event()

    def commit(*args):
        current.state = SnapshotState.CONSUMED
        entered.set()
        assert release.wait(5)

    store = SimpleNamespace(load=lambda *a, **kw: current, commit_direct_bound=commit)
    handed = []
    entry = SimpleNamespace(claim_id="P0", manifest=offer(), workset_lease=object())
    self = SimpleNamespace(
        _agentic_snapshot_store=lambda: store,
        agentic_p_workset_broker=SimpleNamespace(
            handoff_to_req=lambda *a: handed.append(1)
        ),
        agentic_early_direct_receives={request.snapshot_id: entry},
        agentic_early_direct_terminal={},
    )
    req = SimpleNamespace(rid="rid")
    try:
        assert callback(self, req, request, entry, tp_size=1, marker_store=None)
        assert entered.wait(1)
        assert callback(self, req, request, entry, tp_size=1, marker_store=None)
        assert not handed
        release.set()
        entry._agentic_lifecycle_futures["direct-bind"].result(2)
        assert not callback(self, req, request, entry, tp_size=1, marker_store=None)
        assert handed == [1]
    finally:
        release.set()


def test_scheduler_drop_only_signals_existing_transport_worker(monkeypatch):
    path = Path(__file__).parents[1] / "managers" / "scheduler.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_agentic_drop_early_direct_receive"
    )
    scope = {"threading": threading, "nullcontext": nullcontext}
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        scope,
    )
    monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "enabled")
    entry = SimpleNamespace(abort_requested=False, abort_release_claim=False)
    owner = SimpleNamespace(agentic_early_direct_progress_thread=object())
    scope[method.name](owner, entry, object(), release_claim=True, reason="abort")
    assert entry.abort_requested and entry.abort_release_claim
    assert entry.abort_reason == "abort"
