import json
import pytest
from pathlib import Path

from sglang.srt.disaggregation import agentic_host_staging as staging


class _FakeMapping:
    def __init__(self, path: str):
        self.path = path
        self.byte_size = 1024
        self.waits = []

    def wait_prewarm(self, timeout):
        self.waits.append(timeout)


@pytest.mark.parametrize("role", ["prefill", "decode"])
@pytest.mark.parametrize("fails", [False, True])
def test_multinode_prewarm_only_local_source(monkeypatch, tmp_path, role, fails):
    _configure(monkeypatch, tmp_path)
    opened = []

    def open_mapping(path, device):
        assert path == "local-source"
        opened.append(path)
        if fails:
            raise RuntimeError("injected registration failure")
        return _FakeMapping(path)

    monkeypatch.setattr(staging, "_registered_host_arena", open_mapping)
    worker = staging.start_registered_host_arena_startup_prewarm(
        role=role, engine_id="engine", tp_rank=0, device="cuda:0",
        d2p_arena_path="remote-must-not-open", p2d_arena_path="remote-must-not-open",
        local_source_paths=["local-source"],
    )
    # No remote arena manifests exist: explicit source mode must not need them.
    assert not list((tmp_path / "complete").glob("*.json"))
    (tmp_path / "start").touch()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert opened == ["local-source"]
    assert len(list((tmp_path / "failed").glob("*.json"))) == int(fails)
    assert len(list((tmp_path / "complete").glob("*.json"))) == int(not fails)


def test_direct_only_decode_reports_zero_byte_prewarm(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    monkeypatch.setenv("SGLANG_AGENTIC_KV_HOST_STAGING", "false")
    def no_mapping(*args, **kwargs):
        pytest.fail("Direct-only D must not register any Host arena")
    monkeypatch.setattr(staging, "_registered_host_arena", no_mapping)
    worker = staging.start_registered_host_arena_startup_prewarm(
        role="decode", engine_id="engine", tp_rank=0, device="cuda:0",
        local_source_paths=[],
    )
    (tmp_path / "start").touch()
    worker.join(timeout=5)
    assert not worker.is_alive()
    record = json.loads(next((tmp_path / "complete").glob("*.json")).read_text())
    assert record["arena_count"] == 0
    assert record["registered_bytes"] == 0


@pytest.mark.parametrize("role,enabled", [("prefill", "false"), ("decode", "true")])
def test_empty_source_rejected_for_enabled_host(monkeypatch, tmp_path, role, enabled):
    _configure(monkeypatch, tmp_path)
    monkeypatch.setenv("SGLANG_AGENTIC_KV_HOST_STAGING", enabled)
    with pytest.raises(ValueError, match="requires a source arena"):
        staging.start_registered_host_arena_startup_prewarm(
            role=role, engine_id="engine", tp_rank=0, device="cuda:0",
            local_source_paths=[],
        )


def test_prewarm_captures_rank_device_before_spawning(monkeypatch, tmp_path):
    import threading
    _configure(monkeypatch, tmp_path)
    caller = threading.get_ident()
    devices = []

    def current_device():
        assert threading.get_ident() == caller
        return 7

    def open_mapping(path, device):
        assert threading.get_ident() != caller
        devices.append(str(device))
        return _FakeMapping(path)

    monkeypatch.setattr(staging.torch.cuda, "current_device", current_device)
    monkeypatch.setattr(staging, "_registered_host_arena", open_mapping)
    (tmp_path / "start").touch()
    worker = staging.start_registered_host_arena_startup_prewarm(
        role="decode", engine_id="decode-0", tp_rank=0, device="cuda",
        local_source_paths=["source-local"],
    )
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert devices == ["cuda:7"]


def _configure(monkeypatch, root: Path, *, domain: int = 0):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_REGISTER_STARTUP_BARRIER", "1")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_REGISTER_EAGER_ARENA", "1")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_REGISTER_PREWARM_DIR", str(root))
    monkeypatch.setenv("SGLANG_AGENTIC_KV_REGISTER_PREWARM_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_PREFILL_DOMAIN_COUNT", "2")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_PREFILL_DOMAIN", str(domain))
    monkeypatch.setenv("SGLANG_AGENTIC_KV_TP_SIZE", "1")
    monkeypatch.setattr(staging, "_REGISTERED_HOST_PREWARM_PINS", [])


def _write_manifest(root: Path, domain: int):
    directory = root / "arenas"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"domain-{domain}-rank-0.json").write_text(
        json.dumps(
            {
                "domain": domain,
                "tp_rank": 0,
                "d2p_path": f"d2p-{domain}",
                "p2d_path": f"p2d-{domain}",
                "owner_pid": 1,
            }
        ),
        encoding="utf-8",
    )


def test_prefill_startup_prewarm_registers_all_d2p_and_own_p2d(
    monkeypatch, tmp_path
):
    _configure(monkeypatch, tmp_path, domain=0)
    _write_manifest(tmp_path, 1)
    mappings = {}

    def open_mapping(path, device):
        del device
        mappings[path] = mapping = _FakeMapping(path)
        return mapping

    monkeypatch.setattr(staging, "_registered_host_arena", open_mapping)
    worker = staging.start_registered_host_arena_startup_prewarm(
        role="prefill",
        engine_id="prefill-0",
        tp_rank=0,
        device="cuda:0",
        d2p_arena_path="d2p-0",
        p2d_arena_path="p2d-0",
    )
    (tmp_path / "start").touch()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert set(mappings) == {"d2p-0", "d2p-1", "p2d-0"}
    assert all(mapping.waits for mapping in mappings.values())
    completion = json.loads(
        next((tmp_path / "complete").glob("*.json")).read_text(encoding="utf-8")
    )
    assert completion["arena_count"] == 3
    assert not list((tmp_path / "failed").glob("*.json"))


def test_decode_startup_prewarm_registers_both_directions(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    _write_manifest(tmp_path, 0)
    _write_manifest(tmp_path, 1)
    mappings = {}

    def open_mapping(path, device):
        del device
        mappings[path] = mapping = _FakeMapping(path)
        return mapping

    monkeypatch.setattr(staging, "_registered_host_arena", open_mapping)
    (tmp_path / "start").touch()
    worker = staging.start_registered_host_arena_startup_prewarm(
        role="decode", engine_id="decode-0", tp_rank=0, device="cuda:0"
    )
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert set(mappings) == {"d2p-0", "d2p-1", "p2d-0", "p2d-1"}
    completion = json.loads(
        next((tmp_path / "complete").glob("*.json")).read_text(encoding="utf-8")
    )
    assert completion["registered_bytes"] == 4 * 1024
    assert not list((tmp_path / "failed").glob("*.json"))


def test_direct_only_prefill_prewarm_uses_p2d_arena(monkeypatch, tmp_path):
    """Disabling D->P Host must not disable P->D startup registration."""

    _configure(monkeypatch, tmp_path, domain=0)
    directory = tmp_path / "arenas"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "domain-1-rank-0.json").write_text(
        json.dumps(
            {
                "domain": 1,
                "tp_rank": 0,
                "d2p_path": None,
                "p2d_path": "p2d-1",
                "owner_pid": 1,
            }
        ),
        encoding="utf-8",
    )
    mappings = {}

    def open_mapping(path, device):
        del device
        mappings[path] = mapping = _FakeMapping(path)
        return mapping

    monkeypatch.setattr(staging, "_registered_host_arena", open_mapping)
    worker = staging.start_registered_host_arena_startup_prewarm(
        role="prefill",
        engine_id="prefill-0",
        tp_rank=0,
        device="cuda:0",
        d2p_arena_path=None,
        p2d_arena_path="p2d-0",
    )
    (tmp_path / "start").touch()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert set(mappings) == {"p2d-0"}
    completion = json.loads(
        next((tmp_path / "complete").glob("*.json")).read_text(encoding="utf-8")
    )
    assert completion["arena_count"] == 1
    assert not list((tmp_path / "failed").glob("*.json"))
