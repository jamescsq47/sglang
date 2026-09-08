import json
from pathlib import Path

from sglang.srt.disaggregation import agentic_host_staging as staging


class _FakeMapping:
    def __init__(self, path: str):
        self.path = path
        self.byte_size = 1024
        self.waits = []

    def wait_prewarm(self, timeout):
        self.waits.append(timeout)


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
