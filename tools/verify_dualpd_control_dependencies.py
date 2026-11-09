"""Whole-module proof for control-plane dependency extraction (source only)."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

BASE = "a550ccaecad5d2b27ebb15a2788c1e1dc0daff8f"
ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT
EXT = "python/sglang/srt/disaggregation/dualpd/"
ROUTER = EXT + "router/"
NAMES = {"P2D_CUSTOM_SNAPSHOT_ID", "P2D_CUSTOM_PREFILL_DOMAIN", "p2d_snapshot_id"}
LEDGER = ("HostStageState", "SharedHostStagingLedger")
PROTOCOL = ("P2D_CUSTOM_PREFILL_DOMAIN", "P2D_CUSTOM_SNAPSHOT_ID", "p2d_snapshot_id")


def dump(body):
    return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)


def original(relative):
    return subprocess.check_output(
        ["git", "-C", str(REFERENCE), "show", f"{BASE}:{relative}"], text=True)


def label(node):
    if isinstance(node, ast.FunctionDef):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        return getattr(node.targets[0], "id", None)


def verify(root=ROOT, *, router_overrides=None):
    router_overrides = router_overrides or {}
    counts = {"late_binding_router.py": 2, "local_admission.py": 2,
              "local_capacity.py": 1, "local_types.py": 1}
    for name, expected_count in counts.items():
        tree = ast.parse(router_overrides[name] if name in router_overrides
                         else (root / ROUTER / name).read_text())
        changed = 0
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "local_dependencies":
                assert node.level == 1 and all(a.asname is None for a in node.names)
                names = tuple(a.name for a in node.names)
                if names == PROTOCOL:
                    assert name in {"late_binding_router.py", "local_admission.py"}
                    node.module = "sglang.srt.disaggregation.p2d_host_staging"
                else:
                    assert names == (LEDGER if name == "late_binding_router.py" else ("HostStageState",))
                    node.module = "sglang.srt.disaggregation.agentic_host_staging"
                node.level = 0
                changed += 1
        assert changed == expected_count, (name, "import count")
        assert dump(tree.body) == dump(ast.parse(original(ROUTER + name)).body), (name, "entire module changed")

    old_worker = ast.parse(original(EXT + "p2d_host_staging.py"))
    moved = [n for n in old_worker.body if label(n) in NAMES]
    assert len(moved) == 3
    protocol = ast.parse((root / EXT / "p2d_host_protocol.py").read_text())
    header = ast.parse(
        '\"\"\"Pure P-to-D wire declarations; no Host state, worker or device imports.\"\"\"\n'
        'from __future__ import annotations\n'
    ).body
    assert dump(protocol.body) == dump(header + moved), "entire protocol module changed"
    worker = ast.parse((root / EXT / "p2d_host_staging.py").read_text())
    imports = [n for n in worker.body if isinstance(n, ast.ImportFrom)
               and n.module == "sglang.srt.disaggregation.p2d_host_protocol"]
    assert len(imports) == 1 and imports[0].level == 0
    assert [(a.name, a.asname) for a in imports[0].names] == [
        ("P2D_CUSTOM_SNAPSHOT_ID", None), ("P2D_CUSTOM_PREFILL_DOMAIN", None),
        ("p2d_snapshot_id", None)]
    assert dump([n for n in worker.body if n not in imports]) == dump(
        [n for n in old_worker.body if label(n) not in NAMES]), "remaining complete worker changed"
    expected_worker = [n for n in old_worker.body if label(n) not in NAMES]
    torch_indices = [i for i, n in enumerate(expected_worker)
                     if isinstance(n, ast.Import)
                     and [(a.name, a.asname) for a in n.names] == [("torch", None)]]
    assert len(torch_indices) == 1
    protocol_import = ast.parse(
        "from sglang.srt.disaggregation.p2d_host_protocol import ("
        "P2D_CUSTOM_SNAPSHOT_ID, P2D_CUSTOM_PREFILL_DOMAIN, p2d_snapshot_id)"
    ).body[0]
    expected_worker.insert(torch_indices[0] + 1, protocol_import)
    assert dump(worker.body) == dump(expected_worker), "worker protocol import timing changed"
    bridge = (root / ROUTER / "local_dependencies.py").read_bytes()
    assert hashlib.sha256(bridge).hexdigest() == "a5860d6237d2a19962526186f413b8a79142e0a3997ad645e378793b2d9fcb7b", "reviewed complete bridge changed"
    return {"reference_commit": BASE, "entire_router_modules_reconstructed": 4,
            "exact_import_relocations": 6, "protocol_original_nodes": 3,
            "whole_protocol_and_remaining_worker_verified": True,
            "worker_protocol_import_exact_position_verified": True,
            "reviewed_complete_bridge_verified": True,
            "scope": "Source equivalence only; native, overlay and GPU gates are separate."}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
