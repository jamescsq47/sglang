"""Read-only AST equivalence gate for the first engine cleanup.

The reference is the pre-cleanup engine commit, not a mutable sibling checkout.
Moving ownership code must not silently change a single state/CAS/fence branch.
"""
import ast
import json
import re
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = "bc67aa0814364f2c7ce3df3f83f9d5ff013c5b5f"
EXT = "python/sglang/srt/disaggregation/dualpd/"
LEDGER_NAMES = {
    "HostStageState", "PermanentHostStageError", "SharedHostStagingLedger",
    "_TERMINAL_STATES", "_ALLOWED_STAGE_TRANSITIONS", "_LEDGER_OWNERSHIP_CHANGED",
    "P2D_RELEASE_NATIVE_WON", "P2D_RELEASE_HOST_OWNED", "P2D_RELEASE_HOST_TERMINAL",
}


def original(name):
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f"{BASE}:{EXT}{name}"], text=True)


def current(name):
    return (ROOT / EXT / name).read_text()


def label(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


def dump(value):
    if isinstance(value, list):
        value = ast.Module(body=value, type_ignores=[])
    return ast.dump(value, include_attributes=False)


def module_body(source, *, exclude=(), imports=True):
    return [n for n in ast.parse(source).body
            if label(n) not in exclude
            and (imports or not isinstance(n, (ast.Import, ast.ImportFrom)))]


def verify():
    old = ast.parse(original("agentic_host_staging.py"))
    ledger = ast.parse(current("agentic_host_ledger.py"))
    expected = [n for n in old.body if label(n) in LEDGER_NAMES]
    actual = [n for n in ledger.body if label(n) in LEDGER_NAMES]
    assert len(expected) == len(actual) == len(LEDGER_NAMES)
    assert dump(expected) == dump(actual), "Host ledger implementation/state table changed"
    extra = [n for n in ledger.body if label(n) not in LEDGER_NAMES
             and not isinstance(n, (ast.Import, ast.ImportFrom))
             and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                      and isinstance(n.value.value, str))]
    assert not extra, "Unexpected executable code in the extracted ledger"
    assert dump(module_body(original("agentic_host_staging.py"), exclude=LEDGER_NAMES, imports=False)) == dump(
        module_body(current("agentic_host_staging.py"), imports=False)), "Host DMA/worker logic changed"

    old_bridge = ast.parse(original("agentic_memory_scheduler_bridge.py"))
    old_bridge.body = [n for n in old_bridge.body
                       if label(n) != "memory_authority_v2_enabled"
                       and not (isinstance(n, ast.Import) and [x.name for x in n.names] == ["os"])]
    assert dump(old_bridge) == dump(ast.parse(current("agentic_memory_scheduler_bridge.py")))
    assert dump(module_body(original("agentic_multinode.py"), exclude={"control_poll_interval"})) == dump(
        module_body(current("agentic_multinode.py"))), "Multi-node config/runtime changed"

    test_bridge = ast.parse(original("test_agentic_memory_scheduler_bridge.py"))
    test_bridge.body = [n for n in test_bridge.body if label(n) != "test_v2_flag_is_default_off"]
    for n in test_bridge.body:
        if isinstance(n, ast.ImportFrom):
            n.names = [a for a in n.names if a.name != "memory_authority_v2_enabled"]
    assert dump(test_bridge) == dump(ast.parse(current("tests/test_agentic_memory_scheduler_bridge.py")))
    test_multinode = original("test_agentic_multinode.py").replace(
        'pathlib.Path(__file__).with_name("agentic_multinode.py")',
        'pathlib.Path(__file__).resolve().parents[1] / "agentic_multinode.py"')
    assert dump(ast.parse(test_multinode)) == dump(ast.parse(current("tests/test_agentic_multinode.py")))

    cross_import_tests = {
        "test_agentic_token_content_hash.py", "test_agentic_host_event_progress.py",
        "test_agentic_host_async_prepare.py", "test_agentic_handoff_payload_refs.py",
    }
    for name in cross_import_tests:
        expected_test = original(name).replace(
            "sglang.srt.disaggregation.test_agentic_",
            "sglang.srt.disaggregation.dualpd.tests.test_agentic_")
        assert dump(ast.parse(expected_test)) == dump(ast.parse(current("tests/" + name)))
    # All other relocated source/test files remain byte-identical. Current
    # hashes are explicit for the reviewed transformations above.
    source_map = json.loads((ROOT / "dualpd-source-map.json").read_text())
    changed = {item["original"].rsplit("/", 1)[-1] for item in source_map["files"]
               if "current_sha256" in item}
    assert changed == {"agentic_host_staging.py", "agentic_memory_scheduler_bridge.py",
                       "agentic_multinode.py", "test_agentic_memory_scheduler_bridge.py",
                       "test_agentic_multinode.py"} | cross_import_tests, changed
    # Inventory literal environment keys across every extension implementation,
    # including both Router variants. Only the proven no-op selector disappears.
    before = subprocess.check_output([
        "git", "-C", str(ROOT), "grep", "-h", "-o", "-E",
        "SGLANG_[A-Z0-9_]+", BASE, "--", EXT,
        ":(exclude)**/test*.py", ":(exclude)**/*.md",
    ], text=True)
    before_keys = set(before.splitlines())
    after_keys = set()
    for path in (ROOT / EXT).rglob("*.py"):
        if not path.name.startswith("test"):
            after_keys.update(re.findall(r"\bSGLANG_[A-Z0-9_]+\b", path.read_text()))
    assert before_keys - after_keys == {"SGLANG_AGENTIC_MEMORY_AUTHORITY_V2"}
    assert not after_keys - before_keys, sorted(after_keys - before_keys)
    return {
        "reference_commit": BASE,
        "ledger_nodes_ast_identical": len(LEDGER_NAMES),
        "host_dma_and_worker_ast_identical": True,
        "preserved_environment_identifiers": len(after_keys),
        "runtime_changes": "only two uncalled helpers removed",
        "test_changes": "directory relocation, source lookup and obsolete helper assertion",
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
