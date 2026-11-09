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
DMA_NAMES = {
    "_CUDAMemLocation", "_CUDAMemcpyAttributes", "_cuda_runtime",
    "_cuda_memcpy_async", "_cuda_driver_batch_memcpy", "_cuda_batch_memcpy_async",
    "_CUDA_RUNTIME", "_CUDA_RUNTIME_LOCK", "_CUDA_DRIVER_BATCH", "_CUDA_DRIVER_BATCH_LOCK",
}
REGISTRATION_NAMES = {
    "_is_memfd_backing_path", "_validate_shared_host_backing_path",
    "_open_shared_host_backing", "_RegisteredHostArenaMapping",
    "_REGISTERED_HOST_ARENAS", "_REGISTERED_HOST_ARENAS_LOCK",
    "_REGISTERED_HOST_PREWARM_PINS", "_registered_host_arena",
    "_write_startup_prewarm_record", "start_registered_host_arena_startup_prewarm",
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
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
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
    dma = ast.parse(current("agentic_host_dma.py"))
    expected_dma = [n for n in old.body if label(n) in DMA_NAMES]
    actual_dma = [n for n in dma.body if label(n) in DMA_NAMES]
    assert len(expected_dma) == len(actual_dma) == len(DMA_NAMES)
    assert dump(expected_dma) == dump(actual_dma), "CUDA ABI, cache or submission branches changed"
    extra_dma = [n for n in dma.body if label(n) not in DMA_NAMES
                 and not isinstance(n, (ast.Import, ast.ImportFrom))
                 and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                          and isinstance(n.value.value, str))]
    assert not extra_dma, "Unexpected executable code in extracted DMA module"
    registration = ast.parse(current("agentic_host_registration.py"))
    expected_registration = [n for n in old.body if label(n) in REGISTRATION_NAMES]
    actual_registration = [n for n in registration.body if label(n) in REGISTRATION_NAMES]
    assert len(expected_registration) == len(actual_registration) == len(REGISTRATION_NAMES)
    assert dump(expected_registration) == dump(actual_registration), "Host registration/cache/prewarm branches changed"
    registration_logger = ast.parse(
        'logger = logging.getLogger("sglang.srt.disaggregation.agentic_host_staging")'
    ).body[0]
    extra_registration = [n for n in registration.body
                          if label(n) not in REGISTRATION_NAMES
                          and not isinstance(n, (ast.Import, ast.ImportFrom))
                          and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                                   and isinstance(n.value.value, str))]
    assert len(extra_registration) == 1 and dump(extra_registration[0]) == dump(registration_logger)
    assert dump(module_body(original("agentic_host_staging.py"), exclude=LEDGER_NAMES | DMA_NAMES | REGISTRATION_NAMES, imports=False)) == dump(
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
    expected_prewarm_test = original("test_agentic_startup_prewarm.py").replace(
        "import agentic_host_staging as staging", "import agentic_host_registration as staging"
    )
    assert dump(ast.parse(expected_prewarm_test)) == dump(ast.parse(current("tests/test_agentic_startup_prewarm.py")))
    # All other relocated source/test files remain byte-identical. Current
    # hashes are explicit for the reviewed transformations above.
    source_map = json.loads((ROOT / "dualpd-source-map.json").read_text())
    changed = {item["original"].rsplit("/", 1)[-1] for item in source_map["files"]
               if "current_sha256" in item}
    assert changed == {"agentic_host_staging.py", "agentic_memory_scheduler_bridge.py",
                       "agentic_multinode.py", "test_agentic_memory_scheduler_bridge.py",
                       "test_agentic_multinode.py", "test_agentic_startup_prewarm.py"} | cross_import_tests, changed
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
        "dma_nodes_ast_identical": len(DMA_NAMES),
        "registration_nodes_ast_identical": len(REGISTRATION_NAMES),
        "host_dma_and_worker_ast_identical": True,
        "preserved_environment_identifiers": len(after_keys),
        "runtime_changes": "ledger, CUDA ABI and registration relocation; only two uncalled helpers removed",
        "test_changes": "directory relocation, source lookup, prewarm private-hook import and obsolete helper assertion",
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
