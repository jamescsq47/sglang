"""Reconstruct the complete pre-refactor local Router; never waive its body checks.

Only the concrete class's method-only MRO, component import locations and the
standalone file-as-package loader change. TCP/pinned policies remain separate.
"""
import ast
import builtins
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import symtable

ROOT = Path(__file__).resolve().parents[1]
BASE = "09911f4ed3f58e85a71cc7016ce5f999b9880911"
REL = "python/sglang/srt/disaggregation/dualpd/router/"
PARTS = {
    "local_capacity.py": ("LocalCapacityMixin", 33),
    "local_admission.py": ("LocalAdmissionMixin", 38),
    "local_http.py": ("LocalHTTPMixin", 27),
}
TYPES = {"DecodeLoad", "DecodeReservation", "_PrefillWorkReservation",
         "_PrefillAdmissionWaiter", "_PReadyAdmission", "_GenerationResponse",
         "_PrefillRedirect", "_HOST_STAGE_EVICTING", "_HOST_STAGE_RECOMPUTE_REQUIRED"}
CONFIG = {"_env_float", "_env_int", "_env_bool"}


def original(name):
    return subprocess.check_output(["git", "-C", str(ROOT), "show",
                                    f"{BASE}:{REL}{name}"], text=True)


def dump(node):
    return ast.dump(node, include_attributes=False)


def label(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


def is_doc(node):
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def definitions(tree):
    result = {}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) or is_doc(node):
            continue
        name = label(node)
        assert name is not None and name not in result, "Extra/duplicate executable node"
        result[name] = node
    return result


def methods(cls):
    result = {}
    for node in cls.body:
        if is_doc(node):
            continue
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)), "Class executable/state added"
        assert node.name not in result, "Duplicate method"
        result[node.name] = node
    return result


def import_specs(tree):
    result = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            result.update((None, a.name, a.asname, 0) for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.update((node.module, a.name, a.asname, node.level) for a in node.names)
    return result


def verify_global_bindings(source, tree, filename, *, definition_names=None):
    """AST reconstruction must not hide missing imports in relocated bodies."""
    bound = set(definitions(tree) if definition_names is None else definition_names) | {
        "__name__", "__package__", "__file__", "__doc__", "__builtins__"
    }
    for node in tree.body:
        if isinstance(node, ast.Import):
            bound.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert all(alias.name != "*" for alias in node.names)
            bound.update(alias.asname or alias.name for alias in node.names)
    referenced = set()
    def visit(table):
        referenced.update(symbol.get_name() for symbol in table.get_symbols()
                          if symbol.is_global() and symbol.is_referenced())
        for child in table.get_children():
            visit(child)
    visit(symtable.symtable(source, filename, "exec"))
    missing = referenced - bound - set(vars(builtins))
    assert not missing, (filename, "Unbound runtime globals", sorted(missing))


def verify_tcp(overrides=None):
    """Reconstruct TCP separately; retain every root callback and service guard."""
    overrides = overrides or {}
    def read(name):
        return overrides[name] if name in overrides else (ROOT / REL / name).read_text()
    old_source = original("global_pd_router.py")
    old = ast.parse(old_source)
    names = ("global_pd_router.py", "tcp_capacity.py", "tcp_http.py")
    trees = {name: ast.parse(read(name)) for name in names}

    def top(tree):
        result = {}
        for index, node in enumerate(tree.body):
            if isinstance(node, (ast.Import, ast.ImportFrom)) or (index == 0 and is_doc(node)):
                continue
            name = label(node)
            if name is None and isinstance(node, ast.If):
                name = "service_guard"
            assert name is not None and name not in result, "Extra/duplicate TCP executable node"
            result[name] = node
        return result

    expected = top(old)
    defs = {name: top(tree) for name, tree in trees.items()}
    main = defs["global_pd_router.py"]
    assert set(main) == set(expected) - {"_DecodeReservation"}
    assert dump(trees["global_pd_router.py"].body[0]) == dump(old.body[0])
    assert set(defs["tcp_capacity.py"]) == {"TCPCapacityMixin", "_DecodeReservation"}
    assert set(defs["tcp_http.py"]) == {"TCPHTTPMixin"}
    expected_imports = {
        "global_pd_router.py": import_specs(old) - import_specs(ast.parse(
            "import time, copy, uuid\nfrom dataclasses import dataclass\n"
            "from fastapi.responses import ORJSONResponse, StreamingResponse")) | import_specs(ast.parse(
            "from .tcp_capacity import TCPCapacityMixin, _DecodeReservation\n"
            "from .tcp_http import TCPHTTPMixin")),
        "tcp_capacity.py": import_specs(ast.parse(
            "from __future__ import annotations\nimport asyncio, time, uuid\n"
            "from dataclasses import dataclass\nimport aiohttp")),
        "tcp_http.py": import_specs(ast.parse(
            "from __future__ import annotations\nimport asyncio, copy, json, logging, uuid\n"
            "import aiohttp\nfrom fastapi.responses import ORJSONResponse, StreamingResponse")),
    }
    for name, tree in trees.items():
        assert import_specs(tree) == expected_imports[name], (name, "TCP import owners changed")
        verify_global_bindings(read(name), tree, name, definition_names=set(defs[name]) - {"service_guard"})
    record = defs["tcp_capacity.py"]["_DecodeReservation"]
    assert dump(record) == dump(expected["_DecodeReservation"]), "TCP record changed"
    concrete = main["GlobalEndpointMiniLoadBalancer"]
    assert [ast.unparse(node) for node in concrete.bases] == [
        "TCPCapacityMixin", "TCPHTTPMixin", "mini_lb_module.MiniLoadBalancer"]
    moved = methods(concrete)
    assert set(moved) == {"__init__", "close", "select_pair", "_annotate_prefill",
                          "_annotate_decode", "_room", "publish_prefill_ready", "_wait_prefill_ready"}
    counts = {}
    for filename, name, count in [("tcp_capacity.py", "TCPCapacityMixin", 12),
                                  ("tcp_http.py", "TCPHTTPMixin", 9)]:
        cls = defs[filename][name]
        assert not cls.bases and not cls.keywords and not cls.decorator_list
        component = methods(cls)
        assert len(component) == count and "__init__" not in component
        assert not set(component) & set(moved)
        assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                       and node.func.id == "super" for node in ast.walk(cls))
        moved.update(component)
        counts[name] = len(component)
    original_cls = expected["GlobalEndpointMiniLoadBalancer"]
    old_methods = methods(original_cls)
    assert len(moved) == len(old_methods) == 29 and moved.keys() == old_methods.keys()
    for name, method in moved.items():
        assert dump(method) == dump(old_methods[name]), (name, "TCP policy/lifecycle body changed")
    rebuilt = copy.deepcopy(concrete)
    rebuilt.bases = copy.deepcopy(original_cls.bases)
    rebuilt.body = [moved[node.name] for node in original_cls.body]
    assert dump(rebuilt) == dump(original_cls), "Complete TCP class does not reconstruct"
    reconstructed = dict(main, _DecodeReservation=record, GlobalEndpointMiniLoadBalancer=rebuilt)
    for name in expected:
        assert dump(reconstructed[name]) == dump(expected[name]), (name, "TCP callback/main/guard changed")
    tree = ast.Module(body=[reconstructed[label(node) or "service_guard"]
                           if not isinstance(node, (ast.Import, ast.ImportFrom)) and not is_doc(node)
                           else node for node in old.body], type_ignores=[])
    assert dump(tree) == dump(old), "Complete TCP module does not reconstruct"
    source_map = json.loads((ROOT / "router-source-map.json").read_text())
    entry = next(item for item in source_map["files"] if item["path"].endswith("/global_pd_router.py"))
    assert hashlib.sha256(old_source.encode()).hexdigest() == entry["sha256"]
    assert original("tests/test_global_pd_router.py") == read("tests/test_global_pd_router.py"), "Existing TCP test source changed"
    return {"methods_ast_identical": 29, "component_methods": counts,
            "record_ast_identical": True, "complete_original_module_reconstructed": True,
            "callbacks_main_and_service_guard_ast_identical": True,
            "runtime_global_bindings_verified": True, "existing_tests_unchanged": True}


def verify(overrides=None, *, return_tree=False):
    overrides = overrides or {}
    spec = importlib.util.spec_from_file_location(
        "router_control_dependency_proof", ROOT / "tools/verify_dualpd_control_dependencies.py")
    control_proof = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control_proof)
    control_result = control_proof.verify(router_overrides=overrides)
    def read(name):
        return overrides[name] if name in overrides else (ROOT / REL / name).read_text()
    old_source = original("late_binding_router.py")
    old = ast.parse(old_source)
    expected = definitions(old)
    trees = {name: ast.parse(read(name)) for name in
             ["late_binding_router.py", "local_types.py", "local_config.py", *PARTS]}
    # The separate whole-module proof above validates all six exact relocations.
    # Invert only their import owners before the original complete reconstruction.
    for tree in trees.values():
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "local_dependencies":
                names = tuple(a.name for a in node.names)
                node.module = ("sglang.srt.disaggregation.p2d_host_staging"
                               if names == control_proof.PROTOCOL else
                               "sglang.srt.disaggregation.agentic_host_staging")
                node.level = 0
    defs = {name: definitions(tree) for name, tree in trees.items()}
    main = defs["late_binding_router.py"]
    assert set(main) == {"logger", "LateBindingMiniLoadBalancer"}
    assert dump(main["logger"]) == dump(expected["logger"])
    assert dump(trees["late_binding_router.py"].body[0]) == dump(old.body[0])
    assert set(defs["local_types.py"]) == TYPES | {"logger"}
    assert set(defs["local_config.py"]) == CONFIG
    logger = ast.parse('logger = logging.getLogger(__package__ if __package__ == '
                       '"dualpd_engine_late_binding_router" else f"{__package__}.late_binding_router")').body[0]
    assert dump(defs["local_types.py"]["logger"]) == dump(logger), "Historical logger context changed"

    owners = {name: "local_types" for name in TYPES}
    owners.update({name: "local_config" for name in CONFIG})
    owners.update({"_sync_json_get": "local_capacity", "_PrefillAdmissionGate": "local_admission"})
    for filename, (name, _) in PARTS.items():
        owners[name] = filename.removesuffix(".py")
    allowed_imports = import_specs(old) | {
        (module, name, None, 1) for name, module in owners.items()
    } | {("local_types", "logger", None, 1)}
    for name, tree in trees.items():
        assert import_specs(tree) <= allowed_imports, (name, "Unexpected import binding")
        verify_global_bindings(read(name), tree, name)

    actual_globals = {"logger": main["logger"]}
    for name in TYPES:
        actual_globals[name] = defs["local_types.py"][name]
    for name in CONFIG:
        actual_globals[name] = defs["local_config.py"][name]
    for name, filename in [("_sync_json_get", "local_capacity.py"),
                           ("_PrefillAdmissionGate", "local_admission.py")]:
        actual_globals[name] = defs[filename][name]
    for name, node in actual_globals.items():
        assert dump(node) == dump(expected[name]), (name, "Top-level policy/data changed")

    concrete = main["LateBindingMiniLoadBalancer"]
    assert [dump(n) for n in concrete.bases] == [dump(ast.Name(id=name, ctx=ast.Load())) for name in
        ["LocalCapacityMixin", "LocalAdmissionMixin", "LocalHTTPMixin", "MiniLoadBalancer"]]
    assert not concrete.keywords and not concrete.decorator_list
    actual_methods = methods(concrete)
    assert set(actual_methods) == {"__init__", "select_pair"}
    counts = {}
    for filename, (name, count) in PARTS.items():
        extras = {"_sync_json_get"} if filename == "local_capacity.py" else (
            {"_PrefillAdmissionGate"} if filename == "local_admission.py" else set())
        assert set(defs[filename]) == {name} | extras
        cls = defs[filename][name]
        assert not cls.bases and not cls.keywords and not cls.decorator_list
        moved = methods(cls)
        assert len(moved) == count and "__init__" not in moved
        assert not (set(moved) & set(actual_methods)), "Method assigned more than once"
        assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "super"
                       for n in ast.walk(cls)), "Mixin added a super/MRO dependency"
        actual_methods.update(moved)
        counts[name] = len(moved)
    old_cls = expected["LateBindingMiniLoadBalancer"]
    old_methods = methods(old_cls)
    assert len(actual_methods) == len(old_methods) == 100
    assert actual_methods.keys() == old_methods.keys(), "Missing or additional method"
    for name, node in actual_methods.items():
        assert dump(node) == dump(old_methods[name]), (name, "Routing/ownership/HTTP body changed")
    assert dump(concrete.body[0]) == dump(old_cls.body[0])
    rebuilt = copy.deepcopy(concrete)
    rebuilt.bases = copy.deepcopy(old_cls.bases)
    rebuilt.body = [copy.deepcopy(concrete.body[0])] + [actual_methods[n.name] for n in old_cls.body[1:]]
    assert dump(rebuilt) == dump(old_cls), "Whole original class does not reconstruct"
    actual_globals[rebuilt.name] = rebuilt
    assert set(actual_globals) == set(expected), "Top-level definition inventory changed"
    reconstructed = ast.Module(body=[actual_globals[label(n)] if label(n) else n
                                    for n in old.body], type_ignores=[])
    assert dump(reconstructed) == dump(old), "Complete original module does not reconstruct"

    launch = ast.parse(original("launch_late_binding_router.py"))
    specs = [n for n in ast.walk(launch) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "spec_from_file_location"]
    assert len(specs) == 1
    specs[0].keywords.append(ast.keyword(arg="submodule_search_locations",
        value=ast.parse("[str(Path(__file__).parent)]", mode="eval").body))
    assert dump(launch) == dump(ast.parse(read("launch_late_binding_router.py"))), "Overlay loader changed beyond package search path"
    test = original("tests/test_late_binding_router.py")
    hook = "late_binding_router_module._sync_json_get"
    assert test.count(hook) == 3
    test = test.replace("import sglang.srt.disaggregation.dualpd.router.late_binding_router as late_binding_router_module",
        "import sglang.srt.disaggregation.dualpd.router.late_binding_router as late_binding_router_module\n"
        "import sglang.srt.disaggregation.dualpd.router.local_capacity as local_capacity_module")
    test = test.replace(hook, "local_capacity_module._sync_json_get")
    assert dump(ast.parse(test)) == dump(ast.parse(read("tests/test_late_binding_router.py"))), "Existing test assertions changed"
    source_map = json.loads((ROOT / "router-source-map.json").read_text())
    local = next(item for item in source_map["files"] if item["path"].endswith("/late_binding_router.py"))
    assert hashlib.sha256(old_source.encode()).hexdigest() == local["sha256"], "Reference provenance changed"
    result = {"reference_commit": BASE, "local_methods_ast_identical": 100,
              "control_dependency_reconstruction": control_result,
              "component_methods": counts, "moved_top_level_nodes_ast_identical": 14,
              "complete_original_module_reconstructed": True,
              "runtime_global_bindings_verified": True,
              "standalone_loader_change": "same alias/native overlay; file-as-package search path only",
              "existing_test_change": "one defining-module import and three private hook references; assertions identical"}
    if any(item["path"].endswith("/global_pd_router.py") and "current_sha256" in item
           for item in source_map["files"]):
        result["tcp_reconstruction"] = verify_tcp(overrides)
    return (result, reconstructed) if return_tree else result


if __name__ == "__main__":
    result, tree = verify(return_tree=True)
    if "--normalized-local-ast" in sys.argv:
        class RemoveImports(ast.NodeTransformer):
            def visit_Import(self, node):
                return None
            def visit_ImportFrom(self, node):
                return None
        result["normalized_local_ast"] = dump(RemoveImports().visit(copy.deepcopy(tree)))
    print(json.dumps(result, indent=2))
