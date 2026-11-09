"""Whole-class reconstruction, MRO, aliases and the historical file entry."""
import ast
import importlib.util
import inspect
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import typing

import pytest

from sglang.srt.disaggregation.dualpd.router import (
    late_binding_router as root,
    local_admission as admission,
    local_capacity as capacity,
    local_config as config,
    local_http as http,
    local_types as records,
)

ENGINE = Path(__file__).resolve().parents[7]
ROUTER = ENGINE / "python/sglang/srt/disaggregation/dualpd/router"
spec = importlib.util.spec_from_file_location("router_structure_verifier", ENGINE / "tools/verify_dualpd_router_cleanup.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
RECORD_NAMES = ("DecodeLoad", "DecodeReservation", "_PrefillWorkReservation",
                "_PrefillAdmissionWaiter", "_PReadyAdmission", "_GenerationResponse",
                "_PrefillRedirect")


def test_canonical_records_helpers_and_loggers_have_one_identity():
    for name in (*RECORD_NAMES, "_HOST_STAGE_EVICTING", "_HOST_STAGE_RECOMPUTE_REQUIRED"):
        assert getattr(root, name) is getattr(records, name)
    for name in ("_env_float", "_env_int", "_env_bool"):
        assert getattr(root, name) is getattr(config, name)
    assert root._PrefillAdmissionGate is admission._PrefillAdmissionGate
    assert root._sync_json_get is capacity._sync_json_get
    for module in (records, admission, capacity, http):
        assert module.logger is root.logger
    assert root.logger.name == root.__name__


def test_each_method_has_one_owner_and_concrete_class_keeps_native_constructor():
    concrete = root.LateBindingMiniLoadBalancer
    assert concrete.__bases__ == (capacity.LocalCapacityMixin, admission.LocalAdmissionMixin,
                                  http.LocalHTTPMixin, root.MiniLoadBalancer)
    owners = {}
    for cls, count in [(concrete, 2), (capacity.LocalCapacityMixin, 33),
                       (admission.LocalAdmissionMixin, 38), (http.LocalHTTPMixin, 27)]:
        methods = {name: value for name, value in vars(cls).items()
                   if inspect.isfunction(value) or isinstance(value, (staticmethod, classmethod))}
        assert len(methods) == count
        if cls is not concrete:
            assert "__init__" not in methods
        for name, value in methods.items():
            assert name not in owners
            owners[name] = cls
            assert inspect.getattr_static(concrete, name) is value
    assert len(owners) == 100


@pytest.mark.parametrize("name", RECORD_NAMES)
def test_old_pickle_class_paths_resolve_same_record(name):
    cls = getattr(records, name)
    legacy = pickle.dumps(cls, protocol=0).replace(
        b"sglang.srt.disaggregation.dualpd.router.local_types\n",
        b"sglang.srt.disaggregation.dualpd.router.late_binding_router\n")
    assert pickle.loads(legacy) is cls


def test_old_decode_load_instance_pickle_keeps_all_fields():
    value = root.DecodeLoad("http://d", 1, 2, 3, 4, 5, 6, 7)
    legacy = pickle.dumps(value, protocol=0).replace(
        b"sglang.srt.disaggregation.dualpd.router.local_types\n",
        b"sglang.srt.disaggregation.dualpd.router.late_binding_router\n")
    restored = pickle.loads(legacy)
    assert type(restored) is root.DecodeLoad and restored == value


def test_parsing_still_reads_dynamic_values_and_keeps_empty_defaults(monkeypatch):
    key = "DUALPD_ROUTER_TEST_VALUE"
    monkeypatch.delenv(key, raising=False)
    assert root._env_bool(key, True) is True
    monkeypatch.setenv(key, " YES ")
    assert root._env_bool(key) is True
    monkeypatch.setenv(key, "unknown")
    assert root._env_bool(key, True) is False
    for text, expected in [("2.5", 2.5), ("3.5", 3.5), ("", 9)]:
        monkeypatch.setenv(key, text)
        assert root._env_float(key, 9) == expected
    assert root._env_int(key, 4) == 4


def test_file_package_keeps_native_overlay_private_logger_and_lazy_components(tmp_path):
    code = '''
import asyncio,importlib.abc,importlib.util,json,sys
from pathlib import Path
import aiohttp
import sglang.srt.disaggregation
from sglang.srt.disaggregation import agentic_host_staging,p2d_host_staging
from sglang_router.mini_lb import MiniLoadBalancer
overlay = sglang.srt.disaggregation
overlay_paths = list(overlay.__path__)
paths = list(sys.path)
class OldOverlay(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname == "sglang.srt.disaggregation.dualpd" or fullname.startswith("sglang.srt.disaggregation.dualpd."):
   raise AssertionError("replaced native overlay: " + fullname)
sys.meta_path.insert(0, OldOverlay())
def forbidden(*args, **kwargs):
 raise AssertionError("component import started a task/session")
asyncio.create_task = forbidden
aiohttp.ClientSession = forbidden
path = Path(sys.argv[1])
alias = "dualpd_engine_late_binding_router"
spec = importlib.util.spec_from_file_location(alias,path,submodule_search_locations=[str(path.parent)])
module = importlib.util.module_from_spec(spec)
sys.modules[alias] = module
spec.loader.exec_module(module)
assert module.LateBindingMiniLoadBalancer.__module__ == alias
assert module.LateBindingMiniLoadBalancer.__bases__[-1] is MiniLoadBalancer
for leaf in ("local_types", "local_capacity", "local_admission", "local_http"):
 part = sys.modules[alias + "." + leaf]
 assert part.logger is module.logger
assert module.logger.name == alias
for name in ("DecodeLoad", "DecodeReservation", "_PReadyAdmission"):
 assert getattr(module,name) is getattr(sys.modules[alias + ".local_types"],name)
assert sys.modules["sglang.srt.disaggregation"] is overlay
assert list(overlay.__path__) == overlay_paths
assert [p for p in sys.path if p in paths] == paths
assert all(Path(p).is_relative_to("/tmp") for p in sys.path if p not in paths)
print(json.dumps({"private_package":alias,"native_overlay_preserved":True,"no_import_tasks":True}))
'''
    result = subprocess.run([sys.executable, "-c", code, str(ROUTER / "late_binding_router.py")],
                            cwd=tmp_path, env=dict(os.environ), text=True,
                            capture_output=True, check=True, timeout=90)
    assert json.loads(result.stdout.splitlines()[-1])["native_overlay_preserved"]


def test_complete_original_module_reconstructs_all_methods_and_data():
    result = verifier.verify()
    assert result["local_methods_ast_identical"] == 100
    assert result["moved_top_level_nodes_ast_identical"] == 14
    assert result["complete_original_module_reconstructed"]
    assert result["runtime_global_bindings_verified"]


def test_relocated_annotations_resolve_in_their_defining_modules():
    for cls in (root.LateBindingMiniLoadBalancer, capacity.LocalCapacityMixin,
                admission.LocalAdmissionMixin, http.LocalHTTPMixin,
                admission._PrefillAdmissionGate):
        for value in vars(cls).values():
            if isinstance(value, (staticmethod, classmethod)):
                value = value.__func__
            if inspect.isfunction(value):
                typing.get_type_hints(value)
    for name in RECORD_NAMES:
        typing.get_type_hints(getattr(records, name))


@pytest.mark.parametrize("mutation", ["body", "missing", "extra", "parser", "mro", "loader", "assertion", "logger", "import_task", "missing_import", "wrong_import", "missing_record_import"])
def test_verifier_rejects_behavior_loss_or_weakened_gates(mutation):
    filename = {"body":"local_capacity.py", "missing":"local_capacity.py", "extra":"local_http.py",
                "parser":"local_config.py", "mro":"late_binding_router.py", "loader":"launch_late_binding_router.py",
                "assertion":"tests/test_late_binding_router.py", "logger":"local_types.py",
                "import_task":"local_capacity.py", "missing_import":"local_http.py",
                "wrong_import":"local_http.py", "missing_record_import":"local_capacity.py"}[mutation]
    source = (ROUTER / filename).read_text()
    if mutation == "body":
        source = source.replace("if drain:", "if not drain:", 1)
    elif mutation == "missing":
        tree = ast.parse(source)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LocalCapacityMixin")
        cls.body.remove(next(n for n in cls.body if isinstance(n, ast.FunctionDef)))
        source = ast.unparse(tree)
    elif mutation == "extra":
        source += "\n    def unexpected(self):\n        return None\n"
    elif mutation == "parser":
        source = source.replace('"true"', '"truthy"')
    elif mutation == "mro":
        source = source.replace("LocalCapacityMixin, LocalAdmissionMixin, LocalHTTPMixin, MiniLoadBalancer",
                                "LocalAdmissionMixin, LocalCapacityMixin, LocalHTTPMixin, MiniLoadBalancer")
    elif mutation == "loader":
        source = source.replace("dualpd_engine_late_binding_router", "different_overlay")
    elif mutation == "assertion":
        source = source.replace("assertNotEqual(worker_threads[0], loop_thread)", "assertEqual(worker_threads[0], loop_thread)")
    elif mutation == "logger":
        source = source.replace("dualpd_engine_late_binding_router", "different_private_logger")
    elif mutation == "missing_import":
        source = source.replace("from .local_config import _env_float, _env_int\n", "")
    elif mutation == "wrong_import":
        source = source.replace("from .local_config import _env_float, _env_int", "from .local_types import _env_float, _env_int")
    elif mutation == "missing_record_import":
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "local_types":
                node.names = [alias for alias in node.names if alias.name != "DecodeReservation"]
        source = ast.unparse(tree)
    else:
        source += "\nasyncio.create_task(unexpected())\n"
    assert source != (ROUTER / filename).read_text(), "Mutation fixture did not change anything"
    with pytest.raises(AssertionError):
        verifier.verify({filename: source})
