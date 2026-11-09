"""Portable Scheduler source/dispatch gates; real native imports tested separately."""
import importlib.abc
import ast
import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "tools/verify_dualpd_scheduler_components.py").is_file())
spec = importlib.util.spec_from_file_location(
    "scheduler_component_proof", ROOT / "tools/verify_dualpd_scheduler_components.py")
proof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proof)
D = proof.EXT + "agentic_scheduler_direct.py"
T = proof.EXT + "agentic_scheduler_tp.py"
R = proof.EXT + "agentic_scheduler_ready.py"
C = proof.EXT + "agentic_scheduler_compat.py"
O = proof.EXT + "agentic_scheduler_ownership.py"
CANONICAL = "sglang.srt.managers.scheduler"


def resolver():
    spec = importlib.util.spec_from_file_location(
        "isolated_scheduler_candidate_compat", proof.ROOT / C)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._native_scheduler_class


def test_complete_candidate_proof():
    result = proof.verify()
    assert result["whole_native_module_reconstructed"]
    assert result["complete_methods_checked"] == 54
    assert result["native_explicit_wrappers_verified"] == 4
    assert result["exact_concrete_lookups"] == 7
    assert result["imports_and_global_bindings_verified"]
    assert result["compat_complete_ast_verified"]
    assert result["existing_rankset_assertions_unchanged"]


@pytest.mark.parametrize("path,before,after", [
    (D, "import torch", ""),
    (D, "from typing import Optional", "from typing import Any as Optional"),
    (D, "from sglang.srt.managers.schedule_batch import Req",
     "from sglang.srt.managers.schedule_batch import ScheduleBatch as Req"),
    (D, "import logging", "from __future__ import annotations\nimport logging"),
    (D, "import logging", "import logging\nimport sglang.srt.managers.scheduler"),
    (D, "class AgenticDirectMixin:", "class AgenticDirectMixin:\n    unexpected = object()"),
    (D, "class AgenticDirectMixin:", "class Hidden:\n    x = object()\n\nclass AgenticDirectMixin:"),
    (D, 'AgenticEarlyDirectReceive.__module__ = "sglang.srt.managers.scheduler"',
     "AgenticEarlyDirectReceive.__module__ = __name__"),
    (D, "    def _agentic_poll_early_direct_receives(",
     "    def _agentic_changed_poll_early_direct_receives("),
    (D, "_native_scheduler_class()._agentic_early_direct_slots_used(self)",
     "type(self)._agentic_early_direct_slots_used(self)"),
    (D, "_native_scheduler_class()._agentic_early_direct_slots_used(self)",
     "self._agentic_early_direct_slots_used()"),
    (T, "_native_scheduler_class()._AGENTIC_TP_CONTROL_KEY",
     "_native_scheduler_class()._AGENTIC_V2_ACTIVATION_KEY"),
    (T, "from sglang.srt.disaggregation.agentic_scheduler_compat import _native_scheduler_class",
     ""),
    (R, "import logging", "import logging\nclass Hidden:\n    side_effect = object()"),
    (C, 'sys.modules.get("sglang.srt.managers.scheduler")',
     'sys.modules.get("other_fork.scheduler")'),
    (C, "    return module.Scheduler", "    return object"),
    (C, "import sys", "import sys\nimport sglang.srt.managers.scheduler"),
    (C, "    return module.Scheduler", "    return _cached_scheduler"),
    (C, "import sys", "import sys\n_cached_scheduler = object"),
    (proof.NATIVE, "    AgenticDirectMixin,\n    AgenticTPMixin,",
     "    AgenticTPMixin,\n    AgenticDirectMixin,"),
    (proof.NATIVE, "logger = logging.getLogger(__name__)",
     'logger = logging.getLogger("changed-native")'),
    (proof.NATIVE, "return AgenticOwnershipHooks._agentic_snapshot_store(self)",
     "AgenticOwnershipHooks._agentic_snapshot_store(self)"),
    (proof.NATIVE, "AgenticOwnershipHooks._agentic_abort_cleanup(self, req)",
     "AgenticOwnershipHooks._agentic_abort_cleanup(req, self)"),
    (proof.NATIVE, "AgenticOwnershipHooks._record_agentic_forward_read_fence(self, batch)",
     "AgenticOwnershipHooks._record_agentic_forward_read_fence(self)"),
    (proof.NATIVE, "return AgenticOwnershipHooks._agentic_service_p_workset_leases(self)",
     "return None"),
    (proof.NATIVE, "        return AgenticOwnershipHooks._agentic_abort_cleanup(self, req)",
     "        try:\n            return AgenticOwnershipHooks._agentic_abort_cleanup(self, req)\n        except Exception:\n            return None"),
    (O, "import logging", "import logging\nimport sglang.srt.managers.scheduler"),
    (proof.TEST, 'monkeypatch.setattr(tp_module, "set", RankSet, raising=False)',
     'monkeypatch.setattr(scheduler_module, "set", RankSet, raising=False)'),
    (proof.TEST, "reverse=bool(rank)", "reverse=False"),
])
def test_candidate_proof_rejects_semantic_import_owner_and_gate_mutations(path, before, after):
    source = (proof.ROOT / path).read_text()
    assert source.count(before) == 1
    with pytest.raises(AssertionError):
        proof.verify({path: source.replace(before, after)})


def test_resolver_missing_or_partial_root_never_imports_or_selects_an_alias(monkeypatch):
    attempts = []
    class BlockNative(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == CANONICAL:
                attempts.append(fullname)
                raise AssertionError("resolver imported native Scheduler")
    blocker = BlockNative()
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])
    monkeypatch.delitem(sys.modules, CANONICAL, raising=False)
    foreign = ModuleType("other_fork.scheduler")
    foreign.Scheduler = object()
    monkeypatch.setitem(sys.modules, "other_fork.scheduler", foreign)
    lookup = resolver()
    with pytest.raises(RuntimeError):
        lookup()
    monkeypatch.setitem(sys.modules, CANONICAL, ModuleType(CANONICAL))
    with pytest.raises(RuntimeError):
        lookup()
    assert attempts == []


def test_every_method_and_constant_lookup_is_dynamic_and_accepts_namespace_owner(monkeypatch):
    root = ModuleType(CANONICAL)
    class First:
        pass
    root.Scheduler = First
    monkeypatch.setitem(sys.modules, CANONICAL, root)
    lookup = resolver()
    owner = SimpleNamespace()
    calls = []
    for attr in ("_agentic_tp_abort_mailbox_write",
                 "_agentic_tp_abort_mailbox_write",
                 "_agentic_early_direct_slots_used",
                 "_agentic_abort_tp_direct_grant",
                 "_agentic_abort_tp_direct_grant"):
        marker = object()
        def method(value, marker=marker):
            assert value is owner
            calls.append(marker)
            return marker
        monkeypatch.setattr(First, attr, staticmethod(method), raising=False)
        assert getattr(lookup(), attr)(owner) is marker
    assert len(calls) == 5
    for attr in ("_AGENTIC_TP_CONTROL_KEY", "_AGENTIC_V2_ACTIVATION_KEY"):
        for value in ("original", "patched"):
            monkeypatch.setattr(First, attr, value, raising=False)
            assert getattr(lookup(), attr) == value
    class Replacement:
        pass
    root.Scheduler = Replacement
    assert lookup() is Replacement
    root.Scheduler = First
    assert lookup() is First


@pytest.mark.parametrize("name", proof.OWNERSHIP)
def test_native_wrapper_forwards_namespace_args_return_and_exception_without_policy(name):
    # Compile only the exact candidate wrapper: this is not a native import or
    # proof of real allocator/fence behavior (that remains in the H100 gate).
    tree = ast.parse((proof.ROOT / proof.NATIVE).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    expected_owner = SimpleNamespace()
    extra = [object() for _ in method.args.args[1:]]
    marker = object()
    calls = []
    class Hooks:
        pass
    def implementation(*args):
        calls.append(args)
        assert args == (expected_owner, *extra)
        return marker
    setattr(Hooks, name, staticmethod(implementation))
    scope = dict(AgenticOwnershipHooks=Hooks, Req=object, ScheduleBatch=object)
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[copy.deepcopy(method)], type_ignores=[])), "<candidate wrapper>", "exec"), scope)
    assert scope[name](expected_owner, *extra) is marker
    assert calls == [(expected_owner, *extra)]
    failure = ValueError("original hook failure")
    def fail(*args):
        assert args == (expected_owner, *extra)
        raise failure
    setattr(Hooks, name, staticmethod(fail))
    with pytest.raises(ValueError) as captured:
        scope[name](expected_owner, *extra)
    assert captured.value is failure
