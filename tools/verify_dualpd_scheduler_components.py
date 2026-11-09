"""Complete native Scheduler reconstruction, including explicit integration hooks."""
import ast
import builtins
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import symtable

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = ROOT
BASE = "2dbb67ccc5c4c2a61a00440efe58831b08d1c08c"
NATIVE = "python/sglang/srt/managers/scheduler.py"
EXT = "python/sglang/srt/disaggregation/dualpd/"
TEST = EXT + "tests/test_agentic_tp_host_commit_order.py"
GROUPS = {
    "AgenticDirectMixin": """
_agentic_start_early_direct_receive _agentic_drop_early_direct_receive
_agentic_handle_unstarted_direct_abort _agentic_collect_direct_abort_events
_agentic_return_early_direct_to_slow _agentic_early_direct_progress_worker
_agentic_poll_early_direct_receives _agentic_collect_direct_arrivals
_agentic_early_direct_slots_used _agentic_admit_queued_direct_receives
_agentic_poll_early_direct_receives_once _agentic_bind_early_direct_receive
_agentic_try_finalize_early_direct_bind _agentic_finalize_early_direct_bind
_agentic_admit_early_direct_bind _agentic_rollback_prepared_direct_bind
_agentic_start_direct_load _agentic_return_request_direct_to_decode
_agentic_poll_direct_load _agentic_clear_direct_receiver""".split(),
    "AgenticTPMixin": """
_agentic_tp_abort_mailbox_write _agentic_mark_tp_direct_failed
_agentic_progress_tp_direct_grants _agentic_abort_tp_direct_grant
_agentic_commit_tp_direct_groups _agentic_fail_tp_direct_bind
_agentic_v2_prepare_activation_control _agentic_v2_consume_activation_control
_agentic_tp_reduce_direct_status _agentic_tp_reduce_host_status
_agentic_tp_reduce_workset_retire_status _agentic_tp_host_next_action
_agentic_tp_prepare_admission_control _agentic_tp_consume_admission_control
_agentic_tp_start_direct_shard""".split(),
    "AgenticReadyMixin": """
_agentic_refill_host_events _agentic_should_defer _agentic_queue_class
_agentic_io_active _agentic_io_kind _agentic_progress_granted_slow
_agentic_bind_completed_waiters _drain_agentic_kv_waiting_queue_tp1
_drain_agentic_kv_waiting_queue _prioritize_agentic_prefill_ready
_agentic_mark_p_host _agentic_abandon_load _agentic_consume_if_already_on_gpu
_agentic_publish_p_accepted _agentic_publish_p_scheduled""".split(),
}
FILENAMES = dict(zip(GROUPS, ["agentic_scheduler_direct.py",
                            "agentic_scheduler_tp.py", "agentic_scheduler_ready.py"]))
OWNERSHIP = [
    "_agentic_snapshot_store", "_agentic_service_p_workset_leases",
    "_agentic_abort_cleanup", "_record_agentic_forward_read_fence",
]
OWNERS = dict(GROUPS, AgenticOwnershipHooks=OWNERSHIP)
FILENAMES["AgenticOwnershipHooks"] = "agentic_scheduler_ownership.py"
EXPECTED_REFS = {
    "_agentic_handle_unstarted_direct_abort": {"_agentic_tp_abort_mailbox_write": 2},
    "_agentic_admit_queued_direct_receives": {"_agentic_early_direct_slots_used": 1},
    "_agentic_commit_tp_direct_groups": {"_agentic_abort_tp_direct_grant": 2},
    "_agentic_tp_consume_admission_control":
        {"_AGENTIC_TP_CONTROL_KEY": 1, "_AGENTIC_V2_ACTIVATION_KEY": 1},
}
COMPAT = '''"""Late concrete dispatch for native Scheduler integration, never an import."""

import sys


def _native_scheduler_class():
    module = sys.modules.get("sglang.srt.managers.scheduler")
    if module is None or not hasattr(module, "Scheduler"):
        raise RuntimeError("native Scheduler is not initialized")
    return module.Scheduler
'''


def global_names(source):
    names = set()
    def visit(table):
        names.update(s.get_name() for s in table.get_symbols()
                     if s.is_global() and s.is_referenced())
        for child in table.get_children():
            visit(child)
    visit(symtable.symtable(source, "<candidate>", "exec"))
    return names - set(vars(builtins))


def import_bindings(tree):
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                result.setdefault(alias.asname or alias.name.split(".")[0],
                                  ast.Import(names=[alias]))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                result[alias.asname or alias.name] = ast.ImportFrom(
                    module=node.module, names=[alias], level=node.level)
    return result


def dump(n):
    return ast.dump(n, include_attributes=False)


def original():
    return subprocess.check_output(
        ["git", "-C", str(REFERENCE_ROOT), "show", BASE + ":" + NATIVE], text=True)


def verify(overrides=None, *, return_tree=False):
    overrides = overrides or {}
    def read(path):
        return overrides.get(path, (ROOT / path).read_text())
    old = ast.parse(original())
    old_cls = next(n for n in old.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    old_methods = {n.name: n for n in old_cls.body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    old_record = next(n for n in old.body if isinstance(n, ast.ClassDef)
                      and n.name == "AgenticEarlyDirectReceive")
    bindings = import_bindings(old)
    assert dump(ast.parse(read(EXT + "agentic_scheduler_compat.py"))) == dump(ast.parse(COMPAT))
    moved = {}
    actual_record = None
    rewrite_count = 0
    for owner, names in OWNERS.items():
        leaf_source = read(EXT + FILENAMES[owner])
        tree = ast.parse(leaf_source)
        expected_classes = {owner} | (
            {"AgenticEarlyDirectReceive"} if owner == "AgenticDirectMixin" else set())
        assert {n.name for n in tree.body if isinstance(n, ast.ClassDef)} == expected_classes
        assert sum(isinstance(n, ast.ClassDef) for n in tree.body) == len(expected_classes)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
        assert not cls.bases and not cls.keywords and not cls.decorator_list
        assert all(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in cls.body)
        assert [n.name for n in cls.body] == names
        for method in cls.body:
            counts = {}
            class RestoreConcreteDispatch(ast.NodeTransformer):
                def visit_Attribute(self, node):
                    if (isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Name)
                        and node.value.func.id == "_native_scheduler_class"):
                        assert not node.value.args and not node.value.keywords
                        counts[node.attr] = counts.get(node.attr, 0) + 1
                        node.value = ast.Name(id="Scheduler", ctx=ast.Load())
                    return self.generic_visit(node)
            restored = RestoreConcreteDispatch().visit(copy.deepcopy(method))
            assert counts == EXPECTED_REFS.get(method.name, {}), (method.name, counts)
            rewrite_count += sum(counts.values())
            assert dump(restored) == dump(old_methods[method.name]), method.name
            moved[method.name] = restored
        records = [n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "AgenticEarlyDirectReceive"]
        if owner == "AgenticDirectMixin":
            assert len(records) == 1 and dump(records[0]) == dump(old_record)
            actual_record = records[0]
            identity = ast.parse(
                'AgenticEarlyDirectReceive.__module__ = "sglang.srt.managers.scheduler"').body[0]
        else:
            assert not records
            identity = None
        # Independent original-source dependency inventory, not candidate imports.
        original_owner = ast.ClassDef(
            name=owner, bases=[], keywords=[], decorator_list=[],
            body=[old_methods[name] for name in names])
        original_nodes = ([old_record] if owner == "AgenticDirectMixin" else []) + [original_owner]
        deps = global_names(ast.unparse(ast.Module(body=original_nodes, type_ignores=[])))
        deps -= expected_classes | {"Scheduler", "logger"}
        expected_imports = ast.parse("import logging").body
        for dep in sorted(deps):
            if dep == "AgenticEarlyDirectReceive":
                expected_imports += ast.parse(
                    "from sglang.srt.disaggregation.agentic_scheduler_direct import AgenticEarlyDirectReceive").body
            else:
                assert dep in bindings, (owner, dep)
                expected_imports.append(bindings[dep])
        if any(name in EXPECTED_REFS for name in names):
            expected_imports += ast.parse(
                "from sglang.srt.disaggregation.agentic_scheduler_compat import _native_scheduler_class").body
        expected_nodes = ast.parse(
            '"""' + owner + ': shared native Scheduler state; no independent state owner."""').body
        expected_nodes += expected_imports
        expected_nodes += ast.parse(
            'logger = logging.getLogger("sglang.srt.managers.scheduler")').body
        if identity:
            expected_nodes += [records[0], identity]
        expected_nodes += [cls]
        assert dump(tree) == dump(ast.Module(body=expected_nodes, type_ignores=[])), (
            owner, "Imports/provenance/order or extra executable definitions changed")
        bound = set(import_bindings(tree)) | expected_classes | {"logger"}
        assert not global_names(leaf_source) - bound, (owner, global_names(leaf_source) - bound)
    assert len(moved) == 54 and rewrite_count == 7
    current = ast.parse(read(NATIVE))
    idx = old.body.index(old_record)
    imports = ast.parse("""from sglang.srt.disaggregation.agentic_scheduler_direct import (
    AgenticDirectMixin, AgenticEarlyDirectReceive,
)
from sglang.srt.disaggregation.agentic_scheduler_tp import AgenticTPMixin
from sglang.srt.disaggregation.agentic_scheduler_ready import AgenticReadyMixin
from sglang.srt.disaggregation.agentic_scheduler_ownership import AgenticOwnershipHooks
""").body
    assert dump(ast.Module(body=current.body[idx:idx + 4], type_ignores=[])) == dump(
        ast.Module(body=imports, type_ignores=[]))
    current.body[idx:idx + 4] = [actual_record]
    cls = next(n for n in current.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    assert [ast.unparse(n) for n in cls.bases[:3]] == list(GROUPS)
    cls.bases = cls.bases[3:]
    actual_wrappers = {n.name: n for n in cls.body
                       if isinstance(n, ast.FunctionDef) and n.name in OWNERSHIP}
    assert set(actual_wrappers) == set(OWNERSHIP)
    for name in OWNERSHIP:
        expected_wrapper = copy.deepcopy(old_methods[name])
        doc = ([expected_wrapper.body[0]] if isinstance(expected_wrapper.body[0], ast.Expr)
               and isinstance(expected_wrapper.body[0].value, ast.Constant)
               and isinstance(expected_wrapper.body[0].value.value, str) else [])
        expected_wrapper.body = doc + [ast.Return(value=ast.Call(
            func=ast.Attribute(value=ast.Name(id="AgenticOwnershipHooks", ctx=ast.Load()),
                               attr=name, ctx=ast.Load()),
            args=[ast.Name(id=arg.arg, ctx=ast.Load()) for arg in expected_wrapper.args.args],
            keywords=[]))]
        assert dump(actual_wrappers[name]) == dump(expected_wrapper), (name, "Native wrapper changed")
    cls.body = [moved[n.name] if isinstance(n, ast.FunctionDef) and n.name in OWNERSHIP
                else n for n in cls.body]
    distributed = set().union(*GROUPS.values())
    remaining = [n for n in old_cls.body
                 if not (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in distributed)]
    assert dump(ast.Module(body=cls.body, type_ignores=[])) == dump(
        ast.Module(body=remaining, type_ignores=[]))
    actual_remaining = iter(cls.body)
    cls.body = [
        moved[n.name] if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in distributed
        else next(actual_remaining) for n in old_cls.body]
    assert dump(current) == dump(old), "Entire native module reconstruction failed"
    old_test = subprocess.check_output(
        ["git", "-C", str(REFERENCE_ROOT), "show", BASE + ":" + TEST], text=True)
    for before, after in [
        ("import sglang.srt.managers.scheduler as scheduler_module",
         "import sglang.srt.disaggregation.agentic_scheduler_tp as tp_module"),
        ('monkeypatch.setattr(scheduler_module, "set", RankSet, raising=False)',
         'monkeypatch.setattr(tp_module, "set", RankSet, raising=False)'),
    ]:
        assert old_test.count(before) == 1
        old_test = old_test.replace(before, after)
    assert dump(ast.parse(read(TEST))) == dump(ast.parse(old_test)), "Existing RankSet test/assertions changed"
    result = {"reference_commit": BASE, "whole_native_module_reconstructed": True,
            "method_groups": {key: len(value) for key, value in OWNERS.items()},
            "complete_methods_checked": len(moved), "exact_concrete_lookups": rewrite_count,
            "native_explicit_wrappers_verified": len(OWNERSHIP),
            "eager_direct_record_ast_identical": True,
            "native_mixed_methods_ast_identical": True,
            "imports_and_global_bindings_verified": True,
            "compat_complete_ast_verified": True,
            "existing_rankset_assertions_unchanged": True,
            "scope": "Source reconstruction only; native functionality and performance are separate gates."}
    return (result, current) if return_tree else result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
