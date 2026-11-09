"""Reconstruct the complete native Scheduler, not just extracted methods."""

import ast
import builtins
import hashlib
import json
from pathlib import Path
import subprocess
import symtable

ROOT = Path(__file__).resolve().parents[1]
BASE = "a995eb0afd596ba8eebbaf9377c51693541b96ca"
SCHEDULER = "python/sglang/srt/managers/scheduler.py"
WORKSET = "python/sglang/srt/disaggregation/dualpd/agentic_workset.py"
TEST = "python/sglang/srt/disaggregation/dualpd/tests/test_agentic_h2d_decoupling.py"
NAMES = ("AgenticPWorksetLease", "AgenticPWorksetLeaseBroker")
IMPORT = """from sglang.srt.disaggregation.agentic_workset import (
    AgenticPWorksetLease,
    AgenticPWorksetLeaseBroker,
)"""
HEADER = '''"""Scheduler-owned complete P worksets for Direct and Host restoration.

Only the model scheduler services physical allocation and release. Transport
workers publish intents and fenced transitions through this shared broker.
The historical Scheduler aliases remain the public serialization identity.
"""

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from sglang.srt.disaggregation.utils import kv_to_page_indices
'''
ALIASES = '''
AgenticPWorksetLease.__module__ = "sglang.srt.managers.scheduler"
AgenticPWorksetLeaseBroker.__module__ = "sglang.srt.managers.scheduler"
'''


def original(path):
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f"{BASE}:{path}"], text=True)


def dump(node):
    return ast.dump(node, include_attributes=False)


def verify(overrides=None):
    overrides = overrides or {}
    def read(path):
        return overrides[path] if path in overrides else (ROOT / path).read_text()

    old = ast.parse(original(SCHEDULER))
    moved = [n for n in old.body if isinstance(n, ast.ClassDef) and n.name in NAMES]
    assert tuple(n.name for n in moved) == NAMES
    start = old.body.index(moved[0])
    assert old.body[start:start + 2] == moved, "Non-contiguous workset owner"
    leaf_source = read(WORKSET)
    leaf = ast.parse(leaf_source)
    expected_leaf = ast.parse(HEADER).body + moved + ast.parse(ALIASES).body
    assert dump(leaf) == dump(ast.Module(body=expected_leaf, type_ignores=[])), (
        "Workset bodies, imports, eager annotations or historical identities changed")
    # Check referenced globals independently: whole-body equality alone would
    # otherwise miss a removed/wrong defining-module import.
    bound = set(NAMES) | {"__name__", "__file__", "__package__", "__doc__"}
    for n in leaf.body:
        if isinstance(n, ast.Import):
            bound.update(a.asname or a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            bound.update(a.asname or a.name for a in n.names)
    referenced = set()
    def visit(table):
        referenced.update(s.get_name() for s in table.get_symbols()
                          if s.is_global() and s.is_referenced())
        for child in table.get_children():
            visit(child)
    visit(symtable.symtable(leaf_source, WORKSET, "exec"))
    assert not referenced - bound - set(vars(builtins)), "Unbound workset global"

    current = ast.parse(read(SCHEDULER))
    alias = ast.parse(IMPORT).body[0]
    assert dump(current.body[start]) == dump(alias), "Alias/import timing changed"
    reconstructed = ast.Module(
        body=current.body[:start] + moved + current.body[start + 1:],
        type_ignores=current.type_ignores)
    assert dump(reconstructed) == dump(old), (
        "Native Scheduler module, mixed hooks, imports or initialization changed")

    expected_test = original(TEST)
    replacements = [
        ("from sglang.srt.managers import scheduler as sched_module",
         "from sglang.srt.managers import scheduler as sched_module\n"
         "from sglang.srt.disaggregation import agentic_workset as workset_module"),
        ("original = sched_module.kv_to_page_indices",
         "original = workset_module.kv_to_page_indices"),
        ("monkeypatch.setattr(sched_module, 'kv_to_page_indices'",
         "monkeypatch.setattr(workset_module, 'kv_to_page_indices'"),
    ]
    for before, after in replacements:
        assert expected_test.count(before) == 1
        expected_test = expected_test.replace(before, after)
    assert dump(ast.parse(expected_test)) == dump(ast.parse(read(TEST))), (
        "Existing H2D test changed beyond defining-module hook retarget")
    source_map = json.loads((ROOT / "dualpd-source-map.json").read_text())
    for path in (SCHEDULER, TEST):
        entry = next(item for item in source_map["files"] if item["path"] == path)
        assert entry["current_sha256"] == hashlib.sha256(read(path).encode()).hexdigest()
    return {
        "reference_commit": BASE,
        "whole_native_scheduler_reconstructed": True,
        "workset_classes_ast_identical": len(moved),
        "broker_methods_ast_identical": sum(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in moved[1].body),
        "eager_annotations_and_alias_timing_preserved": True,
        "historical_serialization_identity_preserved": True,
        "existing_h2d_assertions_unchanged": True,
        "runtime_global_bindings_verified": True,
        "remaining_scheduler_responsibilities": "Direct, TP coordination, ready/admission",
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
