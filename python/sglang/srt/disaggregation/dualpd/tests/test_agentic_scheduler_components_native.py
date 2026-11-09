"""Actual native module identity/import tests; run on the owned H100 node."""
import importlib
import importlib.util
from pathlib import Path
import pickle
import subprocess
import sys
from types import MethodType, SimpleNamespace
import typing

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "tools/verify_dualpd_scheduler_components.py").is_file())
spec = importlib.util.spec_from_file_location(
    "native_scheduler_component_proof", ROOT / "tools/verify_dualpd_scheduler_components.py")
proof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proof)


def modules():
    owners = {
        owner: importlib.import_module(
            "sglang.srt.disaggregation." + filename.removesuffix(".py"))
        for owner, filename in proof.FILENAMES.items()
    }
    native = importlib.import_module("sglang.srt.managers.scheduler")
    return native, owners


def test_real_native_kernel_is_not_a_cpu_stub():
    import sgl_kernel
    assert Path(sgl_kernel.common_ops.__file__).suffix == ".so"


def test_real_native_mro_method_owners_annotations_and_logger_identity():
    native, owners = modules()
    assert [len(names) for names in proof.OWNERS.values()] == [20, 15, 15, 4]
    assert native.Scheduler.__bases__[:3] == tuple(
        getattr(owners[owner], owner) for owner in proof.GROUPS)
    for owner, names in proof.OWNERS.items():
        module = owners[owner]
        cls = getattr(module, owner)
        assert cls.__bases__ == (object,)
        assert "__init__" not in vars(cls)
        assert module.logger is native.logger
        for name in names:
            method = getattr(cls, name)
            typing.get_type_hints(method)
            if name in proof.OWNERSHIP:
                assert name in vars(native.Scheduler)
            else:
                assert name not in vars(native.Scheduler)
                assert getattr(native.Scheduler, name) is method
    record = owners["AgenticDirectMixin"].AgenticEarlyDirectReceive
    assert native.AgenticEarlyDirectReceive is record
    assert record.__module__ == "sglang.srt.managers.scheduler"
    typing.get_type_hints(record)
    assert pickle.loads(b"csglang.srt.managers.scheduler\nAgenticEarlyDirectReceive\n.") is record
    assert pickle.loads(pickle.dumps(record, protocol=0)) is record
    from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration
    from sglang.srt.disaggregation.agentic_workset import AgenticPWorksetLease
    assert record.__annotations__["workset_lease"] == typing.Optional[AgenticPWorksetLease]
    value = record(request=RequestGeneration("identity", 0), manifest=None,
                   claim_id="claim", receiver=None, device_indices=None,
                   started_at=0.0, arrived_at=0.0)
    restored = pickle.loads(pickle.dumps(value, protocol=2))
    assert type(restored) is record and restored == value


def test_cold_native_component_imports_do_not_load_scheduler_back():
    code = """
import importlib, importlib.abc, sys
class BlockScheduler(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sglang.srt.managers.scheduler":
            raise AssertionError("component imports Scheduler back")
sys.meta_path.insert(0, BlockScheduler())
for name in ("direct", "tp", "ready", "ownership", "compat"):
    importlib.import_module("sglang.srt.disaggregation.agentic_scheduler_" + name)
assert "sglang.srt.managers.scheduler" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


@pytest.mark.parametrize("name", proof.OWNERSHIP)
def test_real_native_wrapper_preserves_unbound_namespace_return_exception_and_patches(name, monkeypatch):
    native, owners = modules()
    implementation = owners["AgenticOwnershipHooks"].AgenticOwnershipHooks
    wrapper = getattr(native.Scheduler, name)
    extra = [object()] if name in {"_agentic_abort_cleanup", "_record_agentic_forward_read_fence"} else []
    owner = SimpleNamespace()
    marker = object()
    def operation(*args):
        assert args == (owner, *extra)
        return marker
    monkeypatch.setattr(implementation, name, staticmethod(operation))
    assert wrapper(owner, *extra) is marker
    failure = RuntimeError("original native hook failure")
    def fail(*args):
        assert args == (owner, *extra)
        raise failure
    monkeypatch.setattr(implementation, name, staticmethod(fail))
    with pytest.raises(RuntimeError) as captured:
        wrapper(owner, *extra)
    assert captured.value is failure
    instance = object.__new__(native.Scheduler)
    class_marker, instance_marker = object(), object()
    monkeypatch.setattr(native.Scheduler, name, lambda self, *args: class_marker)
    assert getattr(instance, name)(*extra) is class_marker
    setattr(instance, name, MethodType(lambda self, *args: instance_marker, instance))
    assert getattr(instance, name)(*extra) is instance_marker
