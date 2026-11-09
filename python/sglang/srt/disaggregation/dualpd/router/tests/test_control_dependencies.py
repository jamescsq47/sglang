"""Control-plane resolution faults and actual maintained Router import closure."""
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

SOURCE = Path(__file__).parents[1].joinpath("local_dependencies.py").read_text()
PREFIX = "sglang.srt.disaggregation."
PRIVATE = "dualpd_engine_late_binding_router"
CANONICAL = "sglang.srt.disaggregation.dualpd.router"


def bindings(tag):
    return SimpleNamespace(
        HostStageState=type("State_" + tag, (), {}),
        SharedHostStagingLedger=type("Ledger_" + tag, (), {}),
        P2D_CUSTOM_PREFILL_DOMAIN="domain_" + tag,
        P2D_CUSTOM_SNAPSHOT_ID="snapshot_" + tag,
        p2d_snapshot_id=lambda value: (tag, value),
    )


class DependencyResolution(TestCase):
    def setUp(self):
        self.modern_ledger = bindings("modern_ledger")
        self.modern_protocol = bindings("modern_protocol")
        self.old_ledger = bindings("old_ledger")
        self.old_protocol = bindings("old_protocol")
        self.modules = {
            PREFIX + "agentic_host_ledger": self.modern_ledger,
            PREFIX + "p2d_host_protocol": self.modern_protocol,
            PREFIX + "agentic_host_staging": self.old_ledger,
            PREFIX + "p2d_host_staging": self.old_protocol,
        }
        self.calls = []

    def run_candidate(self, package, failures=None):
        failures = failures or {}

        def selected_import(name):
            self.calls.append(name)
            if name in failures:
                raise failures[name]
            if name not in self.modules:
                raise ModuleNotFoundError("missing " + name, name=name)
            return self.modules[name]

        candidate = ModuleType(package + ".local_dependencies")
        candidate.__package__ = package
        with patch("importlib.import_module", selected_import):
            exec(compile(SOURCE, "local_dependencies.py", "exec"), candidate.__dict__)
        return candidate

    def assert_bindings(self, candidate, ledger, protocol):
        for name in ("HostStageState", "SharedHostStagingLedger"):
            self.assertIs(getattr(candidate, name), getattr(ledger, name))
        for name in (
            "P2D_CUSTOM_PREFILL_DOMAIN", "P2D_CUSTOM_SNAPSHOT_ID", "p2d_snapshot_id"
        ):
            self.assertIs(getattr(candidate, name), getattr(protocol, name))

    def test_private_four_leaf_availability_combinations(self):
        for ledger_present in (False, True):
            for protocol_present in (False, True):
                with self.subTest(ledger=ledger_present, protocol=protocol_present):
                    self.calls.clear()
                    failures = {}
                    for leaf, present in (
                        ("agentic_host_ledger", ledger_present),
                        ("p2d_host_protocol", protocol_present),
                    ):
                        if not present:
                            full = PREFIX + leaf
                            failures[full] = ModuleNotFoundError(full, name=full)
                    candidate = self.run_candidate(PRIVATE, failures)
                    self.assert_bindings(
                        candidate,
                        self.modern_ledger if ledger_present else self.old_ledger,
                        self.modern_protocol if protocol_present else self.old_protocol,
                    )
                    expected = [PREFIX + "agentic_host_ledger"]
                    if not ledger_present:
                        expected.append(PREFIX + "agentic_host_staging")
                    expected.append(PREFIX + "p2d_host_protocol")
                    if not protocol_present:
                        expected.append(PREFIX + "p2d_host_staging")
                    self.assertEqual(self.calls, expected)

    def test_canonical_modern_objects_without_legacy_import(self):
        candidate = self.run_candidate(CANONICAL)
        self.assert_bindings(candidate, self.modern_ledger, self.modern_protocol)
        self.assertEqual(self.calls, [PREFIX + "agentic_host_ledger", PREFIX + "p2d_host_protocol"])

    def test_canonical_missing_either_leaf_never_falls_back(self):
        for leaf in ("agentic_host_ledger", "p2d_host_protocol"):
            with self.subTest(leaf=leaf):
                self.calls.clear()
                full = PREFIX + leaf
                failure = ModuleNotFoundError(full, name=full)
                with self.assertRaises(ModuleNotFoundError) as error:
                    self.run_candidate(CANONICAL, {full: failure})
                self.assertIs(error.exception, failure)
                self.assertNotIn(PREFIX + "agentic_host_staging", self.calls)
                self.assertNotIn(PREFIX + "p2d_host_staging", self.calls)

    def test_private_propagates_other_failures_unchanged(self):
        for leaf in ("agentic_host_ledger", "p2d_host_protocol"):
            for failure in (
                ModuleNotFoundError("nested dependency", name="unrelated_dependency"),
                ModuleNotFoundError("missing parent", name="sglang.srt.disaggregation"),
                ImportError("broken extension"),
                RuntimeError("runtime failure"),
            ):
                with self.subTest(leaf=leaf, failure=repr(failure)):
                    self.calls.clear()
                    with self.assertRaises(type(failure)) as error:
                        self.run_candidate(PRIVATE, {PREFIX + leaf: failure})
                    self.assertIs(error.exception, failure)
                    self.assertNotIn(PREFIX + "agentic_host_staging", self.calls)
                    self.assertNotIn(PREFIX + "p2d_host_staging", self.calls)

    def test_private_module_with_missing_symbol_does_not_fall_back(self):
        for leaf in ("agentic_host_ledger", "p2d_host_protocol"):
            with self.subTest(leaf=leaf):
                original = self.modules[PREFIX + leaf]
                self.modules[PREFIX + leaf] = SimpleNamespace()
                self.calls.clear()
                try:
                    with self.assertRaises(AttributeError):
                        self.run_candidate(PRIVATE)
                    self.assertNotIn(PREFIX + "agentic_host_staging", self.calls)
                    self.assertNotIn(PREFIX + "p2d_host_staging", self.calls)
                finally:
                    self.modules[PREFIX + leaf] = original

    def test_similar_package_name_cannot_enable_legacy_fallback(self):
        full = PREFIX + "agentic_host_ledger"
        failure = ModuleNotFoundError(full, name=full)
        for package in (PRIVATE + "_other", "other." + PRIVATE, CANONICAL):
            with self.subTest(package=package):
                with self.assertRaises(ModuleNotFoundError) as error:
                    self.run_candidate(package, {full: failure})
                self.assertIs(error.exception, failure)

    def test_broken_ledger_symbols_stop_before_protocol_resolution(self):
        self.modules[PREFIX + "agentic_host_ledger"] = SimpleNamespace()
        full = PREFIX + "p2d_host_protocol"
        with self.assertRaises(AttributeError):
            self.run_candidate(PRIVATE, {full: ModuleNotFoundError(full, name=full)})
        self.assertEqual(self.calls, [PREFIX + "agentic_host_ledger"])

    def test_legacy_fallback_failure_propagates_unchanged(self):
        for leaf, legacy in (
            ("agentic_host_ledger", "agentic_host_staging"),
            ("p2d_host_protocol", "p2d_host_staging"),
        ):
            for failure in (ModuleNotFoundError("old dependency", name="old_dependency"),
                            RuntimeError("old runtime failed")):
                with self.subTest(leaf=leaf, failure=repr(failure)):
                    self.calls.clear()
                    full = PREFIX + leaf
                    with self.assertRaises(type(failure)) as error:
                        self.run_candidate(PRIVATE, {
                            full: ModuleNotFoundError(full, name=full),
                            PREFIX + legacy: failure,
                        })
                    self.assertIs(error.exception, failure)
                    self.assertEqual(self.calls[-2:], [full, PREFIX + legacy])


class CanonicalImportClosure(TestCase):
    def test_fresh_canonical_router_without_transport_or_frontend(self):
        import os
        import subprocess
        import sys
        import tempfile

        code = r'''
import asyncio,importlib,importlib.abc,pickle,sys
from pathlib import Path
import aiohttp
leaves = ("agentic_host_staging","p2d_host_staging","agentic_host_buffers",
          "agentic_host_dma","agentic_host_registration","agentic_host_writer")
blocked = {"sglang.srt.disaggregation." + leaf for leaf in leaves}
blocked |= {"sglang.srt.disaggregation.dualpd." + leaf for leaf in leaves}
blocked.add("sglang.srt.managers.scheduler")
class NoPhysicalRuntime(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname in blocked or any(fullname.startswith(n + ".") for n in blocked):
   raise AssertionError("control plane imported physical runtime: " + fullname)
  if fullname.split(".")[0] in {"dualpd","slime"}:
   raise AssertionError("control plane imported experiment frontend: " + fullname)
assert not blocked.intersection(sys.modules)
assert not any(n.split(".")[0] in {"dualpd","slime"} for n in sys.modules)
sys.meta_path.insert(0,NoPhysicalRuntime())
def forbidden(*args,**kwargs):
 raise AssertionError("Router import created a task/session")
asyncio.create_task = forbidden
aiohttp.ClientSession = forbidden
root = importlib.import_module("sglang.srt.disaggregation.dualpd.router.late_binding_router")
assert Path(root.__file__).resolve() == Path(sys.argv[1]).resolve()
package = "sglang.srt.disaggregation.dualpd.router"
deps = importlib.import_module(package + ".local_dependencies")
ledger = importlib.import_module("sglang.srt.disaggregation.agentic_host_ledger")
protocol = importlib.import_module("sglang.srt.disaggregation.p2d_host_protocol")
source_directory = Path(root.__file__).resolve().parent.parent
assert Path(ledger.__file__).resolve() == source_directory / "agentic_host_ledger.py"
assert Path(protocol.__file__).resolve() == source_directory / "p2d_host_protocol.py"
for name in ("late_binding_router","local_capacity","local_admission","local_types"):
 module = importlib.import_module(package + "." + name)
 assert module.HostStageState is deps.HostStageState is ledger.HostStageState
assert root.SharedHostStagingLedger is deps.SharedHostStagingLedger is ledger.SharedHostStagingLedger
admission = importlib.import_module(package + ".local_admission")
for name in ("P2D_CUSTOM_PREFILL_DOMAIN","P2D_CUSTOM_SNAPSHOT_ID","p2d_snapshot_id"):
 assert getattr(root,name) is getattr(admission,name) is getattr(deps,name) is getattr(protocol,name)
assert protocol.p2d_snapshot_id(7) == "p2d:7"
assert protocol.p2d_snapshot_id.__module__ == "sglang.srt.disaggregation.p2d_host_protocol"
assert pickle.loads(pickle.dumps(protocol.p2d_snapshot_id)) is protocol.p2d_snapshot_id
assert not blocked.intersection(sys.modules)
assert not any(n.split(".")[0] in {"dualpd","slime"} for n in sys.modules)
'''
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        with tempfile.TemporaryDirectory(prefix="dualpd-canonical-import-") as directory:
            result = subprocess.run(
                [sys.executable, "-c", code,
                 str(Path(__file__).parents[1] / "late_binding_router.py")],
                env=env, cwd=directory, text=True, capture_output=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    main(verbosity=2)
