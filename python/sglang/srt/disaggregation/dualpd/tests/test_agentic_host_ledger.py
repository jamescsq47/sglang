"""Ledger-only regression and old import identity after the module split."""
import os
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile

from sglang.srt.disaggregation.agentic_host_ledger import (
    HostStageState, SharedHostStagingLedger,
)


def test_historical_imports_resolve_to_same_classes():
    from sglang.srt.disaggregation import agentic_host_staging as staging
    from sglang.srt.disaggregation import agentic_host_ledger as ledger

    for name in ("HostStageState", "PermanentHostStageError", "SharedHostStagingLedger",
                 "_ALLOWED_STAGE_TRANSITIONS", "_TERMINAL_STATES", "_LEDGER_OWNERSHIP_CHANGED"):
        assert getattr(staging, name) is getattr(ledger, name)


def test_host_ledger_replay_keeps_claim_and_state():
    # Match the production contract: control ledgers live only in tmpfs.
    with tempfile.TemporaryDirectory(prefix="dualpd-ledger-test-", dir="/dev/shm") as directory:
        ledger = SharedHostStagingLedger(str(Path(directory) / "ledger.json"))
        entry = {"snapshot_id": "request:0", "token_count": 64, "token_digest": "digest",
                 "tp_size": 1, "tp_rank": 0, "d_pid": os.getpid()}
        assert ledger.offer(entry)["state"] == HostStageState.OFFERED.value
        assert ledger.claim(entry["snapshot_id"], "p:0")["p_owner"] == "p:0"
        ledger.offer(entry)
        assert ledger.get(entry["snapshot_id"])["state"] == HostStageState.HOST_RESERVED.value
        assert ledger.get(entry["snapshot_id"])["p_owner"] == "p:0"


def test_old_and_new_pickle_class_paths_resolve_to_same_objects(monkeypatch):
    from sglang.srt.disaggregation import agentic_host_staging as staging

    for cls in (HostStageState, SharedHostStagingLedger):
        assert pickle.loads(pickle.dumps(cls)) is cls
        with monkeypatch.context() as patch:
            patch.setattr(cls, "__module__", staging.__name__)
            legacy_pickle = pickle.dumps(cls)
        assert pickle.loads(legacy_pickle) is cls
    with monkeypatch.context() as patch:
        patch.setattr(HostStageState, "__module__", staging.__name__)
        legacy_value = pickle.dumps(HostStageState.HOST_READY)
    assert pickle.loads(legacy_value) is HostStageState.HOST_READY


def test_ledger_import_does_not_initialize_cuda_or_frontend(tmp_path):
    code = '''
import importlib.abc, importlib.util, sys
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "sgl_kernel", "dualpd", "slime"}:
            raise AssertionError("ledger tried to import " + fullname)
sys.meta_path.insert(0, BlockRuntime())
spec = importlib.util.spec_from_file_location("ledger_without_engine", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert module.SharedHostStagingLedger.VERSION == 1
'''
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    source = sys.modules[SharedHostStagingLedger.__module__].__file__
    result = subprocess.run([sys.executable, "-c", code, source], cwd=tmp_path, env=env,
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
