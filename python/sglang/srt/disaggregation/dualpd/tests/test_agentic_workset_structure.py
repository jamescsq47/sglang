"""Complete native reconstruction and historical workset identity contracts."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "tools/verify_dualpd_scheduler_cleanup.py").is_file())
spec = importlib.util.spec_from_file_location(
    "workset_reconstruction", ROOT / "tools/verify_dualpd_scheduler_cleanup.py")
proof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proof)


def test_complete_native_scheduler_reconstruction():
    result = proof.verify()
    assert result["whole_native_scheduler_reconstructed"]
    assert result["broker_methods_ast_identical"] == 39


@pytest.mark.parametrize("path,before,after", [
    (proof.WORKSET, "import threading", ""),
    (proof.WORKSET, "import numpy as np", "import json as np"),
    (proof.WORKSET, "from sglang.srt.disaggregation.utils import kv_to_page_indices",
     "from sglang.srt.disaggregation.utils import is_npu as kv_to_page_indices"),
    (proof.WORKSET, "import threading", "from __future__ import annotations\nimport threading"),
    (proof.WORKSET, 'state: str = "active"', 'state: str = "consumed"'),
    (proof.WORKSET, "parent_tokens <= 0", "parent_tokens < 0"),
    (proof.WORKSET, "parent_page_indices: np.ndarray", "parent_page_indices: str"),
    (proof.WORKSET, 'AgenticPWorksetLease.__module__ = "sglang.srt.managers.scheduler"',
     'AgenticPWorksetLease.__module__ = __name__'),
    (proof.WORKSET, 'AgenticPWorksetLeaseBroker.__module__ = "sglang.srt.managers.scheduler"',
     'AgenticPWorksetLeaseBroker.__module__ = __name__'),
    (proof.WORKSET, "import threading", "import threading\n_extra_lock = threading.RLock()"),
    (proof.SCHEDULER, "    AgenticPWorksetLeaseBroker,",
     "    AgenticPWorksetLeaseBroker as OtherBroker,"),
    (proof.SCHEDULER, "logger = logging.getLogger(__name__)",
     "logger = logging.getLogger('changed-native-hook')"),
    (proof.DIRECT_RECORD, "prefill_domain: Optional[int] = None\n    workset_lease: Optional[AgenticPWorksetLease] = None",
     "prefill_domain: Optional[int] = None\n    workset_lease: Optional[str] = None"),
    (proof.TEST, "assert len(calls) == 2", "assert len(calls) >= 0"),
    (proof.TEST, "monkeypatch.setattr(workset_module, 'kv_to_page_indices'",
     "monkeypatch.setattr(sched_module, 'kv_to_page_indices'"),
])
def test_reconstruction_rejects_incomplete_or_changed_extraction(path, before, after):
    source = (ROOT / path).read_text()
    assert source.count(before) == 1
    with pytest.raises(AssertionError):
        proof.verify({path: source.replace(before, after)})


def test_workset_cold_import_has_no_scheduler_dependency():
    code = """
import importlib.abc, sys
class NoScheduler(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'sglang.srt.managers.scheduler':
            raise AssertionError('workset imports Scheduler back')
sys.meta_path.insert(0, NoScheduler())
from sglang.srt.disaggregation import agentic_workset as workset
assert 'sglang.srt.managers.scheduler' not in sys.modules
assert workset.AgenticPWorksetLeaseBroker.service.__globals__ is vars(workset)
assert workset.AgenticPWorksetLease.__module__ == 'sglang.srt.managers.scheduler'
assert workset.AgenticPWorksetLeaseBroker.__module__ == 'sglang.srt.managers.scheduler'
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
