"""Router services must not depend on the experiment/client repository."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", [
    "late_binding_router", "global_pd_router", "legacy_mamba_router",
])
def test_router_import_without_frontend(module, tmp_path):
    code = '''
import importlib, importlib.abc, sys
class BlockFrontend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"dualpd", "slime"}:
            raise AssertionError("engine tried to import " + fullname)
sys.meta_path.insert(0, BlockFrontend())
module = importlib.import_module(sys.argv[1])
if hasattr(module, "load_router_module"):
    pinned, digest = module.load_router_module()
    assert pinned.LateBindingMiniLoadBalancer
assert not any(n.split(".")[0] in {"dualpd", "slime"} for n in sys.modules)
'''
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-c", code,
        f"sglang.srt.disaggregation.dualpd.router.{module}"], cwd=tmp_path,
        env=env, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_package_import_does_not_activate_service_modules():
    code = '''
import sys
import sglang.srt.disaggregation.dualpd.router
assert "sglang.srt.disaggregation.dualpd.router.global_pd_router" not in sys.modules
assert "sglang.srt.disaggregation.dualpd.router.late_binding_router" not in sys.modules
'''
    result = subprocess.run([sys.executable, "-c", code], text=True,
                            capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
