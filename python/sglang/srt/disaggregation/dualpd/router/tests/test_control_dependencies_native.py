"""Real-kernel/native compatibility gates; run explicitly on the owned H100.

The old-overlay case requires DUALPD_TEST_LEGACY_ENGINE_ROOT pointing to a
frozen legacy python/ tree genuinely lacking both new pure leaves. It never
installs, patches, writes bytecode into, or launches workers from that tree.
"""
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from typing import get_type_hints


def test_actual_native_public_aliases_and_literal_old_pickle_paths():
    import sgl_kernel
    assert Path(sgl_kernel.common_ops.__file__).suffix == ".so"
    from sglang.srt.disaggregation import agentic_host_ledger as ledger
    from sglang.srt.disaggregation import agentic_host_staging as staging
    from sglang.srt.disaggregation import p2d_host_protocol as protocol
    from sglang.srt.disaggregation import p2d_host_staging as worker
    from sglang.srt.disaggregation.dualpd.router import late_binding_router as root
    from sglang.srt.disaggregation.dualpd.router import local_dependencies as dependencies

    assert Path(worker.__file__).resolve() == Path(__file__).parents[2].joinpath("p2d_host_staging.py").resolve()
    for name in ("HostStageState", "SharedHostStagingLedger"):
        assert getattr(staging, name) is getattr(ledger, name) is getattr(root, name) is getattr(dependencies, name)
        literal = ("csglang.srt.disaggregation.agentic_host_staging\n" + name + "\n.").encode()
        assert pickle.loads(literal) is getattr(ledger, name)
    for name in ("P2D_CUSTOM_PREFILL_DOMAIN", "P2D_CUSTOM_SNAPSHOT_ID", "p2d_snapshot_id"):
        assert getattr(worker, name) is getattr(protocol, name) is getattr(root, name) is getattr(dependencies, name)
    assert pickle.loads(b"csglang.srt.disaggregation.p2d_host_staging\np2d_snapshot_id\n.") is protocol.p2d_snapshot_id
    assert pickle.loads(pickle.dumps(protocol.p2d_snapshot_id)) is protocol.p2d_snapshot_id
    assert protocol.p2d_snapshot_id.__module__ == "sglang.srt.disaggregation.p2d_host_protocol"
    assert get_type_hints(protocol.p2d_snapshot_id) == {"bootstrap_room": int, "return": str}


def test_real_selected_legacy_overlay_with_missing_pure_leaves():
    legacy = Path(os.environ["DUALPD_TEST_LEGACY_ENGINE_ROOT"])
    assert legacy.is_absolute() and legacy.joinpath("sglang/srt/disaggregation/agentic_host_staging.py").is_file()
    code = r'''
import asyncio,importlib,importlib.util,json,pickle,sys
from pathlib import Path
import aiohttp
import sgl_kernel
assert Path(sgl_kernel.common_ops.__file__).suffix == ".so"
import sglang.srt.disaggregation as native
legacy = Path(sys.argv[2]).resolve()
assert [Path(p).resolve() for p in native.__path__] == [
 legacy / "sglang/srt/disaggregation"]
for leaf in ("agentic_host_ledger","p2d_host_protocol"):
 assert importlib.util.find_spec("sglang.srt.disaggregation." + leaf) is None
# These are actual selected-old modules, not maintained preimports or fake types.
old_host = importlib.import_module("sglang.srt.disaggregation.agentic_host_staging")
old_p2d = importlib.import_module("sglang.srt.disaggregation.p2d_host_staging")
assert Path(old_host.__file__).resolve().is_relative_to(legacy)
assert Path(old_p2d.__file__).resolve().is_relative_to(legacy)
before_path = list(sys.path)
before_native = list(native.__path__)
def forbidden(*args,**kwargs):
 raise AssertionError("maintained Router import created a task/session")
asyncio.create_task = forbidden
aiohttp.ClientSession = forbidden
source = Path(sys.argv[1])
alias = "dualpd_engine_late_binding_router"
spec = importlib.util.spec_from_file_location(alias,source,submodule_search_locations=[str(source.parent)])
root = importlib.util.module_from_spec(spec)
sys.modules[alias] = root
spec.loader.exec_module(root)
dependencies = sys.modules[alias + ".local_dependencies"]
for name in ("HostStageState","SharedHostStagingLedger"):
 assert getattr(root,name) is getattr(dependencies,name) is getattr(old_host,name)
 literal = ("csglang.srt.disaggregation.agentic_host_staging\n" + name + "\n.").encode()
 assert pickle.loads(literal) is getattr(old_host,name)
for leaf in ("local_capacity","local_admission","local_types"):
 assert sys.modules[alias + "." + leaf].HostStageState is old_host.HostStageState
for name in ("P2D_CUSTOM_PREFILL_DOMAIN","P2D_CUSTOM_SNAPSHOT_ID","p2d_snapshot_id"):
 assert getattr(root,name) is getattr(dependencies,name) is getattr(old_p2d,name)
assert pickle.loads(b"csglang.srt.disaggregation.p2d_host_staging\np2d_snapshot_id\n.") is old_p2d.p2d_snapshot_id
assert sys.modules["sglang.srt.disaggregation"] is native
assert list(native.__path__) == before_native
assert sys.path == before_path
assert root.logger.name == alias
for leaf in ("agentic_host_ledger","p2d_host_protocol"):
 assert "sglang.srt.disaggregation." + leaf not in sys.modules
assert "sglang.srt.disaggregation.dualpd" not in sys.modules
assert not any(name.split(".")[0] in {"dualpd","slime"} for name in sys.modules)
print(json.dumps({"actual_old_overlay":str(legacy),"same_native_objects":True,"native_namespace_preserved":True}))
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(legacy)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory(prefix="dualpd-real-legacy-import-") as directory:
        result = subprocess.run(
            [sys.executable, "-c", code,
             str(Path(__file__).parents[1] / "late_binding_router.py"), str(legacy)],
            env=env, cwd=directory, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
