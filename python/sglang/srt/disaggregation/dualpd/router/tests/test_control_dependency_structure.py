"""Destructive mutations are isolated in temporary fixtures, never serving code."""
import importlib.util
from pathlib import Path
import shutil
import tempfile
from unittest import TestCase, main

ENGINE = next(p for p in Path(__file__).resolve().parents if (p / "tools/verify_dualpd_control_dependencies.py").is_file())
spec = importlib.util.spec_from_file_location("control_dependency_candidate_proof", ENGINE / "tools/verify_dualpd_control_dependencies.py")
proof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proof)


def copy_fixture(root):
    relative = "python/sglang/srt/disaggregation/dualpd/"
    paths = ["p2d_host_staging.py", "p2d_host_protocol.py"] + [
        "router/" + n for n in ("late_binding_router.py", "local_admission.py",
                                 "local_capacity.py", "local_types.py", "local_dependencies.py")]
    for name in paths:
        source = ENGINE / relative / name
        target = root / relative / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


class CandidateProof(TestCase):
    def test_rejects_protocol_import_moved_to_worker_end(self):
        with tempfile.TemporaryDirectory(prefix="dualpd-control-import-order-") as directory:
            root = Path(directory) / "sglang_pd"
            copy_fixture(root)
            path = root / "python/sglang/srt/disaggregation/dualpd/p2d_host_staging.py"
            source = path.read_text()
            block = ("from sglang.srt.disaggregation.p2d_host_protocol import (\n"
                     "    P2D_CUSTOM_SNAPSHOT_ID,\n"
                     "    P2D_CUSTOM_PREFILL_DOMAIN,\n"
                     "    p2d_snapshot_id,\n)\n")
            self.assertEqual(source.count(block), 1)
            path.write_text(source.replace(block, "") + "\n" + block)
            with self.assertRaisesRegex(AssertionError, "import timing"):
                proof.verify(root)

    def test_actual_maintained_source(self):
        result = proof.verify()
        self.assertEqual(result["entire_router_modules_reconstructed"], 4)
        self.assertEqual(result["protocol_original_nodes"], 3)

    def test_rejects_exact_import_body_protocol_worker_and_bridge_mutations(self):
        ext = "python/sglang/srt/disaggregation/dualpd/"
        cases = [
            ("root import owner", "router/late_binding_router.py", "from .local_dependencies import (", "from .wrong_owner import ("),
            ("capacity import owner", "router/local_capacity.py", "from .local_dependencies import HostStageState", "from .wrong_owner import HostStageState"),
            ("admission wire alias", "router/local_admission.py", "    P2D_CUSTOM_SNAPSHOT_ID,", "    P2D_CUSTOM_SNAPSHOT_ID as wrong_key,"),
            ("types host alias", "router/local_types.py", "import HostStageState", "import HostStageState as wrong_state"),
            ("root new executable", "router/late_binding_router.py", "logger = logging.getLogger(__name__)", "logger = logging.getLogger(__name__)\nextra_state = {}"),
            ("protocol id format", "p2d_host_protocol.py", 'return f"p2d:{int(bootstrap_room)}"', 'return f"wrong:{int(bootstrap_room)}"'),
            ("protocol extra state", "p2d_host_protocol.py", "from __future__ import annotations", "from __future__ import annotations\nextra_owner = {}"),
            ("protocol future alias", "p2d_host_protocol.py", "from __future__ import annotations", "from __future__ import annotations as renamed"),
            ("worker capacity default", "p2d_host_staging.py", "P2D_HOST_CAPACITY_LIMIT_DEFAULT = 1.0", "P2D_HOST_CAPACITY_LIMIT_DEFAULT = 0.5"),
            ("worker protocol import alias", "p2d_host_staging.py", "    p2d_snapshot_id,", "    p2d_snapshot_id as wrong_function,"),
            ("bridge private guard", "router/local_dependencies.py", '__package__ != "dualpd_engine_late_binding_router"', 'False'),
            ("bridge catches broad errors", "router/local_dependencies.py", "except ModuleNotFoundError as exc:", "except ImportError as exc:"),
        ]
        with tempfile.TemporaryDirectory(prefix="dualpd-control-proof-") as directory:
            root = Path(directory) / "sglang_pd"
            copy_fixture(root)
            for label, relative, before, after in cases:
                with self.subTest(mutation=label):
                    path = root / ext / relative
                    original = path.read_text()
                    self.assertIn(before, original)
                    try:
                        path.write_text(original.replace(before, after))
                        with self.assertRaises(AssertionError):
                            proof.verify(root)
                    finally:
                        path.write_text(original)


if __name__ == "__main__":
    main(verbosity=2)
