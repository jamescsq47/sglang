import threading
import unittest
from types import SimpleNamespace

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.nixl.conn import NixlKVManager, NixlKVSender
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNixlSenderFailureCleanup(unittest.TestCase):
    def test_reverse_disable_does_not_prepare_handles(self):
        manager = NixlKVManager.__new__(NixlKVManager)
        manager.disable_prepped_kv = True
        manager.is_mla_backend = False
        manager.attn_tp_size = 2
        manager._init_equal_tp_prep_handle = lambda *_args, **_kwargs: self.fail(
            "reverse manager must not create prepped handles"
        )
        manager._init_hetero_tp_prep_handle = lambda *_args, **_kwargs: self.fail(
            "reverse manager must not create sliced prepped handles"
        )

        manager._prepare_payload_xfer(
            SimpleNamespace(decode_tp_size=2)
        )

    @staticmethod
    def _attention_manager(*, disable_prepped: bool):
        class _Agent:
            def __init__(self):
                self.prepped_calls = 0
                self.explicit_calls = 0

            def make_prepped_xfer(self, *_args):
                self.prepped_calls += 1
                return "prepped-handle"

            def get_xfer_descs(self, requests, _memory_type):
                self.explicit_calls += 1
                return requests

            def initialize_xfer(self, *_args):
                return "explicit-handle"

            def transfer(self, _handle):
                return "DONE"

        manager = NixlKVManager.__new__(NixlKVManager)
        manager.agent = _Agent()
        manager.disable_prepped_kv = disable_prepped
        manager.kv_args = SimpleNamespace(
            gpu_id=0,
            prefill_start_layer=0,
            kv_data_ptrs=[1000, 2000],
        )
        manager.is_mla_backend = False
        manager.prep_handles = {"": "src-prepped", "peer": "dst-prepped"}
        manager.decode_kv_args_table = {
            "peer": SimpleNamespace(dst_num_slots=16)
        }
        manager._num_slots_src = 16
        return manager

    def test_reverse_disable_ignores_stale_prepped_handles(self):
        manager = self._attention_manager(disable_prepped=True)

        manager._send_kvcache_generic(
            peer_name="peer",
            src_data_ptrs=manager.kv_args.kv_data_ptrs,
            dst_data_ptrs=[3000, 4000],
            item_lens=[10, 10],
            prefill_data_indices=np.asarray([0, 1], dtype=np.int32),
            dst_data_indices=np.asarray([2, 3], dtype=np.int32),
            dst_gpu_id=1,
            notif="1_kv_0_1_0",
        )

        self.assertEqual(manager.agent.prepped_calls, 0)
        self.assertEqual(manager.agent.explicit_calls, 2)

    def test_normal_manager_keeps_prepped_path(self):
        manager = self._attention_manager(disable_prepped=False)

        manager._send_kvcache_generic(
            peer_name="peer",
            src_data_ptrs=manager.kv_args.kv_data_ptrs,
            dst_data_ptrs=[3000, 4000],
            item_lens=[10, 10],
            prefill_data_indices=np.asarray([0, 1], dtype=np.int32),
            dst_data_indices=np.asarray([2, 3], dtype=np.int32),
            dst_gpu_id=1,
            notif="1_kv_0_1_0",
        )

        self.assertEqual(manager.agent.prepped_calls, 1)
        self.assertEqual(manager.agent.explicit_calls, 0)

    def test_mamba_state_transfer_carries_active_and_checkpoint_slots(self):
        captured = {}

        class _Agent:
            def get_xfer_descs(self, addrs, memory_type):
                captured.setdefault(memory_type, []).append(addrs)
                return addrs

            def initialize_xfer(self, *_args):
                return "handle"

            def transfer(self, handle):
                self.handle = handle
                return "DONE"

        manager = NixlKVManager.__new__(NixlKVManager)
        manager.agent = _Agent()
        manager.kv_args = SimpleNamespace(gpu_id=0)
        manager._send_mamba_state(
            "peer",
            [2, 3],
            [1000, 2000],
            [100, 200],
            [3000, 4000],
            [5, 7],
            1,
            "state",
        )

        src, dst = captured["VRAM"]
        self.assertEqual(
            src, [(1200, 100, 0), (1300, 100, 0), (2400, 200, 0), (2600, 200, 0)]
        )
        self.assertEqual(
            dst, [(3500, 100, 1), (3700, 100, 1), (5000, 200, 1), (5400, 200, 1)]
        )

    def test_unqueryable_failed_handle_does_not_block_other_room_fence(self):
        manager = NixlKVManager.__new__(NixlKVManager)
        manager.agent = SimpleNamespace(
            check_xfer_state=lambda handle: (
                (_ for _ in ()).throw(RuntimeError("unqueryable"))
                if handle == "stuck"
                else "DONE"
            )
        )
        manager._failed_fence_lock = threading.Lock()
        manager._failed_fence_rooms = {1, 2}
        completed = []
        manager._publish_fenced_transfer_failure = lambda room, error: completed.append(
            (room, error)
        )

        pending = manager._poll_failed_fences_once(
            [
                (1, ("stuck",), RuntimeError("first")),
                (2, ("done",), RuntimeError("second")),
            ]
        )

        self.assertEqual([item[0] for item in pending], [1])
        self.assertEqual([item[0] for item in completed], [2])
        self.assertEqual(manager._failed_fence_rooms, {1})

    def test_failure_exception_cleans_room_state_before_raising(self):
        room = 7
        expected_exc = RuntimeError("transfer failed")
        sender = NixlKVSender.__new__(NixlKVSender)
        sender.bootstrap_room = room
        sender.conclude_state = None
        sender._send_failed = False
        sender._send_error = None
        staging_ctx = SimpleNamespace(
            prefetched_rooms={room, 8},
            prefetch_requested={(room, 0, "session-a"), (8, 0, "session-b")},
        )
        sender.kv_mgr = SimpleNamespace(
            enable_staging=True,
            _staging_ctx=staging_ctx,
            request_status={room: object()},
            req_to_decode_prefix_len={room: 3},
            transfer_infos={room: object()},
            exceptions={room: expected_exc},
            failure_records={room: "transfer failed"},
            failure_lock=threading.Lock(),
        )

        with self.assertRaises(RuntimeError) as cm:
            sender.failure_exception()

        self.assertIs(cm.exception, expected_exc)
        self.assertTrue(sender._send_failed)
        self.assertEqual(sender.conclude_state, KVPoll.Failed)
        self.assertNotIn(room, sender.kv_mgr.request_status)
        self.assertNotIn(room, sender.kv_mgr.req_to_decode_prefix_len)
        self.assertNotIn(room, sender.kv_mgr.transfer_infos)
        self.assertNotIn(room, sender.kv_mgr.exceptions)
        self.assertNotIn(room, sender.kv_mgr.failure_records)
        self.assertNotIn(room, staging_ctx.prefetched_rooms)
        self.assertNotIn((room, 0, "session-a"), staging_ctx.prefetch_requested)
        self.assertIn(8, staging_ctx.prefetched_rooms)
        self.assertIn((8, 0, "session-b"), staging_ctx.prefetch_requested)


if __name__ == "__main__":
    unittest.main()
