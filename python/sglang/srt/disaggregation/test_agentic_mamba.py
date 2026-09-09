import os
import queue
import tempfile
import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.disaggregation.agentic_direct_transfer import _make_kv_args
from sglang.srt.disaggregation.agentic_host_staging import (
    AgenticDHostStagingClient,
    AgenticPHostStagingManager,
    HostStageState,
    SharedHostStagingLedger,
)
from sglang.srt.disaggregation.agentic_tp_control import TPGroupMailbox
from sglang.srt.disaggregation.agentic_workset import AgenticPWorksetLeaseBroker
from sglang.srt.disaggregation.agentic_hybrid_snapshot import (
    HybridSnapshotLayout,
    SharedHybridHostSnapshot,
    SharedMambaHostSnapshot,
)
from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    RequestGeneration,
    SnapshotManifest,
    SnapshotState,
)
from sglang.srt.disaggregation.agentic_hybrid_transfer import (
    complete_snapshot_bytes,
    debug_mamba_digest,
    freeze_p2d_mamba_checkpoint_after_cache,
    mamba_checkpoint_index_for_req,
    p2d_mamba_destination_indices,
    p2d_mamba_source_indices,
    snapshot_token_count_for_req,
    state_indices_for_req,
    state_indices_for_workset,
    submit_reverse_receive,
    submit_reverse_send,
    validate_agentic_mamba_tracking,
)
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.agentic_decode_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.disaggregation.nixl.conn import NixlKVSender
import sglang.srt.disaggregation.p2d_host_staging as p2d_host_module
import sglang.srt.disaggregation.agentic_direct_transfer as direct_transfer_module
import sglang.srt.disaggregation.agentic_hybrid_transfer as hybrid_transfer_module
from sglang.srt.disaggregation.p2d_host_staging import (
    AgenticPToDHostLoadManager,
)
from sglang.srt.disaggregation.utils import TransferBackend
import sglang.srt.managers.schedule_batch as schedule_batch_module
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)


class _FakeKVPool:
    start_layer = 0
    head_num = 4
    page_size = 64

    @staticmethod
    def get_contiguous_buf_infos():
        return [1000, 2000], [4096, 4096], [128, 128]


class _FakeHybridReqPool:
    @staticmethod
    def get_state_buf_infos():
        # Temporal state and convolution state, one item of each per request.
        return [3000, 4000], [8192, 2048], [512, 128]

    @staticmethod
    def get_state_dim_per_tensor():
        return [4, 3]


def _server_args():
    return SimpleNamespace(
        disaggregation_ib_device="",
        disaggregation_ib_traffic_class="",
    )


def test_reverse_direct_runtime_describes_complete_qwen35_state():
    args, _aux = _make_kv_args(
        transfer_backend=TransferBackend.NIXL,
        kv_pool=_FakeKVPool(),
        server_args=_server_args(),
        engine_rank=0,
        pp_rank=0,
        gpu_id=0,
        total_kv_heads=8,
        req_to_token_pool=_FakeHybridReqPool(),
    )

    assert args.state_types == [StateType.MAMBA]
    assert args.state_data_ptrs == [[3000, 4000]]
    assert args.state_data_lens == [[8192, 2048]]
    assert args.state_item_lens == [[512, 128]]
    assert args.state_dim_per_tensor == [[4, 3]]


def test_dense_reverse_runtime_has_no_auxiliary_state_component():
    args, _aux = _make_kv_args(
        transfer_backend=TransferBackend.NIXL,
        kv_pool=_FakeKVPool(),
        server_args=_server_args(),
        engine_rank=0,
        pp_rank=0,
        gpu_id=0,
        total_kv_heads=8,
    )

    assert args.state_types == []
    assert args.state_data_ptrs == []


class _Allocator:
    def __init__(self, available=256):
        self.available = available
        self.cursor = 0
        self.freed = []

    def available_size(self):
        return self.available

    def alloc(self, count):
        if count > self.available:
            return None
        result = torch.arange(self.cursor, self.cursor + count, dtype=torch.int64)
        self.cursor += count
        self.available -= count
        return result

    def free(self, indices):
        self.available += int(indices.numel())
        self.freed.append(indices.clone())


def test_agentic_decode_chunk_cache_release_frees_hybrid_mamba_state():
    """A consumed reverse snapshot releases both KV and all Mamba slots."""

    events = []
    req_pool = HybridReqToTokenPool.__new__(HybridReqToTokenPool)
    req_pool.req_to_token = torch.arange(16, dtype=torch.int64).reshape(1, 16)
    req_pool.free_mamba_cache = lambda req: events.append(("mamba", req.rid))
    req_pool.free = lambda req: events.append(("request", req.rid))
    manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
    manager.tree_cache = SimpleNamespace(disable=True, protected_size_=0)
    manager.req_to_token_pool = req_pool
    manager.token_to_kv_pool_allocator = SimpleNamespace(
        free=lambda indices: events.append(("kv", indices.tolist()))
    )
    manager.page_size = 1
    manager.offloaded_state = {}
    req = SimpleNamespace(
        rid="req",
        req_pool_idx=0,
        prefix_indices=torch.tensor([], dtype=torch.int64),
        pop_committed_kv_cache=lambda: 8,
        pop_overallocated_kv_cache=lambda: (8, 8),
    )

    manager._release_finished_req(req, 0)

    assert events == [("kv", list(range(8))), ("mamba", "req"), ("request", "req")]


def test_hybrid_workset_grant_and_handoff_are_one_composite_ownership_unit():
    kv_allocator = _Allocator()
    mamba_allocator = _Allocator(available=2)
    broker = AgenticPWorksetLeaseBroker(
        page_size=64, state_allocators=[mamba_allocator]
    )
    broker.request("req:1", parent_tokens=65, prompt_tokens=70)
    broker.service(kv_allocator)
    lease = broker.get("req:1")

    assert lease is not None
    assert lease.parent_allocated_tokens == 128
    assert len(lease.state_device_indices) == 1
    assert lease.state_device_indices[0].numel() == 1

    assert broker.begin_bind("req:1", lease)
    broker.commit_parent_bound("req:1", lease, state_donated_to_radix=True)
    req = SimpleNamespace(origin_input_ids=list(range(70)), mamba_pool_idx=None)
    broker.attach_runtime_state_for_bind("req:1", req, lease)
    broker.handoff_to_req("req:1", req, lease)
    # Radix owns the received checkpoint; the Req owns a distinct, atomically
    # pre-reserved active slot and COWs the parent into it before Forward.
    assert req.mamba_pool_idx is not None
    assert lease.state_device_indices == ()
    assert lease.runtime_state_device_indices == ()
    assert req.mamba_last_track_seqlen == lease.parent_tokens
    assert req.mamba_last_track_seqlen == 65
    assert mamba_allocator.freed == []


def test_reverse_parent_short_suffix_preserves_checkpoint_through_p2d():
    kv_allocator = _Allocator()
    mamba_allocator = _Allocator(available=8)
    state = torch.zeros(1, 8, 1)

    class ReqPool:
        enable_mamba_extra_buffer = True
        enable_mamba_extra_buffer_lazy = False
        mamba_ping_pong_track_buffer_size = 2

        @staticmethod
        def get_mamba_ping_pong_keep_idx(_req):
            return 1

    req_pool = ReqPool()
    broker = AgenticPWorksetLeaseBroker(
        page_size=64,
        state_allocators=[mamba_allocator],
        mamba_req_to_token_pool=req_pool,
    )
    broker.request("short-suffix:1", parent_tokens=128, prompt_tokens=150)
    broker.service(kv_allocator)
    lease = broker.get("short-suffix:1")
    assert lease is not None
    received_checkpoint = int(lease.state_device_indices[0][0])
    state[:, received_checkpoint].fill_(42)

    assert broker.begin_bind("short-suffix:1", lease)
    broker.commit_parent_bound("short-suffix:1", lease, state_donated_to_radix=True)
    req = SimpleNamespace(origin_input_ids=list(range(150)), mamba_pool_idx=None)
    broker.attach_runtime_state_for_bind("short-suffix:1", req, lease)
    # Native match_prefix points at the checkpoint donated to Radix.
    req.mamba_cow_src_index = torch.tensor([received_checkpoint])
    broker.stage_runtime_checkpoint_cow_for_bind("short-suffix:1", req, lease)
    sources = req.mamba_cow_src_index
    destinations = req._agentic_mamba_cow_dst_indices
    state[:, destinations] = state[:, sources]
    broker.handoff_to_req("short-suffix:1", req, lease)

    # A 22-token suffix does not cross another 64-token track boundary.  Both
    # the active state and retained parent checkpoint must nevertheless be
    # initialized from the received generation.
    assert torch.all(state[:, int(req.mamba_pool_idx)] == 42)
    keep = req_pool.get_mamba_ping_pong_keep_idx(req)
    assert torch.all(state[:, int(req.mamba_ping_pong_track_buffer[keep])] == 42)

    # cache_unfinished donates that checkpoint into the locked Radix node and
    # clears mamba_last_track_seqlen.  P->D must use the post-cache node, not
    # the replacement (uninitialized) ping-pong slot.
    req.last_node = SimpleNamespace(mamba_value=torch.tensor([received_checkpoint]))
    req.mamba_last_track_seqlen = None
    freeze_p2d_mamba_checkpoint_after_cache(req, 128, 64)
    payload = p2d_mamba_source_indices(req, 64)[0]
    assert payload.tolist() == [int(req.mamba_pool_idx), received_checkpoint]
    assert torch.all(state[:, int(payload[1])] == 42)


def test_batched_agentic_cow_reexpands_native_rematched_sources():
    """A later native match may collapse each staged 1-to-2 COW source."""

    reqs = [
        SimpleNamespace(
            mamba_cow_src_index=torch.tensor([3]),
            _agentic_mamba_cow_dst_indices=torch.tensor([10, 11]),
            mamba_pool_idx=torch.tensor(10),
            mamba_needs_clear=False,
        ),
        SimpleNamespace(
            mamba_cow_src_index=torch.tensor([7]),
            _agentic_mamba_cow_dst_indices=torch.tensor([20, 21]),
            mamba_pool_idx=torch.tensor(20),
            mamba_needs_clear=False,
        ),
    ]
    batch = ScheduleBatch.__new__(ScheduleBatch)
    batch._collect_deferred_mamba_cow_and_clear(reqs)

    assert batch.mamba_cow_src_indices.tolist() == [3, 3, 7, 7]
    assert batch.mamba_cow_dst_indices.tolist() == [10, 11, 20, 21]
    assert all(req.mamba_cow_src_index is None for req in reqs)
    assert all(not hasattr(req, "_agentic_mamba_cow_dst_indices") for req in reqs)


def test_agentic_cow_rejects_ambiguous_source_destination_counts():
    req = SimpleNamespace(
        mamba_cow_src_index=torch.tensor([3, 4]),
        _agentic_mamba_cow_dst_indices=torch.tensor([10, 11, 12]),
        mamba_pool_idx=torch.tensor(10),
        mamba_needs_clear=False,
    )
    batch = ScheduleBatch.__new__(ScheduleBatch)

    with pytest.raises(RuntimeError, match="source/destination count mismatch"):
        batch._collect_deferred_mamba_cow_and_clear([req])

    # Fail before consuming retryable request metadata.
    assert req.mamba_cow_src_index.tolist() == [3, 4]
    assert req._agentic_mamba_cow_dst_indices.tolist() == [10, 11, 12]


def test_hybrid_workset_rolls_back_attention_when_mamba_slot_is_unavailable():
    kv_allocator = _Allocator()
    mamba_allocator = _Allocator(available=0)
    broker = AgenticPWorksetLeaseBroker(
        page_size=64, state_allocators=[mamba_allocator]
    )
    broker.request("req:2", parent_tokens=64, prompt_tokens=65)
    broker.service(kv_allocator)

    assert broker.get("req:2") is None
    assert len(kv_allocator.freed) == 1
    assert kv_allocator.available == 256


def test_tp2_mamba_allocation_failure_retires_attention_and_state_group():
    """One rank's state failure cannot leave its peer's hybrid lease admitted."""

    attention = [_Allocator(available=16), _Allocator(available=16)]
    # A hybrid lease needs one parent checkpoint and one runtime state slot.
    # Rank 1 can allocate the first slot but not the second, exercising the
    # partial-state rollback path after its Attention allocation succeeded.
    state = [_Allocator(available=4), _Allocator(available=1)]
    ranks = [
        AgenticPWorksetLeaseBroker(page_size=4, state_allocators=[state[rank]])
        for rank in range(2)
    ]
    snapshot_id = "hybrid-tp-allocation-failure:1"
    owner = ranks[0].direct_owner(snapshot_id)
    assert ranks[0].request(snapshot_id, 4, 8, owner=owner)
    plan = ranks[0].prepare_tp_plan(1)
    ranks[1].install_tp_plan(1, plan)

    for broker, allocator in zip(ranks, attention):
        broker.service(allocator)

    assert ranks[0].get(snapshot_id, owner=owner) is not None
    assert ranks[1].get(snapshot_id, owner=owner) is None
    assert attention[1].available == 16
    assert state[1].available == 1

    # TP0 observes that not every rank granted the composite workset and
    # broadcasts one retirement.  Neither rank may hand off a partial lease.
    for broker in ranks:
        assert broker.prepare_tp_retire(snapshot_id)
        assert broker.tp_retire_ready(snapshot_id)
    for broker in ranks:
        assert broker.commit_tp_retire(snapshot_id)
    for broker, allocator in zip(ranks, attention):
        broker.service(allocator)

    assert all(broker.get(snapshot_id) is None for broker in ranks)
    assert [allocator.available for allocator in attention] == [16, 16]
    assert [allocator.available for allocator in state] == [4, 1]


def test_tp2_direct_state_error_waits_for_all_physical_handles_and_ranks():
    """Attention DONE plus Mamba ERR is terminal only after every handle/rank."""

    attention_handle, conv_handle, temporal_handle = object(), object(), object()
    handle_states = {
        attention_handle: "DONE",
        conv_handle: "ERR",
        temporal_handle: "PROC",
    }
    sender = NixlKVSender.__new__(NixlKVSender)
    sender.kv_mgr = SimpleNamespace(
        agent=SimpleNamespace(
            check_xfer_state=lambda handle: handle_states[handle]
        )
    )
    sender.bootstrap_room = 1
    sender.xfer_handles = [attention_handle, conv_handle, temporal_handle]
    sender.has_sent = True
    sender.launch_failed = False
    sender.launch_exception = None
    sender._send_failed = False

    assert sender.poll() == KVPoll.Transferring
    handle_states[temporal_handle] = "DONE"
    assert sender.poll() == KVPoll.Failed

    with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
        mailboxes = [
            TPGroupMailbox(
                "hybrid-direct-failure",
                tp_rank=rank,
                tp_size=2,
                directory=directory,
            )
            for rank in range(2)
        ]
        key = "hybrid-direct-failure:1"
        mailboxes[0].publish_local(key, int(KVPoll.Failed))
        mailboxes[1].publish_local(key, int(KVPoll.Transferring))
        status, cancel = mailboxes[0].transfer_group_status(key)
        assert status == int(KVPoll.Transferring)
        assert cancel is True

        mailboxes[1].publish_local(key, int(KVPoll.Success))
        status, cancel = mailboxes[0].transfer_group_status(key)
        assert status == int(KVPoll.Failed)
        assert cancel is True


def test_tp2_slow_h2d_state_failure_keeps_host_and_rearms_complete_snapshot():
    """A Mamba H2D failure cannot publish or bind an Attention-only prefix."""

    request = RequestGeneration("hybrid-host-h2d-failure", 1)
    retry_reasons = []
    quiesced = []

    class FailedStateEvent:
        @staticmethod
        def query():
            return True

        @staticmethod
        def synchronize():
            raise RuntimeError("injected Mamba H2D failure")

    class Ledger:
        @staticmethod
        def get(_snapshot_id):
            return {"state": HostStageState.H2D_LOADING.value}

        @staticmethod
        def request_d2p_retry(_snapshot_id, _owner, *, reason):
            retry_reasons.append(reason)
            return True

    record = {
        "snapshot": SimpleNamespace(),
        "offer": {"token_count": 64, "byte_size": 4096},
        "loading": "h2d",
    }
    load = {
        "record": record,
        "request_generation": request,
        "workset_lease": object(),
        "io_attempt": "hybrid-state-h2d",
        "io_error": None,
        "io_complete": False,
        "h2d_copy_complete": False,
        "start_allowed": True,
        "event": None,
        "state_event": FailedStateEvent(),
        "state_copy_refs": [object()],
        "state_loaded": False,
    }
    manager = SimpleNamespace(
        _h2d_poisoned=False,
        _get_state_lock=nullcontext,
        loads={"child": load},
        ledger=Ledger(),
        tp_size=2,
        owner="p-group:p0",
        workset_broker=SimpleNamespace(
            mark_io_quiesced=lambda *_args: quiesced.append(True) or True
        ),
        _discard_failed_h2d_load=lambda *_args: None,
        _publish_d2p_hbm_ready=lambda *_args: pytest.fail(
            "partial hybrid snapshot became visible"
        ),
    )

    AgenticPHostStagingManager._progress_h2d_loads(manager)

    assert isinstance(load["io_error"], RuntimeError)
    assert retry_reasons == ["slow_h2d_failed:RuntimeError"]
    assert quiesced == []
    assert manager.loads == {"child": load}
    assert record["loading"] == "h2d"


def test_slow_h2d_debug_digest_observes_workset_destination(monkeypatch, caplog):
    """Slow diagnostics hash the restored workset slots, never live Req state."""

    page_indices = torch.tensor([5, 6], dtype=torch.int64)
    state_indices = [torch.tensor([17], dtype=torch.int64)]
    seen = {}

    def page_digest(kv_pool, indices):
        seen["kv_pool"] = kv_pool
        seen["page_indices"] = indices
        return "attention"

    def state_digest(req_pool, indices):
        seen["req_pool"] = req_pool
        seen["state_indices"] = indices
        return "conv=abc,temporal=def"

    monkeypatch.setattr(
        direct_transfer_module, "debug_kv_page_digests", page_digest
    )
    monkeypatch.setattr(hybrid_transfer_module, "debug_mamba_digest", state_digest)
    runtime = SimpleNamespace(kv_pool=object(), req_to_token_pool=None)
    broker_req_pool = object()
    manager = SimpleNamespace(
        runtime=runtime,
        tp_rank=1,
        workset_broker=SimpleNamespace(_mamba_req_to_token_pool=broker_req_pool),
    )
    load = {
        "request_generation": RequestGeneration("slow-debug", 3),
        "device_indices": page_indices,
        "workset_lease": SimpleNamespace(state_device_indices=state_indices),
    }

    with caplog.at_level("INFO"):
        AgenticPHostStagingManager._debug_log_restored_hybrid_snapshot(manager, load)

    assert seen == {
        "kv_pool": runtime.kv_pool,
        "page_indices": page_indices,
        "req_pool": broker_req_pool,
        "state_indices": state_indices,
    }
    assert "p_host_received_page_digests snapshot=slow-debug:3 rank=1" in caplog.text
    assert "p_host_received_state_digest snapshot=slow-debug:3 rank=1" in caplog.text


def test_debug_mamba_digest_accepts_tensor_state_indices(monkeypatch):
    """Workset leases expose tensor indices (CUDA in production)."""

    monkeypatch.setenv("SGLANG_AGENTIC_KV_DEBUG_DIGEST", "1")
    cache = SimpleNamespace(
        conv=[torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)],
        temporal=torch.arange(18, dtype=torch.float32).reshape(2, 3, 3),
    )
    req_pool = SimpleNamespace(mamba_pool=SimpleNamespace(mamba_cache=cache))

    digest = debug_mamba_digest(req_pool, [torch.tensor([1], dtype=torch.int64)])

    assert digest is not None
    assert digest.startswith("conv=")
    assert ",temporal=" in digest


def test_tp2_slow_d2h_state_failure_never_publishes_attention_only_host():
    """A failed Mamba D2H event leaves D as owner and Host non-durable."""

    snapshot_id = "hybrid-host-d2h-failure:1"
    durable = []
    cleaned = []

    class FailedStateEvent:
        @staticmethod
        def query():
            return True

        @staticmethod
        def synchronize():
            raise RuntimeError("injected Mamba D2H failure")

    write = {
        "launch_error": None,
        "state_event": FailedStateEvent(),
        "state_copy_refs": [object()],
        "state_written": False,
        # Attention has already reached the Host extent.
        "event": None,
        "offset": 64,
        "chunk_end": 64,
        "gpu_elapsed_ms": 1.0,
    }
    candidate = {
        "manifest": SimpleNamespace(snapshot_id=snapshot_id),
        "arena_write": write,
        "req": SimpleNamespace(req_pool_idx=1),
    }
    client = AgenticDHostStagingClient.__new__(AgenticDHostStagingClient)
    client.tp_rank = 0
    client.tp_size = 2
    client.relay_enabled = False
    client.ledger = SimpleNamespace(
        complete_host_write=lambda *_args, **_kwargs: durable.append(True) or True
    )
    client._cleanup_write = lambda *_args: cleaned.append(True) or True

    with pytest.raises(RuntimeError, match="injected Mamba D2H failure"):
        client.progress(
            candidate,
            torch.arange(64),
            entry_snapshot={
                "state": HostStageState.HOST_WRITING.value,
                "write_mode": "direct",
            },
        )

    assert durable == []
    assert cleaned == []
    assert candidate["arena_write"] is write
    assert candidate["req"].req_pool_idx == 1


def test_tp2_p2d_host_state_failure_waits_for_group_terminal_before_cleanup(
    monkeypatch,
):
    """P->D Attention success plus Mamba failure cannot expose a D request."""

    snapshot_id = "hybrid-p2d-state-failure:1"
    state = {"value": HostStageState.H2D_LOADING.value}
    failures = []
    drained = []
    closed = []
    terminal = []

    class SuccessfulEvent:
        @staticmethod
        def synchronize():
            return None

    class FailedStateEvent:
        @staticmethod
        def synchronize():
            raise RuntimeError("injected P2D Mamba H2D failure")

    class Snapshot:
        @staticmethod
        def start_load_range_to_device(*_args, **_kwargs):
            return SuccessfulEvent(), [object()]

        @staticmethod
        def start_load_state_to_device(*_args, **_kwargs):
            return FailedStateEvent(), [object()]

        @staticmethod
        def close(*, unlink):
            closed.append(unlink)

    class Ledger:
        @staticmethod
        def get(_snapshot_id):
            return {"state": state["value"]}

        @staticmethod
        def request_host_load_failure(_snapshot_id, _owner, *, reason):
            failures.append(reason)
            state["value"] = HostStageState.ABORTING.value
            return True

        @staticmethod
        def mark_host_load_rank_drained(
            _snapshot_id, _owner, *, tp_rank, tp_size
        ):
            drained.append((tp_rank, tp_size))
            return True

    monkeypatch.setattr(
        p2d_host_module,
        "_OpenedP2DHybridHostSnapshot",
        lambda **_kwargs: Snapshot(),
    )
    monkeypatch.setattr(torch.cuda, "Event", lambda **_kwargs: SuccessfulEvent())

    receiver = SimpleNamespace(
        snapshot_id=snapshot_id,
        abort_pending=False,
        _owner="p-group:p0",
        _grant={
            "arena_path": "/dev/shm/fake-hybrid-p2d",
            "token_count": 4,
            "byte_size": 4096,
            "arena_offset": 0,
            "cache_components": ["attention", "mamba"],
        },
        mark_quarantined=lambda error: pytest.fail(f"unexpected quarantine: {error}"),
        mark_terminal=lambda status, error=None: terminal.append((status, error)),
    )
    manager = AgenticPToDHostLoadManager.__new__(AgenticPToDHostLoadManager)
    manager._work = queue.SimpleQueue()
    manager._work.put((receiver, torch.arange(4), (torch.tensor([1, 2]),)))
    manager._work.put(None)
    manager._stop = threading.Event()
    manager.ledger = Ledger()
    manager.device_pool = object()
    manager.chunk_tokens = 64
    manager.tp_rank = 0
    manager.tp_size = 2
    manager._dma_quarantine = []
    manager._dma_poisoned = False
    manager._completion_lock = threading.RLock()
    manager._group_pending = {}
    manager._group_wakeup = threading.Event()

    manager._worker(0, object(), object(), object())

    assert failures == ["p2d_h2d_failed:injected P2D Mamba H2D failure"]
    assert drained == [(0, 2)]
    assert terminal == []
    assert snapshot_id in manager._group_pending
    assert closed == [False]

    # Peer rank reaches its physical terminal later.  Only the resulting
    # group terminal may expose failure and authorize destination cleanup.
    state["value"] = HostStageState.FAILED.value
    assert manager._progress_group_completions_once() == 1
    assert len(terminal) == 1
    assert terminal[0][0] == KVPoll.Failed
    assert snapshot_id not in manager._group_pending


class _HostKVPool:
    layer_num = 2
    head_num = 2
    head_dim = 4
    v_head_dim = 4
    store_dtype = torch.float16

    def __init__(self):
        self.k_buffer = [torch.empty(8, 2, 4) for _ in range(self.layer_num)]
        self.v_buffer = [torch.empty(8, 2, 4) for _ in range(self.layer_num)]


class _HostMambaPool:
    def __init__(self):
        self.mamba_cache = SimpleNamespace(
            conv=[torch.zeros(2, 3, 2, dtype=torch.float32)],
            temporal=torch.zeros(2, 3, 2, 2, dtype=torch.float32),
        )
        self.loaded = None

    def get_cpu_copy(self, indices):
        return (
            [tensor[:, indices].clone() for tensor in self.mamba_cache.conv],
            self.mamba_cache.temporal[:, indices].clone(),
        )

    def load_cpu_copy(self, state, indices):
        conv, temporal = state
        for destination, source in zip(self.mamba_cache.conv, conv):
            destination[:, indices] = source
        self.mamba_cache.temporal[:, indices] = temporal
        self.loaded = indices.clone()


def test_shared_hybrid_snapshot_round_trips_complete_mamba_slot():
    kv_pool = _HostKVPool()
    source_pool = _HostMambaPool()
    source_pool.mamba_cache.conv[0][:, 1].fill_(3)
    source_pool.mamba_cache.temporal[:, 1].fill_(7)
    layout = HybridSnapshotLayout.from_pools(5, kv_pool, source_pool)
    assert layout.state_offset % 4096 == 0
    assert layout.total_bytes > layout.attention_bytes

    directory = tempfile.mkdtemp(prefix="sglang-agentic-mamba-", dir="/dev/shm")
    path = os.path.join(directory, "snapshot.bin")
    snapshot = SharedHybridHostSnapshot(
        path=path,
        token_count=5,
        kv_pool=kv_pool,
        mamba_pool=source_pool,
        create=True,
        layout=layout,
    )
    try:
        snapshot.mamba.backup_from_device(1)
        destination_pool = _HostMambaPool()
        snapshot.mamba.mamba_pool = destination_pool
        snapshot.mamba.load_to_device(2)
        assert torch.all(destination_pool.mamba_cache.conv[0][:, 2] == 3)
        assert torch.all(destination_pool.mamba_cache.temporal[:, 2] == 7)
        assert destination_pool.loaded.tolist() == [2]
    finally:
        snapshot.close(unlink=True)
        os.rmdir(directory)


def test_foreign_hybrid_recovery_adopts_extent_lazily():
    kv_pool = _HostKVPool()
    mamba_pool = _HostMambaPool()
    device_pool = SimpleNamespace(full_kv_pool=kv_pool, mamba_pool=mamba_pool)
    layout = HybridSnapshotLayout.from_pools(5, kv_pool, mamba_pool)
    directory = tempfile.mkdtemp(prefix="sglang-agentic-foreign-", dir="/dev/shm")
    path = os.path.join(directory, "snapshot.bin")
    with open(path, "wb") as handle:
        handle.truncate(layout.total_bytes)
    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager._state_lock = threading.RLock()
    manager.arena_domain = 1
    manager.owner = "p:recovery"
    manager.tp_rank = 0
    manager.tp_size = 1
    manager.device_pool = device_pool
    manager.host_ready = {}
    entry = {
        "snapshot_id": "foreign-hybrid:1",
        "state": HostStageState.HOST_READY.value,
        "p_owner": "p:storage",
        "recovery_prefill_domain": 1,
        "token_count": 5,
        "rank_grants": {
            "0": {
                "arena_path": path,
                "arena_offset": 0,
                "byte_size": layout.total_bytes,
                "token_count": 5,
            }
        },
    }
    try:
        record = manager._adopt_foreign_host_record("foreign-hybrid:1", entry)
        assert record is not None
        assert record["foreign_host"] is True
        assert record["snapshot"]._materialized is None
        assert manager._release_record(record)
        assert os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)
        os.rmdir(directory)


def test_storage_owner_releases_consumed_foreign_recovery_extent():
    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager._state_lock = threading.RLock()
    manager.owner = "p:storage"
    manager.host_ready = {"foreign:1": {"loading": False, "snapshot": object()}}
    manager.active = {}
    manager._host_eviction_local_released = set()
    released = []
    manager._release_record = lambda record: released.append(record) or True

    manager._progress_host_evictions(
        {
            "foreign:1": {
                "p_owner": "p:storage",
                "state": HostStageState.CONSUMED.value,
            }
        }
    )

    assert len(released) == 1
    assert "foreign:1" not in manager.host_ready


def test_consumed_d2p_ledger_prune_waits_for_all_tp_workset_handoffs():
    """The terminal TTL starts only after every restored TP lease is handed."""

    with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
        ledger = SharedHostStagingLedger(os.path.join(directory, "ledger.json"))
        snapshot_id = "d2p-consumed-before-handoff:1"

        def seed(entries):
            entries[snapshot_id] = {
                "snapshot_id": snapshot_id,
                "state": HostStageState.CONSUMED.value,
                "p_owner": "p:storage",
                "recovery_owner": "p:recovery",
                "recovery_claim_id": "slow:claim",
                "tp_size": 2,
                "recovery_claims": {
                    "0": {
                        "claim_id": "slow:claim",
                        "lease_id": 11,
                        "phase": "handed",
                    },
                    "1": {
                        "claim_id": "slow:claim",
                        "lease_id": 22,
                        "phase": "io_inflight",
                    },
                },
                "updated_at": time.time() - 60,
            }
            return True, True

        ledger._mutate(seed, event_snapshot_id=snapshot_id)
        ledger.prune(consumed_older_than_seconds=0)
        assert ledger.get(snapshot_id) is not None
        ledger.prune(older_than_seconds=0, consumed_older_than_seconds=0)
        assert ledger.get(snapshot_id) is not None

        assert ledger.mark_d2p_recovery_phase_rank(
            snapshot_id,
            "p:recovery",
            tp_rank=1,
            tp_size=2,
            claim_id="slow:claim",
            lease_id=22,
            phase="handed",
        )
        ledger.prune(consumed_older_than_seconds=0)
        assert ledger.get(snapshot_id) is None


def test_d2p_handoff_validates_claim_before_local_commit_and_retries_publish():
    """A lost ledger publish cannot strand a locally handed Slow workset."""

    with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
        ledger = SharedHostStagingLedger(os.path.join(directory, "ledger.json"))
        snapshot_id = "atomic-slow-handoff:1"
        calls = []

        def seed(entries):
            entries[snapshot_id] = {
                "snapshot_id": snapshot_id,
                "state": HostStageState.CONSUMED.value,
                "p_owner": "p:storage",
                "recovery_owner": "p:recovery",
                "tp_size": 1,
                "recovery_claims": {
                    "0": {
                        "claim_id": "slow:claim",
                        "lease_id": 23,
                        "phase": "io_inflight",
                    }
                },
                "updated_at": time.time(),
            }
            return True, True

        ledger._mutate(seed, event_snapshot_id=snapshot_id)

        assert not ledger.commit_d2p_handoff_rank(
            snapshot_id,
            "p:recovery",
            tp_rank=0,
            tp_size=1,
            claim_id="slow:claim",
            lease_id=999,
            handoff=lambda: calls.append("invalid"),
        )
        assert calls == []

        original_publish = ledger._publish_entry_event_locked
        lose_once = {"value": True}

        def fail_first_publish(selected_snapshot_id, entry):
            if lose_once["value"]:
                lose_once["value"] = False
                raise OSError("injected event publication failure")
            return original_publish(selected_snapshot_id, entry)

        ledger._publish_entry_event_locked = fail_first_publish
        with pytest.raises(OSError, match="publication failure"):
            ledger.commit_d2p_handoff_rank(
                snapshot_id,
                "p:recovery",
                tp_rank=0,
                tp_size=1,
                claim_id="slow:claim",
                lease_id=23,
                handoff=lambda: calls.append("handoff"),
            )
        assert calls == ["handoff"]
        assert (
            ledger.get(snapshot_id)["recovery_claims"]["0"]["phase"]
            == "io_inflight"
        )

        assert ledger.commit_d2p_handoff_rank(
            snapshot_id,
            "p:recovery",
            tp_rank=0,
            tp_size=1,
            claim_id="slow:claim",
            lease_id=23,
            handoff=lambda: calls.append("handoff"),
        )
        assert calls == ["handoff", "handoff"]
        assert ledger.get(snapshot_id)["recovery_claims"]["0"]["phase"] == "handed"


def test_foreign_tp_abort_receipts_are_group_atomic_and_idempotent(monkeypatch):
    """A lost ACK cannot strand a physically drained TP rank forever."""

    with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
        ledger = SharedHostStagingLedger(os.path.join(directory, "ledger.json"))
        snapshot_id = "foreign-abort:1"

        def seed(entries):
            entries[snapshot_id] = {
                "snapshot_id": snapshot_id,
                "state": HostStageState.H2D_LOADING.value,
                "p_owner": "p:storage",
                "recovery_owner": "p:recovery",
                "tp_size": 2,
                "recovery_claims": {
                    "0": {"claim_id": "claim", "lease_id": 11},
                    "1": {"claim_id": "claim", "lease_id": 22},
                },
            }
            return True, True

        ledger._mutate(seed, event_snapshot_id=snapshot_id)
        original_mutate = ledger._mutate
        lost_once = {"value": False}

        def commit_then_lose_reply(callback, *, event_snapshot_id=None):
            result = original_mutate(
                callback, event_snapshot_id=event_snapshot_id
            )
            if not lost_once["value"]:
                lost_once["value"] = True
                raise OSError("injected post-commit reply loss")
            return result

        monkeypatch.setattr(ledger, "_mutate", commit_then_lose_reply)
        with pytest.raises(OSError, match="post-commit"):
            ledger.complete_d2p_abort_rank(
                snapshot_id,
                "p:recovery",
                tp_rank=0,
                tp_size=2,
                claim_id="claim",
                lease_id=11,
                reason="request_aborted",
            )
        # Exact retry acknowledges the persisted receipt; a stale lease does not.
        assert ledger.complete_d2p_abort_rank(
            snapshot_id,
            "p:recovery",
            tp_rank=0,
            tp_size=2,
            claim_id="claim",
            lease_id=11,
            reason="request_aborted",
        )
        assert not ledger.complete_d2p_abort_rank(
            snapshot_id,
            "p:recovery",
            tp_rank=0,
            tp_size=2,
            claim_id="claim",
            lease_id=99,
            reason="request_aborted",
        )
        mid = ledger.get(snapshot_id)
        assert mid["state"] == HostStageState.ABORTING.value
        assert set(mid["recovery_claims"]) == {"1"}

        assert ledger.complete_d2p_abort_rank(
            snapshot_id,
            "p:recovery",
            tp_rank=1,
            tp_size=2,
            claim_id="claim",
            lease_id=22,
            reason="request_aborted",
        )
        final = ledger.get(snapshot_id)
        assert final["state"] == HostStageState.FAILED.value
        assert final["recovery_claims"] == {}
        assert set(final["h2d_abort_drained_receipts"]) == {"0", "1"}


def test_pre_io_recovery_lease_can_rebind_after_tp_epoch_reallocation():
    """A recreated active workset keeps the same logical Host claim."""

    with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
        ledger = SharedHostStagingLedger(os.path.join(directory, "ledger.json"))
        snapshot_id = "tp-reallocated-before-h2d:1"

        def seed(entries):
            entries[snapshot_id] = {
                "snapshot_id": snapshot_id,
                "state": HostStageState.H2D_LOADING.value,
                "p_owner": "p:recovery",
                "recovery_owner": "p:recovery",
                "recovery_claim_id": "slow:claim",
                "tp_size": 2,
                "recovery_claims": {
                    "0": {
                        "claim_id": "slow:claim",
                        "lease_id": 2594,
                        "phase": "leased",
                    },
                    "1": {
                        "claim_id": "slow:claim",
                        "lease_id": 2597,
                        "phase": "leased",
                    },
                },
            }
            return True, True

        ledger._mutate(seed, event_snapshot_id=snapshot_id)
        assert ledger.replace_d2p_recovery_lease_rank(
            snapshot_id,
            "p:recovery",
            tp_rank=0,
            tp_size=2,
            claim_id="slow:claim",
            old_lease_id=2594,
            new_lease_id=2598,
        )
        rebound = ledger.get(snapshot_id)
        assert rebound["recovery_claims"]["0"] == {
            "claim_id": "slow:claim",
            "lease_id": 2598,
            "phase": "leased",
        }
        assert rebound["recovery_claims"]["1"]["lease_id"] == 2597

        # The update is an exact CAS and cannot overwrite another replacement.
        assert not ledger.replace_d2p_recovery_lease_rank(
            snapshot_id,
            "p:recovery",
            tp_rank=0,
            tp_size=2,
            claim_id="slow:claim",
            old_lease_id=2594,
            new_lease_id=2600,
        )
        assert ledger.mark_d2p_recovery_phase_rank(
            snapshot_id,
            "p:recovery",
            tp_rank=0,
            tp_size=2,
            claim_id="slow:claim",
            lease_id=2598,
            phase="io_inflight",
        )
        # Once I/O owns the destination, lease replacement is forbidden.
        assert not ledger.replace_d2p_recovery_lease_rank(
            snapshot_id,
            "p:recovery",
            tp_rank=0,
            tp_size=2,
            claim_id="slow:claim",
            old_lease_id=2598,
            new_lease_id=2601,
        )


def test_foreign_unclaimed_digest_failure_is_assignment_scoped():
    """The recovery P may fail its exact assignment, not a foreign extent."""

    with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
        ledger = SharedHostStagingLedger(os.path.join(directory, "ledger.json"))
        snapshot_id = "foreign-digest-mismatch:1"

        def seed(entries):
            entries[snapshot_id] = {
                "snapshot_id": snapshot_id,
                "state": HostStageState.HOST_READY.value,
                "p_owner": "p:storage",
                "recovery_prefill_domain": 1,
                "recovery_reservation_id": "reservation-1",
                "tp_size": 2,
                "recovery_claims": {},
            }
            return True, True

        ledger._mutate(seed, event_snapshot_id=snapshot_id)
        assert not ledger.fail_d2p_recovery_assignment(
            snapshot_id,
            prefill_domain=1,
            reservation_id="stale-reservation",
            reason="permanent_parent_digest_mismatch",
        )
        assert ledger.fail_d2p_recovery_assignment(
            snapshot_id,
            prefill_domain=1,
            reservation_id="reservation-1",
            reason="permanent_parent_digest_mismatch",
        )
        # Exact retry is idempotent even though FAILED no longer represents a
        # live recovery claim.
        assert ledger.fail_d2p_recovery_assignment(
            snapshot_id,
            prefill_domain=1,
            reservation_id="reservation-1",
            reason="permanent_parent_digest_mismatch",
        )
        assert ledger.get(snapshot_id)["state"] == HostStageState.FAILED.value


def test_consumed_host_release_failure_retains_exact_record_for_retry():
    request = RequestGeneration("release-retry", 1)
    record = {"loading": "h2d", "snapshot": object()}
    load = {"request_generation": request, "record": record}
    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager._state_lock = threading.RLock()
    manager.tp_rank = 0
    manager.host_ready = {request.snapshot_id: record}
    manager.ledger = SimpleNamespace(
        get=lambda _snapshot_id: {"state": HostStageState.CONSUMED.value}
    )
    releases = iter((False, True))
    manager._release_record = lambda _record: next(releases)

    assert not manager._release_completed_h2d_host(load)
    assert manager.host_ready[request.snapshot_id] is record
    assert not load.get("host_released", False)
    assert manager._release_completed_h2d_host(load)
    assert request.snapshot_id not in manager.host_ready
    assert load["host_released"] is True


def test_aborted_host_release_failure_retains_load_and_record_for_retry():
    request = RequestGeneration("abort-release-retry", 1)
    record = {"loading": "h2d", "snapshot": object()}
    lease = SimpleNamespace(lease_id=7)
    load = {
        "request_generation": request,
        "record": record,
        "workset_lease": lease,
        "io_attempt": "attempt",
        "io_inflight": False,
        "drop_host_on_abort": True,
        "recovery_claim_id": "claim",
    }
    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager._state_lock = threading.RLock()
    manager._h2d_poisoned = False
    manager.tp_rank = 0
    manager.tp_size = 1
    manager.owner = "p:recovery"
    manager.loads = {"child": load}
    manager.host_ready = {request.snapshot_id: record}
    manager._h2d_lane_reservations = {request.snapshot_id: 0}
    manager.workset_broker = SimpleNamespace(
        cancel_io_attempt=lambda *_args: True,
        request_release=lambda *_args: None,
    )
    manager.ledger = SimpleNamespace(complete_d2p_abort_rank=lambda *_a, **_k: True)
    releases = iter((False, True))
    manager._release_record = lambda _record: next(releases)

    assert not manager._discard_failed_h2d_load("child", load)
    assert manager.loads["child"] is load
    assert manager.host_ready[request.snapshot_id] is record
    assert not load.get("host_released", False)
    assert manager._discard_failed_h2d_load("child", load)
    assert "child" not in manager.loads
    assert request.snapshot_id not in manager.host_ready
    assert request.snapshot_id not in manager._h2d_lane_reservations


def test_tp2_hybrid_snapshot_checksum_includes_attention_conv_and_temporal():
    """Both TP shards preserve every component of one successful snapshot."""

    for rank in range(2):
        source_kv = _HostKVPool()
        destination_kv = _HostKVPool()
        source_state = _HostMambaPool()
        destination_state = _HostMambaPool()
        for layer in range(source_kv.layer_num):
            source_kv.k_buffer[layer].fill_(10 * (rank + 1) + layer)
            source_kv.v_buffer[layer].fill_(20 * (rank + 1) + layer)
            destination_kv.k_buffer[layer].zero_()
            destination_kv.v_buffer[layer].zero_()
        source_state.mamba_cache.conv[0][:, 1].fill_(30 * (rank + 1))
        source_state.mamba_cache.temporal[:, 1].fill_(40 * (rank + 1))

        directory = tempfile.mkdtemp(
            prefix=f"sglang-agentic-mamba-tp{rank}-", dir="/dev/shm"
        )
        path = os.path.join(directory, "snapshot.bin")
        snapshot = SharedHybridHostSnapshot(
            path=path,
            token_count=5,
            kv_pool=source_kv,
            mamba_pool=source_state,
            create=True,
        )
        try:
            # CPU copies emulate the completed per-component DMA fences; the
            # lifecycle tests above cover the group commit ordering itself.
            for layer in range(source_kv.layer_num):
                snapshot.attention.k_buffer[layer].copy_(
                    source_kv.k_buffer[layer][:5]
                )
                snapshot.attention.v_buffer[layer].copy_(
                    source_kv.v_buffer[layer][:5]
                )
            snapshot.mamba.backup_from_device(1)
            for layer in range(destination_kv.layer_num):
                destination_kv.k_buffer[layer][:5].copy_(
                    snapshot.attention.k_buffer[layer]
                )
                destination_kv.v_buffer[layer][:5].copy_(
                    snapshot.attention.v_buffer[layer]
                )
            snapshot.mamba.mamba_pool = destination_state
            snapshot.mamba.load_to_device(0)

            for layer in range(source_kv.layer_num):
                assert torch.equal(
                    destination_kv.k_buffer[layer][:5],
                    source_kv.k_buffer[layer][:5],
                )
                assert torch.equal(
                    destination_kv.v_buffer[layer][:5],
                    source_kv.v_buffer[layer][:5],
                )
            assert torch.equal(
                destination_state.mamba_cache.conv[0][:, 0],
                source_state.mamba_cache.conv[0][:, 1],
            )
            assert torch.equal(
                destination_state.mamba_cache.temporal[:, 0],
                source_state.mamba_cache.temporal[:, 1],
            )
        finally:
            snapshot.close(unlink=True)
            os.rmdir(directory)


def test_p2d_host_snapshot_round_trips_active_and_checkpoint_slots():
    kv_pool = _HostKVPool()
    source_pool = _HostMambaPool()
    source_pool.mamba_cache.conv[0][:, 0].fill_(3)
    source_pool.mamba_cache.temporal[:, 0].fill_(7)
    source_pool.mamba_cache.conv[0][:, 1].fill_(11)
    source_pool.mamba_cache.temporal[:, 1].fill_(13)
    layout = HybridSnapshotLayout.from_pools(5, kv_pool, source_pool, state_slots=2)
    single = HybridSnapshotLayout.from_pools(5, kv_pool, source_pool)
    assert layout.state_bytes == 2 * single.state_bytes

    fd, path = tempfile.mkstemp(prefix="sglang-p2d-mamba-", dir="/dev/shm")
    os.ftruncate(fd, layout.state_bytes)
    os.close(fd)
    snapshot = SharedMambaHostSnapshot(
        path=path,
        mamba_pool=source_pool,
        byte_size=layout.state_bytes,
        file_offset=0,
        state_slots=2,
    )
    try:
        snapshot.backup_from_device(torch.tensor([0, 1]))
        destination_pool = _HostMambaPool()
        snapshot.mamba_pool = destination_pool
        snapshot.load_to_device(torch.tensor([1, 2]))
        assert torch.all(destination_pool.mamba_cache.conv[0][:, 1] == 3)
        assert torch.all(destination_pool.mamba_cache.temporal[:, 1] == 7)
        assert torch.all(destination_pool.mamba_cache.conv[0][:, 2] == 11)
        assert torch.all(destination_pool.mamba_cache.temporal[:, 2] == 13)
        assert destination_pool.loaded.tolist() == [1, 2]
    finally:
        snapshot.close()
        os.unlink(path)


def test_manifest_commits_attention_and_mamba_as_one_generation():
    manifest = SnapshotManifest(
        request=RequestGeneration("hybrid", 2),
        page_keys=("kv-page-0", "mamba-state"),
        token_count=128,
        byte_size=8192,
        state=SnapshotState.MOONCAKE_READY,
        cache_components=("attention", "mamba"),
        state_byte_size=2048,
        state_checkpoint_tokens=128,
    )
    restored = SnapshotManifest.from_bytes(manifest.to_bytes())

    assert restored.cache_components == ("attention", "mamba")
    assert restored.state_byte_size == 2048
    assert restored.state_checkpoint_tokens == 128


def test_reverse_wire_submission_includes_mamba_source_and_destination():
    mamba_type = [StateType.MAMBA]
    req = SimpleNamespace(mamba_pool_idx=torch.tensor(7))
    lease = SimpleNamespace(
        parent_page_indices=torch.tensor([10, 11]).numpy(),
        state_device_indices=(torch.tensor([9]),),
    )
    sent = []
    metadata = []
    sender = SimpleNamespace(
        send=lambda pages, state_indices=None: sent.append((pages, state_indices))
    )
    receiver = SimpleNamespace(
        send_metadata=lambda pages, aux_index=None, state_indices=None: metadata.append(
            (pages, aux_index, state_indices)
        )
    )

    submit_reverse_send(sender, torch.tensor([1, 2]).numpy(), req, mamba_type)
    submit_reverse_receive(receiver, lease, mamba_type)

    assert int(state_indices_for_req(req, mamba_type)[0][0]) == 7
    assert int(state_indices_for_workset(lease, mamba_type)[0][0]) == 9
    assert int(sent[0][1][0][0]) == 7
    assert metadata[0][1] == 0
    assert int(metadata[0][2][0][0]) == 9


def test_reverse_snapshot_uses_nonlazy_ping_pong_checkpoint():
    req = SimpleNamespace(
        mamba_pool_idx=torch.tensor(99),
        mamba_ping_pong_track_buffer=torch.tensor([7, 8]),
        mamba_next_track_idx=1,
        mamba_last_track_seqlen=128,
    )
    assert int(mamba_checkpoint_index_for_req(req)) == 7
    assert snapshot_token_count_for_req(req, 159, [StateType.MAMBA], 64) == 128
    assert int(state_indices_for_req(req, [StateType.MAMBA])[0][0]) == 7


def test_reverse_snapshot_stop_token_boundary_uses_previous_checkpoint():
    req = SimpleNamespace(
        mamba_pool_idx=torch.tensor(99),
        # next=1 means slot 0 contains the latest 1024-token checkpoint and
        # slot 1 still contains the preceding 960-token checkpoint.
        mamba_ping_pong_track_buffer=torch.tensor([7, 8]),
        mamba_next_track_idx=1,
        mamba_last_track_seqlen=1024,
    )

    checkpoint_tokens = snapshot_token_count_for_req(
        req, 1023, [StateType.MAMBA], 64
    )

    assert checkpoint_tokens == 960
    assert (
        int(
            state_indices_for_req(
                req,
                [StateType.MAMBA],
                checkpoint_tokens=checkpoint_tokens,
                page_size=64,
            )[0][0]
        )
        == 8
    )


def test_reverse_snapshot_stop_boundary_rejects_lazy_single_checkpoint():
    req = SimpleNamespace(
        mamba_pool_idx=torch.tensor(99),
        mamba_ping_pong_track_buffer=torch.tensor([-1, 7]),
        mamba_next_track_idx=1,
        mamba_last_track_seqlen=1024,
    )

    with pytest.raises(RuntimeError, match="no retained physical slot"):
        snapshot_token_count_for_req(req, 1023, [StateType.MAMBA], 64)


def test_reverse_snapshot_uses_lazy_ping_pong_checkpoint():
    req = SimpleNamespace(
        mamba_pool_idx=torch.tensor(99),
        mamba_ping_pong_track_buffer=torch.tensor([-1, 8]),
        mamba_next_track_idx=1,
        mamba_last_track_seqlen=128,
    )
    assert int(mamba_checkpoint_index_for_req(req)) == 8
    assert snapshot_token_count_for_req(req, 128, [StateType.MAMBA], 64) == 128


def test_reverse_snapshot_rejects_unaligned_active_state_without_checkpoint():
    req = SimpleNamespace(
        mamba_pool_idx=torch.tensor(4),
        mamba_ping_pong_track_buffer=torch.tensor([7, 8]),
        mamba_next_track_idx=0,
        mamba_last_track_seqlen=None,
    )
    with pytest.raises(RuntimeError, match="no page-boundary"):
        snapshot_token_count_for_req(req, 95, [StateType.MAMBA], 64)
    assert snapshot_token_count_for_req(req, 128, [StateType.MAMBA], 64) == 128
    assert int(mamba_checkpoint_index_for_req(req)) == 4


def test_reverse_snapshot_rejects_invalid_lazy_checkpoint():
    req = SimpleNamespace(
        mamba_pool_idx=torch.tensor(4),
        mamba_lazy_is_insert=False,
        mamba_last_track_seqlen=64,
    )
    with pytest.raises(RuntimeError, match="lazy Mamba checkpoint is invalid"):
        snapshot_token_count_for_req(req, 64, [StateType.MAMBA], 64)


def test_p2d_mamba_transfers_active_and_latest_page_checkpoint():
    source = SimpleNamespace(
        origin_input_ids=list(range(95)),
        mamba_pool_idx=torch.tensor(5),
        mamba_ping_pong_track_buffer=torch.tensor([7, 8]),
        mamba_next_track_idx=0,
        mamba_last_track_seqlen=64,
    )
    assert p2d_mamba_source_indices(source, 64)[0].tolist() == [5, 8]

    destination = SimpleNamespace(
        origin_input_ids=list(range(95)),
        mamba_pool_idx=torch.tensor(15),
        mamba_ping_pong_track_buffer=torch.tensor([17, 18]),
        mamba_next_track_idx=0,
    )
    req_pool = SimpleNamespace(get_mamba_ping_pong_keep_idx=lambda _req: 1)
    assert p2d_mamba_destination_indices(destination, req_pool, 64)[0].tolist() == [
        15,
        18,
    ]
    assert destination.mamba_last_track_seqlen == 64


def test_p2d_subpage_prompt_duplicates_active_without_claiming_checkpoint():
    source = SimpleNamespace(
        origin_input_ids=list(range(31)),
        mamba_pool_idx=torch.tensor(5),
    )
    assert p2d_mamba_source_indices(source, 64)[0].tolist() == [5, 5]

    destination = SimpleNamespace(
        origin_input_ids=list(range(31)),
        mamba_pool_idx=torch.tensor(15),
        mamba_ping_pong_track_buffer=torch.tensor([17, 18]),
        mamba_next_track_idx=0,
    )
    req_pool = SimpleNamespace(get_mamba_ping_pong_keep_idx=lambda _req: 1)
    assert p2d_mamba_destination_indices(destination, req_pool, 64)[0].tolist() == [
        15,
        18,
    ]
    assert destination.mamba_last_track_seqlen is None


def test_page64_snapshot_bytes_use_page_descriptors_not_token_descriptors():
    kv_args = SimpleNamespace(
        page_size=64,
        kv_item_lens=[4096, 4096],
        state_item_lens=[[512, 128]],
    )
    # 128 tokens = 2 pages, then mmap alignment before the 640-byte state.
    assert complete_snapshot_bytes(kv_args, 128) == 16_384 + 640
    with pytest.raises(ValueError, match="page aligned"):
        complete_snapshot_bytes(kv_args, 127)


def test_agentic_mamba_config_requires_page_tracking_and_bf16_checkpoint():
    kv_args = SimpleNamespace(page_size=64, state_types=[StateType.MAMBA])
    validate_agentic_mamba_tracking(
        kv_args,
        SimpleNamespace(
            mamba_track_interval=64,
            enable_int8_mamba_checkpoint=False,
        ),
    )
    with pytest.raises(ValueError, match="mamba-track-interval=64"):
        validate_agentic_mamba_tracking(
            kv_args,
            SimpleNamespace(
                mamba_track_interval=256,
                enable_int8_mamba_checkpoint=False,
            ),
        )
    with pytest.raises(ValueError, match="int8"):
        validate_agentic_mamba_tracking(
            kv_args,
            SimpleNamespace(
                mamba_track_interval=64,
                enable_int8_mamba_checkpoint=True,
            ),
        )


@pytest.mark.parametrize(
    ("agentic_lifecycle", "expected_track"),
    [(False, 384), (True, 1344)],
)
def test_agentic_prefill_prefers_request_tail_checkpoint_over_radix_branch(
    monkeypatch, agentic_lifecycle, expected_track
):
    """The one Prefill tracking slot must make P->D state page-consistent."""

    monkeypatch.setenv(
        "SGLANG_AGENTIC_KV_LIFECYCLE", str(agentic_lifecycle).lower()
    )
    monkeypatch.setattr(
        schedule_batch_module,
        "get_global_server_args",
        lambda: SimpleNamespace(
            mamba_cache_chunk_size=64,
            enable_mamba_extra_buffer_lazy=lambda: False,
        ),
    )
    req = SimpleNamespace(
        extend_input_len=1368,
        prefix_indices=[],
        mamba_ping_pong_track_buffer=torch.tensor([7, 8]),
        mamba_next_track_idx=0,
        mamba_branching_seqlen=384,
    )
    batch = ScheduleBatch(
        reqs=[req],
        req_to_token_pool=SimpleNamespace(
            get_mamba_ping_pong_other_idx=lambda index: 1 - index
        )
    )

    entry = batch._mamba_radix_cache_v2_req_prepare_for_extend(req)

    assert req.mamba_last_track_seqlen == expected_track
    assert entry.track_seqlen == (
        expected_track + 1 if expected_track == 384 else 1368
    )


@pytest.mark.parametrize(
    ("lifecycle", "disable_overlap", "expected"),
    [(False, False, 1), (True, False, 3), (True, True, 2)],
)
def test_decode_mamba_memory_ratio_accounts_for_agentic_tracking_slots(
    monkeypatch, lifecycle, disable_overlap, expected
):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_LIFECYCLE", str(lifecycle).lower())
    server_args = SimpleNamespace(
        disable_radix_cache=True,
        disable_overlap_schedule=disable_overlap,
        enable_mamba_extra_buffer=lambda: lifecycle,
        enable_mamba_extra_buffer_lazy=lambda: False,
    )
    runner = SimpleNamespace(server_args=server_args)
    assert ModelRunnerKVCacheMixin._calculate_mamba_ratio(runner) == expected
