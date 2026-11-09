"""D writer identity, progress dispatch and physical cleanup without CUDA DMA."""
import json
import os
import pickle
import subprocess
import sys
from concurrent.futures import Future
from types import SimpleNamespace as NS

import pytest

from sglang.srt.disaggregation import agentic_host_writer as writer


def test_historical_aliases_sentinel_default_and_logger_are_identical():
    from sglang.srt.disaggregation import agentic_host_staging as staging

    for name in ("_LEDGER_ENTRY_UNSET", "_cleanup_nixl_room",
                 "AgenticDRelayWorker", "AgenticDHostStagingClient"):
        assert getattr(staging, name) is getattr(writer, name)
    assert writer.AgenticDHostStagingClient.progress.__kwdefaults__["entry_snapshot"] is writer._LEDGER_ENTRY_UNSET
    assert writer.logger is staging.logger


@pytest.mark.parametrize("name", ["AgenticDHostStagingClient", "AgenticDRelayWorker"])
def test_legacy_pickle_class_path_resolves_same_class(name):
    cls = getattr(writer, name)
    instance = cls.__new__(cls)
    instance.fixture = "no resources"
    legacy = pickle.dumps(instance, protocol=0).replace(
        b"sglang.srt.disaggregation.agentic_host_writer\n",
        b"sglang.srt.disaggregation.agentic_host_staging\n")
    restored = pickle.loads(legacy)
    assert type(restored) is cls and restored.fixture == "no resources"


def test_cold_import_has_no_cycle_or_extra_native_cuda_initialization():
    env = dict(os.environ)
    code = """
import json,sys,torch
if sys.argv[1] == 'native-baseline':
 from sglang.srt.disaggregation.base import KVPoll
 from sglang.srt.disaggregation.utils import kv_to_page_indices
 print(json.dumps(torch.cuda.is_initialized()))
 raise SystemExit(0)
from sglang.srt.disaggregation import agentic_host_writer as writer
before = 'sglang.srt.disaggregation.agentic_host_staging' in sys.modules
initialized_before_alias = torch.cuda.is_initialized()
from sglang.srt.disaggregation import agentic_host_staging as staging
print(json.dumps([before, initialized_before_alias, torch.cuda.is_initialized(),
 staging.AgenticDHostStagingClient is writer.AgenticDHostStagingClient,
 staging._LEDGER_ENTRY_UNSET is writer._LEDGER_ENTRY_UNSET]))
"""
    # These exact native imports already initialize CUDA on H100 through
    # sgl_kernel's architecture query; CPU short-circuits. Compare two fresh
    # processes, not an absolute no-init promise that the old engine never met.
    baseline = subprocess.run([sys.executable, "-c", code, "native-baseline"], env=env,
                              capture_output=True, text=True, check=True, timeout=60)
    native_initialized = json.loads(baseline.stdout.splitlines()[-1])
    result = subprocess.run([sys.executable, "-c", code, "writer"], env=env,
                            capture_output=True, text=True, check=True, timeout=60)
    assert json.loads(result.stdout.splitlines()[-1]) == [False, native_initialized, native_initialized, True, True]


def client_fixture():
    client = writer.AgenticDHostStagingClient.__new__(writer.AgenticDHostStagingClient)
    calls = []
    client.ledger = NS(get=lambda sid: calls.append(("get", sid)) or {"state": writer.HostStageState.FAILED.value})
    client._cleanup_write = lambda _: True
    client._cleanup_relay_senders = lambda _: calls.append(("relay_cleanup",))
    return client, {"manifest": NS(snapshot_id="fixture:0")}, calls


def test_omitted_entry_reads_ledger_with_canonical_sentinel():
    client, candidate, calls = client_fixture()
    assert client.progress(candidate, []) == "failed"
    assert calls == [("get", "fixture:0"), ("relay_cleanup",)]


def test_explicit_none_does_not_read_ledger_or_recreate_missing_offer():
    client, candidate, calls = client_fixture()
    assert client.progress(candidate, [], entry_snapshot=None) == "waiting"
    assert calls == [] and candidate["missing_host_ledger_logged"]


def test_explicit_entry_bypasses_read_and_observes_durability():
    client, candidate, calls = client_fixture()
    assert client.progress(candidate, [], entry_snapshot={"state": writer.HostStageState.HOST_READY.value}) == "host_ready"
    assert calls == [("relay_cleanup",)]


def test_local_only_rejects_absent_active_write_without_ledger_read():
    client, candidate, calls = client_fixture()
    with pytest.raises(ValueError, match="active Host write"):
        client.progress(candidate, [], local_write_only=True)
    assert calls == []


def test_local_only_progress_retires_cpu_commit_without_reading_ledger():
    client, candidate, calls = client_fixture()
    completed = Future()
    completed.set_result(0.125)
    candidate["arena_write"] = {"chunks": {0: {
        "cpu_future": completed, "start": 0, "end": 4}},
        "host_copy_elapsed_seconds": 0, "committed_tokens": 0}
    client._d2h_lanes = [{"snapshot_id": "fixture:0", "phase": "cpu"}]
    assert client.progress(candidate, [0, 1, 2, 3], local_write_only=True) == "waiting"
    assert calls == []
    assert candidate["arena_write"]["committed_tokens"] == 4
    assert candidate["arena_write"]["host_copy_elapsed_seconds"] == 0.125
    assert candidate["arena_write"]["chunks"] == {}
    assert client._d2h_lanes[0] == {"snapshot_id": None, "phase": "free"}


def test_running_cpu_commit_keeps_mapping_and_lane_until_retry():
    client, candidate, _ = client_fixture()
    pending = Future()
    assert pending.set_running_or_notify_cancel()
    closes = []
    candidate["arena_write"] = {"snapshot": NS(close=lambda **kw: closes.append(kw)),
        "chunks": {0: {"cpu_future": pending}}}
    client._d2h_lanes = [{"snapshot_id": "fixture:0", "phase": "cpu"}]
    assert writer.AgenticDHostStagingClient._cleanup_write(client, candidate) is False
    assert closes == [] and client._d2h_lanes[0]["snapshot_id"] == "fixture:0"
    pending.set_result(0)
    assert writer.AgenticDHostStagingClient._cleanup_write(client, candidate) is True
    assert closes == [{"unlink": False}] and "arena_write" not in candidate
    assert client._d2h_lanes[0]["phase"] == "free"


def test_failed_unregister_keeps_mapping_and_lane_until_retry():
    client, candidate, _ = client_fixture()
    closes = []
    def close(**kwargs):
        closes.append(kwargs)
        if len(closes) == 1:
            raise RuntimeError("unregister failed")
    candidate["arena_write"] = {"snapshot": NS(close=close), "chunks": {0: {}}}
    client._d2h_lanes = [{"snapshot_id": "fixture:0", "phase": "cpu"}]
    assert writer.AgenticDHostStagingClient._cleanup_write(client, candidate) is False
    assert "arena_write" in candidate and client._d2h_lanes[0]["snapshot_id"] == "fixture:0"
    assert writer.AgenticDHostStagingClient._cleanup_write(client, candidate) is True
    assert "arena_write" not in candidate and client._d2h_lanes[0]["phase"] == "free"


def test_room_cleanup_does_not_touch_unrelated_room_or_bootstrap():
    tables = ("request_status", "failure_records", "required_prefill_response_num_table",
              "prefill_response_tracker", "transfer_infos", "transfer_statuses")
    manager = NS(**{name: {11: "owned", 12: "unrelated"} for name in tables},
                 addr_to_rooms_tracker={"one": {11, 12}, "other": {11}})
    writer._cleanup_nixl_room(manager, 11, "one")
    assert all(getattr(manager, name) == {12: "unrelated"} for name in tables)
    assert manager.addr_to_rooms_tracker == {"one": {12}, "other": {11}}
    writer._cleanup_nixl_room(NS(request_status={11: 1}), 11)


def test_relay_finish_fences_before_mapping_close_and_active_release():
    relay = writer.AgenticDRelayWorker.__new__(writer.AgenticDRelayWorker)
    calls = []
    relay.active = {"event": NS(synchronize=lambda: calls.append("fence")),
                    "snapshot": NS(close=lambda **kw: calls.append(("close", kw)))}
    relay._close_receiver = lambda _: calls.append("receiver")
    relay._finish_active()
    assert calls == ["receiver", "fence", ("close", {"unlink": False})]
    assert relay.active is None


def test_relay_successful_failure_cleanup_publishes_fallback_after_close():
    relay = writer.AgenticDRelayWorker.__new__(writer.AgenticDRelayWorker)
    calls = []
    relay.active = {"entry": {"snapshot_id": "fixture:0"},
                    "event": NS(synchronize=lambda: calls.append("fence")),
                    "snapshot": NS(close=lambda **kw: calls.append("close"))}
    relay._close_receiver = lambda _: calls.append("receiver")
    relay.relay_id = "relay"
    relay.ledger = NS(relay_fail_to_direct=lambda *args: calls.append(("fallback", args)))
    relay._fail_active("fixture")
    assert calls == ["receiver", "fence", "close", ("fallback", ("fixture:0", "relay", "fixture"))]
    assert relay.active is None
