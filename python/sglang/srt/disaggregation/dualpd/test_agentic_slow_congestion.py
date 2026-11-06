import json
from unittest.mock import mock_open, patch

import pytest

from sglang.srt.disaggregation.agentic_slow_congestion import (
    SlowCongestionReader, SlowRecoveryCongestion, waiting_for_recovery,
)


def test_hysteresis_and_sampling_interval():
    policy = SlowRecoveryCongestion(32, 8)
    assert not policy.sample(32, 0)["congested"]
    assert not policy.sample(32, 0.2)["congested"]
    assert policy.sample(32, 1)["congested"]
    assert policy.sample(9, 2)["congested"]
    assert not policy.sample(8, 3)["congested"]
    assert not policy.sample(32, 4)["congested"]
    assert not policy.sample(20, 5)["congested"]
    assert not policy.sample(32, 6)["congested"]


def test_parent_reference_counts_survive_redirect_duplicate_and_cancel():
    p = SlowRecoveryCongestion(32, 8)
    p.enter("same:1"); p.enter("same:1")
    assert set(p.parents) == {"same:1"}
    p.leave("same:1")
    assert set(p.parents) == {"same:1"}
    p.leave("same:1")
    assert not p.parents


@pytest.mark.parametrize("state,phase,expected", [
    ("host_writing", None, False), ("host_reserved", None, False),
    ("host_ready", None, True), ("h2d_loading", "pinned", True),
    ("h2d_loading", "leased", True), ("h2d_loading", "io_inflight", False),
    ("hbm_ready", "handed", False), ("consumed", None, False),
    ("evicting", None, False), ("recompute_required", None, False),
    ("aborting", None, False), ("retry_pending", None, False),
])
def test_ledger_phase_not_state_name_determines_wait(state, phase, expected):
    entry = dict(state=state, recovery_claims={} if phase is None else {"0": {"phase": phase}})
    assert waiting_for_recovery(entry) is expected


def test_tp_one_shard_started_excludes_generation_retry_can_reenter():
    entry = dict(state="h2d_loading", recovery_claims={
        "0": {"phase": "leased"}, "1": {"phase": "io_inflight"}})
    assert not waiting_for_recovery(entry)
    entry.update(state="host_ready", recovery_claims={})
    assert waiting_for_recovery(entry)


@pytest.mark.parametrize("payload", [[], None, 1, {}, {"slow_recovery": []},
    {"slow_recovery": {"version": 1, "congested": True}},
    {"slow_recovery": {"version": 1, "sampled_at": 80, "congested": True}},
    {"slow_recovery": {"version": 1, "sampled_at": 110, "congested": True}},
    {"slow_recovery": {"version": 1, "sampled_at": 100, "congested": "true"}},
])
def test_missing_stale_malformed_pressure_is_normal_slow(payload):
    with patch("builtins.open", mock_open(read_data=json.dumps(payload))), patch("time.time", return_value=100):
        assert not SlowCongestionReader("/irrelevant").congested()


def test_reader_is_cached_and_stale_cache_cannot_recompute():
    reader = SlowCongestionReader("/irrelevant")
    body = json.dumps({"slow_recovery": {"version": 1, "sampled_at": 100, "congested": True}})
    with patch("builtins.open", mock_open(read_data=body)) as op, patch("time.time", return_value=100), patch("time.monotonic", return_value=20):
        assert reader.congested()
        assert reader.congested()
        assert op.call_count == 1
    with patch("time.time", return_value=104), patch("time.monotonic", return_value=20.5):
        assert not reader.congested()


def test_io_error_is_normal_slow():
    with patch("builtins.open", side_effect=OSError("injected")):
        assert not SlowCongestionReader("/irrelevant").congested()
