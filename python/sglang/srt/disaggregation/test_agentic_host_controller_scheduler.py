"""Native scheduler consumes prepared Host commands, never selects their IO."""
from types import SimpleNamespace as NS

import pytest

from sglang.srt.disaggregation.agentic_kv_lifecycle import AgenticRequestMetadata, RequestGeneration
from sglang.srt.disaggregation.agentic_workset_admission import DuplicateGenerationRequest
from sglang.srt.disaggregation.test_agentic_tp_host_pipeline import leader
from sglang.srt.managers.scheduler import Scheduler


@pytest.mark.parametrize("size", [2, 8])
def test_native_scheduler_never_selects_host_io_with_controller(size):
    scheduler = leader(tp_size=size)
    scheduler.agentic_host_restore_controller = NS(
        owns=lambda sid: False, native_commands=lambda: [],
    )
    scheduler.agentic_host_staging_manager.snapshot_ready = lambda parent: pytest.fail(
        "native scheduler must not select Host IO"
    )
    control = Scheduler._agentic_tp_prepare_admission_control(scheduler)
    assert control["host_commands"] == []
    assert not scheduler.agentic_tp_host_active_requests


def test_duplicate_generation_rejects_only_incoming_request(monkeypatch):
    import sglang.srt.managers.scheduler as mod
    rejected, streamed = [], []
    req = NS(rid="retry", extra_key="scope", return_logprob=False)
    metadata = NS(current=RequestGeneration("agent", 1))
    monkeypatch.setattr(AgenticRequestMetadata, "from_req", lambda req: metadata)
    monkeypatch.setattr(mod, "prepare_abort", lambda req, msg, status_code: rejected.append((req.rid, status_code)))
    def observe(*args):
        raise DuplicateGenerationRequest("agent:1", "original", "retry")
    scheduler = NS(agentic_fresh_workset_admission=NS(observe=observe),
                   stream_output=lambda reqs, logs: streamed.extend(reqs))
    Scheduler._add_request_to_queue(scheduler, req)
    assert rejected == [("retry", 409)]
    assert streamed == [req]
