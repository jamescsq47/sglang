"""Controller-owned Prefill cancellation uses the native Forward result fence."""
from collections import deque
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.disaggregation import prefill
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.schedule_batch import FINISH_ABORT


def fixture():
    calls = []
    req = NS(rid="cancel", disagg_p_ready_deferred=True,
             disagg_p_ready_transfer_started=False, kv_committed_len=128,
             to_finish=FINISH_ABORT(), _agentic_workset_backed=True)
    s = NS(agentic_p_workset_broker=NS(controller_mode=True),
           disagg_prefill_inflight_queue=[], result_queue=deque(),
           chunked_req=req, tree_cache=NS(),
           req_to_metadata_buffer_idx_allocator=NS(),
           send_to_tokenizer=NS(send_output=lambda *a: calls.append("reply")))
    s._finish_controller_prefill_abort = lambda r: Scheduler._finish_controller_prefill_abort(s, r)
    s._agentic_abort_cleanup = lambda r: calls.append("lease")
    return s, req, calls


def setup_cleanup(monkeypatch, s, calls):
    monkeypatch.setattr(prefill, "release_kv_cache", lambda *a, **kw: calls.append("native"))
    monkeypatch.setattr(prefill, "release_req_to_metadata_buffer", lambda *a: calls.append("metadata"))
    s.tree_cache.release_agentic_request_cache = lambda *a, **kw: calls.append("branch")


def test_completed_chunk_abort_stops_continuation_and_retires_once(monkeypatch):
    s, req, calls = fixture()
    setup_cleanup(monkeypatch, s, calls)
    assert Scheduler._request_controller_prefill_abort(s, req)
    assert s.chunked_req is None
    assert calls == ["lease", "native", "branch", "metadata", "reply"]
    assert req.finished_reason is req.to_finish
    Scheduler._finish_controller_prefill_abort(s, req)
    assert len(calls) == 5


def test_overlap_abort_keeps_workset_until_real_result_boundary(monkeypatch):
    s, req, calls = fixture()
    setup_cleanup(monkeypatch, s, calls)
    batch = NS(reqs=[req], return_logprob=False, prefill_stats=None, dp_cooperation_info=None)
    result = NS(logits_output=None, next_token_ids=torch.tensor([1]),
                extend_input_len_per_req=None, extend_logprob_start_len_per_req=None,
                copy_done=NS(synchronize=lambda: calls.append("forward_done")))
    s.result_queue.append((batch, result))
    s.report_prefill_stats = lambda **kw: None
    assert Scheduler._request_controller_prefill_abort(s, req)
    assert calls == [] and req._agentic_workset_backed
    assert s.chunked_req is None
    Scheduler.process_batch_result_disagg_prefill(s, batch, result)
    assert calls == ["forward_done", "lease", "native", "branch", "metadata", "reply"]


def test_transfer_owned_abort_never_uses_compute_cleanup():
    s, req, calls = fixture()
    s.disagg_prefill_inflight_queue.append(req)
    assert Scheduler._request_controller_prefill_abort(s, req)
    assert not calls
    with pytest.raises(RuntimeError, match="in-flight P->D"):
        Scheduler._finish_controller_prefill_abort(s, req)


def test_legacy_abort_not_migrated():
    s, req, calls = fixture()
    s.agentic_p_workset_broker.controller_mode = False
    assert not Scheduler._request_controller_prefill_abort(s, req)
    assert not calls and s.chunked_req is req


@pytest.mark.parametrize("location", ["chunk", "result", "both"])
def test_actual_http_abort_finds_req_outside_mutable_batches(monkeypatch, location):
    from sglang.srt.disaggregation.utils import DisaggregationMode

    s, req, calls = fixture()
    setup_cleanup(monkeypatch, s, calls)
    req.finished = lambda: getattr(req, "finished_reason", None) is not None
    s.agentic_kv_waiting_queue = []
    s.waiting_queue = []
    s.grammar_manager = NS(abort_requests=lambda _: None)
    s.disaggregation_mode = DisaggregationMode.PREFILL
    s.disagg_prefill_bootstrap_queue = NS(queue=[])
    s.running_batch, s.cur_batch = NS(reqs=[]), None
    s._request_controller_prefill_abort = lambda r: Scheduler._request_controller_prefill_abort(s, r)
    if location == "result":
        s.chunked_req = None
    if location != "chunk":
        batch = NS(reqs=[req], return_logprob=False, prefill_stats=None, dp_cooperation_info=None)
        result = NS(logits_output=None, next_token_ids=torch.tensor([1]),
                    extend_input_len_per_req=None, extend_logprob_start_len_per_req=None,
                    copy_done=NS(synchronize=lambda: calls.append("forward_done")))
        s.result_queue.append((batch, result))
        s.report_prefill_stats = lambda **kw: None
    Scheduler.abort_request(s, NS(abort_all=False, rid=req.rid))
    assert s.chunked_req is None
    if location == "chunk":
        assert calls == ["lease", "native", "branch", "metadata", "reply"]
    else:
        assert calls == []
        Scheduler.process_batch_result_disagg_prefill(s, batch, result)
        assert calls == ["forward_done", "lease", "native", "branch", "metadata", "reply"]
    # The same Req in both staging locations is retired and replied once.
    Scheduler.abort_request(s, NS(abort_all=True, rid=""))
    assert calls.count("reply") == calls.count("native") == 1
