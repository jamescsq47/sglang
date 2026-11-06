"""TP admission must preserve the leader's order across a bounded batch."""

from types import SimpleNamespace

import sglang.srt.managers.scheduler as scheduler_module
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    CUSTOM_GENERATION, CUSTOM_PARENT_GENERATION, CUSTOM_REQUEST_ID,
)


def test_host_commit_order_survives_rank_local_hash_order(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_ADMISSION_BATCH", "8")
    direct = [f"direct-{i}" for i in range(5)]
    host = ["host-c", "host-a", "host-d", "host-b"]
    commands = lambda names, action: [
        dict(snapshot=f"{name}:1", request_id=name, generation=1, action=action)
        for name in names
    ]
    control = {
        Scheduler._AGENTIC_TP_CONTROL_KEY: True,
        "workset_plan_epoch": 1, "workset_allocation_plan": [],
        "direct_commands": commands(direct, "commit_bind"),
        "host_commands": commands(host, "commit"),
    }
    admitted_by_rank = []
    for rank in range(2):
        # Deterministically emulate independent Python hash seeds. Do not rely
        # on a fortuitous set order in the process running this regression.
        class RankSet(set):
            def __iter__(self):
                return iter(sorted(set.copy(self), reverse=bool(rank)))

        monkeypatch.setattr(scheduler_module, "set", RankSet, raising=False)
        admitted = []
        owner = SimpleNamespace(
            tp_size=2, tp_rank=rank,
            disaggregation_mode=DisaggregationMode.PREFILL,
            _AGENTIC_TP_CONTROL_KEY=Scheduler._AGENTIC_TP_CONTROL_KEY,
            agentic_tp_direct_admission_active={},
            agentic_tp_direct_group_status={},
            agentic_tp_direct_local_admitted=set(),
            agentic_tp_direct_local_failed=set(),
            agentic_early_direct_receives={},
            agentic_tp_host_local_admitted=set(),
            agentic_host_staging_manager=SimpleNamespace(),
            agentic_p_workset_broker=SimpleNamespace(
                install_tp_plan=lambda *a, **kw: None,
                get=lambda *a, **kw: None,
            ),
            _agentic_io_active=lambda req: False,
            _agentic_io_kind=lambda req: None,
            _agentic_queue_class=lambda req: "slow" if req.rid in host else "fast",
            _agentic_should_defer=lambda *a, **kw: False,
            _agentic_publish_p_scheduled=lambda req: None,
            _add_request_to_queue=lambda req: admitted.append(req.rid),
        )
        names = direct + host
        if rank:
            names = list(reversed(names))
        owner.agentic_kv_waiting_queue = [
            (SimpleNamespace(
                rid=name,
                sampling_params=SimpleNamespace(custom_params={
                    CUSTOM_REQUEST_ID: name, CUSTOM_GENERATION: 2,
                    CUSTOM_PARENT_GENERATION: 1,
                }),
            ), float(i)) for i, name in enumerate(names)
        ]
        Scheduler._agentic_tp_consume_admission_control(owner, [control])
        Scheduler._drain_agentic_kv_waiting_queue(owner)
        admitted_by_rank.append(admitted)
        assert admitted == direct + host[:3]
        assert owner._agentic_tp_host_commit_snapshot == "host-c:1"
        assert [req.rid for req, _ in owner.agentic_kv_waiting_queue] == [host[3]]
    assert admitted_by_rank[0] == admitted_by_rank[1]
