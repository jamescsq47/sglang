"""Native Scheduler aliases require the installed CUDA driver, even for CPU tensors."""

import pickle
import typing

import pytest


def test_historical_aliases_eager_annotations_and_pickle_identity():
    import numpy as np
    import torch
    from sglang.srt.disaggregation import agentic_workset as workset
    from sglang.srt.managers import scheduler

    for name in ("AgenticPWorksetLease", "AgenticPWorksetLeaseBroker"):
        cls = getattr(workset, name)
        assert getattr(scheduler, name) is cls
        assert pickle.loads(pickle.dumps(cls, protocol=0)) is cls
        old_reference = ("csglang.srt.managers.scheduler\n" + name + "\n.").encode()
        assert pickle.loads(old_reference) is cls
    assert scheduler.AgenticEarlyDirectReceive.__annotations__["workset_lease"] == (
        typing.Optional[workset.AgenticPWorksetLease])
    assert workset.AgenticPWorksetLease.__annotations__["device_indices"] is torch.Tensor
    assert workset.AgenticPWorksetLease.__annotations__["parent_page_indices"] is np.ndarray
    for cls in (workset.AgenticPWorksetLease, workset.AgenticPWorksetLeaseBroker):
        typing.get_type_hints(cls)
        for method in vars(cls).values():
            if isinstance(method, staticmethod):
                method = method.__func__
            if callable(method):
                typing.get_type_hints(method)
    lease = workset.AgenticPWorksetLease(
        snapshot_id="historical:0", lease_id=1, owner="legacy",
        parent_tokens=4, parent_allocated_tokens=4, prompt_tokens=8,
        allocated_tokens=8, device_indices=torch.arange(8),
        parent_page_indices=np.array([0]))
    restored = pickle.loads(pickle.dumps(lease, protocol=2))
    assert type(restored) is scheduler.AgenticPWorksetLease
    assert restored.snapshot_id == lease.snapshot_id
    assert restored.state == "active"
    assert torch.equal(restored.device_indices, lease.device_indices)
    # A live broker owns an RLock and was never serializable as an instance.
    with pytest.raises(TypeError):
        pickle.dumps(workset.AgenticPWorksetLeaseBroker(page_size=4), protocol=2)
