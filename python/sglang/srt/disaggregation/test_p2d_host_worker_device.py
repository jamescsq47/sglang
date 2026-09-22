"""Host workers select their TP-rank GPU before touching queued snapshots."""

import threading
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.disaggregation.p2d_host_staging import (
    AgenticPToDHostLoadManager,
    AgenticPToDHostStagingManager,
)


@pytest.mark.parametrize("rank", [0, 3])
@pytest.mark.parametrize("manager_type", [
    AgenticPToDHostStagingManager, AgenticPToDHostLoadManager,
])
def test_worker_selects_owner_stream_before_dequeue(monkeypatch, rank, manager_type):
    # The owner creates an explicitly indexed stream, but a fresh Python/CUDA
    # worker starts on GPU0 even in a follower process whose owner uses GPU3.
    state = {"current": 0, "selections": []}
    stream = SimpleNamespace(device=torch.device("cuda", rank))

    def set_device(device):
        assert device.index is not None
        state["selections"].append(device.index)
        state["current"] = device.index

    monkeypatch.setattr(torch.cuda, "set_device", set_device)
    manager = manager_type.__new__(manager_type)
    manager._stop = threading.Event()

    class Queue:
        def get(self, *args, **kwargs):
            # No queued item can reach materialize, lease preparation, Host
            # registration or index handling on a different current device.
            assert state["current"] == rank
            assert state["selections"] == [rank]
            return None

    manager._work = Queue()
    manager._worker(0, stream, None, ())
    assert state["selections"] == [rank]
