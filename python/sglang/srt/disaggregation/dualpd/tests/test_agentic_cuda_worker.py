import pytest
import torch

from sglang.srt.disaggregation.agentic_cuda_worker import run_rank_bound_worker


@pytest.mark.parametrize('rank', [0, 1, 3])
def test_rank_device_selected_before_callback(monkeypatch, rank):
    events=[]
    monkeypatch.setattr(torch.cuda, 'set_device', lambda d: events.append(('device',d)))
    def callback(value):
        assert events==[('device',rank)]
        events.append(('callback',value))
        return 42
    assert run_rank_bound_worker(rank,callback,'arg')==42
    assert events==[('device',rank),('callback','arg')]


def test_failed_device_selection_cannot_start_transport(monkeypatch):
    def failed(_): raise RuntimeError('device failure')
    monkeypatch.setattr(torch.cuda,'set_device',failed)
    with pytest.raises(RuntimeError,match='device failure'):
        run_rank_bound_worker(1,lambda:pytest.fail('transport must not start'))


def test_worker_exception_is_not_converted_to_completion(monkeypatch):
    monkeypatch.setattr(torch.cuda,'set_device',lambda _:None)
    def failed():raise RuntimeError('transfer failed')
    with pytest.raises(RuntimeError,match='transfer failed'):
        run_rank_bound_worker(1,failed)
