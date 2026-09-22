"""CPU regression for the scoped CUDA validation driver's header observation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.mark.parametrize("size", [1, 2, 8])
def test_driver_requires_current_header_identity_before_republish(size):
    path = Path(__file__).resolve().parents[4] / "validation" / "workset_gpu.py"
    spec = importlib.util.spec_from_file_location("workset_gpu_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cached = {"command_id": 1, "command_acks": list(range(size))}
    client = NS(size=size, entry=lambda *_: dict(cached))
    assert module.exact_header_complete(client, "headers", "round", 0)
    assert module.exact_header_complete(client, "headers", "round", 1)
    # Header 2 was enqueued but the local pushed mirror still has header 1.
    assert not module.exact_header_complete(client, "headers", "round", 2)
    cached.update(command_id=2, command_acks=list(range(size - 1)))
    assert not module.exact_header_complete(client, "headers", "round", 2)
    cached["command_acks"] = list(range(size))
    assert module.exact_header_complete(client, "headers", "round", 2)
