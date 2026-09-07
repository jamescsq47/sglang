"""CPU-only address checks for the production indexed Host DMA descriptor path."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang.srt.disaggregation import agentic_host_staging as host_module


@pytest.mark.parametrize("host_to_device", [False, True])
@pytest.mark.parametrize("window_bytes", [None, 64])
def test_registered_descriptors_match_tensor_views(monkeypatch, host_to_device, window_bytes):
    layers, tokens, width = 3, 19, 4
    # Include a nonzero storage offset and padded layer stride.
    backing = torch.empty((2, layers, tokens + 3, width), dtype=torch.int32)
    host = backing[:, :, 1:tokens + 1, :]
    device_layers = [torch.empty((32, width), dtype=torch.int32) for _ in range(2 * layers)]
    item_size = width * host.element_size()
    snapshot = SimpleNamespace(
        kv_buffer=host, item_size=item_size, layer_num=layers,
        device_pool=SimpleNamespace(k_buffer=device_layers[:layers], v_buffer=device_layers[layers:]),
        _arena_mapping=None if window_bytes is None else SimpleNamespace(
            raw=backing, window_bytes=window_bytes,
        ),
    )
    indices = np.asarray([8, 9, 2, 3, 4, 18], dtype=np.int64)
    host_start = 2
    captured = []
    monkeypatch.setattr(host_module, "_cuda_driver_batch_memcpy", lambda: object())

    def capture(destinations, sources, sizes, *, stream):
        captured.extend(zip(destinations.tolist(), sources.tolist(), sizes.tolist()))
        return True

    monkeypatch.setattr(host_module, "_cuda_batch_memcpy_async", capture)
    assert host_module._registered_indexed_batch_copy(
        snapshot, device_indices=indices, host_start=host_start,
        stream=None, host_to_device=host_to_device,
    )
    # Compare every byte address so coalescing and registration-boundary splits
    # cannot hide a wrong offset, ordering, source, or destination.
    actual = {(dst + j, src + j) for dst, src, size in captured for j in range(size)}
    expected = set()
    for kv in range(2):
        for layer in range(layers):
            for pos, token in enumerate(indices):
                h = host[kv, layer, host_start + pos].data_ptr()
                d = device_layers[kv * layers + layer][int(token)].data_ptr()
                dst, src = (d, h) if host_to_device else (h, d)
                expected.update((dst + j, src + j) for j in range(item_size))
    assert actual == expected
    assert sum(size for _, _, size in captured) == len(expected)
