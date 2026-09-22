"""CPU-only NIXL API/fence contract tests, not bandwidth or GPU validation."""

import gc
import threading
import time
import weakref

import pytest

from sglang.srt.disaggregation.agentic_remote_host import (
    HostShard, ReadReceipt, RemoteHostTransport, group_ack, mha_read_spans,
    layout_fingerprint,
)


class FakeNixl:
    def __init__(self):
        self.name = "source-agent"
        self.status = "PROC"
        self.post_raises = False
        self.release_raises = False
        self.released = []
        self.deregistered = []
        self.reads = []
        self.metadata_raises = False
        self.deregister_raises = False
        self.removed_peers = []
        self.added_peers = []

    def remove_remote_agent(self, peer):
        self.removed_peers.append(peer)

    def register_memory(self, regions, mem_type, **kwargs):
        assert mem_type == "DRAM"
        return tuple(regions)

    def deregister_memory(self, registration, **kwargs):
        if self.deregister_raises:
            raise RuntimeError("registration busy")
        self.deregistered.append(registration)

    def get_partial_agent_metadata(self, registration, **kwargs):
        assert kwargs["inc_conn_info"]
        if self.metadata_raises:
            raise RuntimeError("metadata unavailable")
        return b"source-agent"

    def add_remote_agent(self, metadata):
        assert metadata == b"source-agent"
        self.added_peers.append(metadata)
        return "source-agent"

    def get_xfer_descs(self, regions, mem_type):
        return mem_type, regions

    def initialize_xfer(self, operation, local, remote, peer, **kwargs):
        assert operation == "READ"
        assert local[0] == "VRAM" and remote[0] == "DRAM"
        self.reads.append((local, remote))
        return object()

    def transfer(self, handle):
        if self.post_raises:
            raise RuntimeError("post submitted before exception")
        return self.status

    def check_xfer_state(self, handle):
        return self.status

    def release_xfer_handle(self, handle):
        if self.release_raises:
            raise RuntimeError("transfer still in flight")
        self.released.append(handle)


def test_peer_connection_persists_across_idle_read_gaps():
    agent = FakeNixl()
    transport = RemoteHostTransport(agent)
    sources = [export(transport, rank, 2) for rank in range(2)]
    transfers = [read(transport, source) for source in sources]
    for pending in transfers:
        pending.start()
    agent.status = "DONE"
    transfers[0].poll()
    transport.retire_read(sources[0].shard.export_id, "read-epoch")
    assert not agent.removed_peers
    transfers[1].poll()
    transport.retire_read(sources[1].shard.export_id, "read-epoch")
    assert not agent.removed_peers
    assert len(agent.added_peers) == 2
    transport.close_remote_peers()
    assert agent.removed_peers == ["source-agent"]


def test_peer_metadata_rollover_is_bounded_and_waits_for_live_reads():
    agent = FakeNixl()
    transport = RemoteHostTransport(agent, max_peer_imports=2)
    first = export(transport, address=1000)
    second = export(transport, address=1100)
    third = export(transport, address=1200)
    first_read, second_read = read(transport, first), read(transport, second)
    first_read.start()
    second_read.start()
    agent.status = "DONE"
    first_read.poll()
    transport.retire_read(first.shard.export_id, "read-epoch")
    assert not agent.removed_peers
    result = []
    worker = threading.Thread(target=lambda: result.append(read(transport, third)))
    worker.start()
    time.sleep(0.02)
    assert worker.is_alive() and not result
    second_read.poll()
    transport.retire_read(second.shard.export_id, "read-epoch")
    worker.join(timeout=1)
    assert not worker.is_alive() and len(result) == 1
    assert agent.removed_peers == ["source-agent"]
    assert len(agent.added_peers) == 3


def test_recycled_source_range_rolls_peer_after_live_reads_drain():
    agent = FakeNixl()
    transport = RemoteHostTransport(agent)
    first = export(transport, address=1000)
    recycled = export(transport, address=1000)
    pending = read(transport, first)
    pending.start()
    result = []
    worker = threading.Thread(target=lambda: result.append(read(transport, recycled)))
    worker.start()
    time.sleep(0.02)
    assert worker.is_alive() and not result
    agent.status = "DONE"
    pending.poll()
    transport.retire_read(first.shard.export_id, "read-epoch")
    worker.join(timeout=1)
    assert not worker.is_alive() and len(result) == 1
    assert agent.removed_peers == ["source-agent"]
    assert len(agent.added_peers) == 2


def export(transport, rank=0, size=1, address=None):
    return transport.export(snapshot_id="request:turn:epoch", tp_rank=rank,
        tp_size=size, layout=layout_fingerprint({"dtype": "fp16", "heads": 1,
                                               "dim": 2, "layers": 1}), token_count=3,
        address=1000 + rank * 100 if address is None else address,
        byte_size=24, keepalive=object())


def read(transport, source, rank=None, spans=None):
    shard = source.claim("read-epoch")
    return transport.prepare_read(shard, read_id="read-epoch",
        tp_rank=shard.tp_rank if rank is None else rank, tp_size=shard.tp_size,
        layout=shard.layout, gpu_id=shard.tp_rank,
        spans=spans or [(0, 10000, 12), (12, 20000, 12)])


def test_real_api_shapes_and_no_release_before_complete():
    agent = FakeNixl()
    transport = RemoteHostTransport(agent)
    source = export(transport)
    pending = read(transport, source)
    assert pending.poll() is None
    pending.start()
    assert pending.poll() is None
    assert source.keepalive is not None and not agent.deregistered
    agent.status = "DONE"
    receipt = pending.poll()
    assert pending.poll() == receipt
    assert not agent.deregistered  # DONE alone is not a whole-group release.
    assert source.release_after_group_ack([source.shard], [receipt], committed_read_id="read-epoch")
    assert not source.release_after_group_ack([source.shard], [receipt], committed_read_id="read-epoch")
    assert source.keepalive is None and len(agent.deregistered) == 1


def test_tp8_group_needs_every_shard():
    agent = FakeNixl()
    transport = RemoteHostTransport(agent)
    sources = [export(transport, rank, 8) for rank in range(8)]
    transfers = [read(transport, source) for source in sources]
    for transfer in transfers:
        transfer.start()
    agent.status = "DONE"
    receipts = [transfer.poll() for transfer in transfers]
    shards = [source.shard for source in sources]
    with pytest.raises(ValueError, match="all TP reads"):
        sources[0].release_after_group_ack(shards, receipts[:-1], committed_read_id="read-epoch")
    assert all(source.keepalive is not None for source in sources)
    for source in sources:
        assert source.release_after_group_ack(shards, receipts, committed_read_id="read-epoch")
    assert len(agent.deregistered) == 8


def test_wrong_rank_and_layout_rejected_before_nixl():
    agent = FakeNixl()
    transport = RemoteHostTransport(agent)
    source = export(transport, 3, 8)
    with pytest.raises(ValueError, match="mismatch"):
        read(transport, source, rank=4)
    assert not agent.reads


@pytest.mark.parametrize("spans", [
    [(0, 10000, 12)],  # missing KV tail
    [(1, 10000, 24)],  # skips prefix
    [(0, 10000, 12), (12, 10008, 12)],  # overlapping GPU memory
    [(0, 10000, 30)],  # overflow
])
def test_bad_spans_fail_before_post(spans):
    agent = FakeNixl()
    transport = RemoteHostTransport(agent)
    with pytest.raises(ValueError):
        read(transport, export(transport), spans=spans)
    assert not agent.reads


def test_ambiguous_post_retains_handle_until_physical_cancel():
    agent = FakeNixl()
    transport = RemoteHostTransport(agent)
    source = export(transport)
    pending = read(transport, source)
    agent.post_raises = True
    pending.start()
    assert pending.state == "UNKNOWN" and pending.handle is not None
    agent.release_raises = True
    with pytest.raises(RuntimeError):
        pending.drain_failure()
    assert pending.handle is not None and source.keepalive is not None
    agent.release_raises = False
    pending.drain_failure()
    assert pending.state == "DRAINED" and pending.handle is None
    assert source.keepalive is not None and not agent.deregistered
    with pytest.raises(RuntimeError):
        pending.poll()


def test_error_is_not_a_fence():
    agent = FakeNixl()
    transport = RemoteHostTransport(agent)
    pending = read(transport, export(transport))
    pending.start()
    agent.status = "ERR"
    assert pending.poll() is None and pending.handle is not None
    assert not agent.released


def test_receipt_epoch_and_duplicate_protection():
    transport = RemoteHostTransport(FakeNixl())
    source = export(transport)
    source.claim("read-epoch")
    shard = source.shard
    bad = ReadReceipt(shard.snapshot_id, "stale", 0, 1, shard.layout, 3, "read-epoch")
    with pytest.raises(ValueError, match="stale"):
        group_ack([shard], [bad])
    correct = ReadReceipt(shard.snapshot_id, shard.export_id, 0, 1,
                          shard.layout, 3, "read-epoch")
    with pytest.raises(ValueError, match="duplicate"):
        group_ack([shard], [correct, correct])
    with pytest.raises(RuntimeError):
        source.claim("another-reader")
    assert HostShard.from_dict(shard.to_dict()) == shard


def test_mha_coalesces_runs_but_preserves_layer_order():
    spans = mha_read_spans(layer_bases=[10000, 20000],
                          token_indices=[2, 3, 8], bytes_per_token=4)
    assert spans == [(0, 10008, 8), (8, 10032, 4),
                     (12, 20008, 8), (20, 20032, 4)]


def test_mha_duplicate_page_refuses_overwrite():
    with pytest.raises(ValueError, match="unique"):
        mha_read_spans(layer_bases=[10000, 20000],
                       token_indices=[2, 2], bytes_per_token=4)


def test_cancel_partial_tp8_then_retry_keeps_source():
    agent = FakeNixl()
    transport = RemoteHostTransport(agent)
    sources = [export(transport, rank, 8) for rank in range(8)]
    transfers = [read(transport, source) for source in sources]
    for transfer in transfers:
        transfer.start()
    agent.status = "DONE"
    receipts = [transfers[0].poll()]
    agent.status = "ERR"
    receipts += [transfer.drain_failure() for transfer in transfers[1:]]
    shards = [source.shard for source in sources]
    with pytest.raises(ValueError, match="cannot commit"):
        sources[0].release_after_group_ack(shards, receipts, committed_read_id="read-epoch")
    for source in sources:
        assert source.unclaim_after_group_cancel(shards, receipts)
        assert source.keepalive is not None
        source.claim("new-reader")
    assert not agent.deregistered


def test_tp_token_counts_must_match():
    transport = RemoteHostTransport(FakeNixl())
    sources = [export(transport, rank, 2) for rank in range(2)]
    wrong = HostShard(**dict(sources[1].shard.to_dict(), token_count=4))
    with pytest.raises(ValueError, match="mixed"):
        group_ack([sources[0].shard, wrong], [])


def test_layout_fingerprint_canonical():
    assert layout_fingerprint({"dtype": "fp16", "heads": 1}) == layout_fingerprint(
        {"heads": 1, "dtype": "fp16"})
    assert layout_fingerprint({"heads": 1}) != layout_fingerprint({"heads": 2})


def test_metadata_and_deregister_failures_keep_mapping_quarantined():
    agent = FakeNixl()
    transport = RemoteHostTransport(agent)
    agent.metadata_raises = True
    agent.deregister_raises = True
    with pytest.raises(RuntimeError, match="metadata unavailable"):
        export(transport)
    assert len(transport.quarantined) == 1
    assert next(iter(transport.quarantined.values()))["keepalive"] is not None
    with pytest.raises(RuntimeError, match="registration busy"):
        transport.retry_unpublished_cleanup()
    assert len(transport.quarantined) == 1
    agent.deregister_raises = False
    assert transport.retry_unpublished_cleanup() == 1
    assert not transport.quarantined


def test_release_failure_does_not_drop_source_mapping():
    agent = FakeNixl()
    transport = RemoteHostTransport(agent)
    source = export(transport)
    pending = read(transport, source)
    pending.start()
    agent.status = "DONE"
    receipt = pending.poll()
    agent.deregister_raises = True
    with pytest.raises(RuntimeError, match="registration busy"):
        source.release_after_group_ack([source.shard], [receipt], committed_read_id="read-epoch")
    assert source.keepalive is not None and not source.closed
    assert source.shard.export_id in transport.exports
    agent.deregister_raises = False
    assert source.release_after_group_ack([source.shard], [receipt], committed_read_id="read-epoch")
    assert source.shard.export_id not in transport.exports


@pytest.mark.parametrize("cancelled", [False, True])
def test_mixed_group_attempts_cannot_release_or_unclaim(cancelled):
    agent = FakeNixl()
    transport = RemoteHostTransport(agent)
    sources = [export(transport, rank, 2) for rank in range(2)]
    receipts = [ReadReceipt(source.shard.snapshot_id, source.shard.export_id,
        rank, 2, source.shard.layout, 3, "attempt-" + str(rank),
        "drained" if cancelled else "loaded") for rank, source in enumerate(sources)]
    with pytest.raises(ValueError, match="mixed TP recovery"):
        group_ack([source.shard for source in sources], receipts, cancelled=cancelled)


def test_dma_done_is_not_destination_ownership_commit():
    agent = FakeNixl()
    transport = RemoteHostTransport(agent)
    source = export(transport)
    pending = read(transport, source)
    pending.start()
    agent.status = "DONE"
    receipt = pending.poll()
    with pytest.raises(ValueError, match="ownership commit"):
        source.release_after_group_ack([source.shard], [receipt], committed_read_id="")
    assert source.keepalive is not None and not source.closed


def test_registry_retains_ambiguous_read_across_caller_gc():
    agent = FakeNixl()
    transport = RemoteHostTransport(agent)
    source = export(transport)
    pending = read(transport, source)
    agent.post_raises = True
    pending.start()
    reference = weakref.ref(pending)
    del pending
    gc.collect()
    assert reference() is not None
    key = (source.shard.export_id, "read-epoch")
    retained = transport.reads[key][1]
    assert retained.state == "UNKNOWN" and retained.handle is not None
    with pytest.raises(RuntimeError, match="unfenced"):
        transport.retire_read(*key)
    agent.release_raises = True
    with pytest.raises(RuntimeError):
        retained.drain_failure()
    assert key in transport.reads
    agent.release_raises = False
    retained.drain_failure()
    assert transport.retire_read(*key)


def test_duplicate_prepare_is_idempotent_but_cannot_change_target():
    agent = FakeNixl()
    transport = RemoteHostTransport(agent)
    source = export(transport)
    pending = read(transport, source)
    assert read(transport, source) is pending
    with pytest.raises(ValueError, match="changed destination"):
        read(transport, source, spans=[(0, 30000, 24)])
    assert len(agent.reads) == 1
    pending.start()
    agent.status = "DONE"
    pending.poll()
    assert read(transport, source) is pending  # Completion cannot cause repost.
    assert len(agent.reads) == 1
    assert transport.retire_read(source.shard.export_id, "read-epoch")


def test_registry_bounded_until_caller_retires_fenced_attempt():
    agent = FakeNixl()
    transport = RemoteHostTransport(agent, max_reads=1)
    first, second = export(transport), export(transport)
    pending = read(transport, first)
    with pytest.raises(RuntimeError, match="capacity"):
        read(transport, second)
    pending.drain_failure()
    transport.retire_read(first.shard.export_id, "read-epoch")
    assert read(transport, second)
