"""Advisory global Slow-recovery congestion; never owns or releases KV."""

from collections import Counter
import json
import time


def waiting_for_recovery(entry):
    """Host is durable, but no shard has entered its H2D worker yet.

    io_inflight includes CPU transfer preparation, not just CUDA DMA. Pinned
    and leased claims are still waiting; a route/domain change is irrelevant.
    """
    if entry.get("state") not in {"host_ready", "h2d_loading"}:
        return False
    return not any(
        claim.get("phase") in {"io_inflight", "handed"}
        for claim in entry.get("recovery_claims", {}).values()
    )


class SlowRecoveryCongestion:
    """Router-event-loop owned references and one-second hysteresis."""

    def __init__(self, high, low):
        if not 0 <= low < high:
            raise ValueError("Slow congestion requires 0 <= low < high")
        self.high, self.low = int(high), int(low)
        self.parents = Counter()
        self.blocked = False
        self.high_samples = 0
        self.next_sample = 0.0
        self.latest = None

    def enter(self, snapshot_id):
        self.parents[snapshot_id] += 1

    def leave(self, snapshot_id):
        self.parents[snapshot_id] -= 1
        if self.parents[snapshot_id] <= 0:
            del self.parents[snapshot_id]

    def sample(self, q, now=None):
        now = time.monotonic() if now is None else now
        if self.latest is not None and now < self.next_sample:
            return self.latest
        self.next_sample = now + 1.0
        if q <= self.low:
            self.blocked = False
            self.high_samples = 0
        elif q >= self.high:
            self.high_samples += 1
            if self.high_samples >= 2:
                self.blocked = True
        else:
            self.high_samples = 0
        self.latest = dict(
            version=1, q=int(q), congested=self.blocked,
            high=self.high, low=self.low, sampled_at=time.time(),
        )
        return self.latest


class SlowCongestionReader:
    """D background worker: at most one existing pressure-file read/second."""

    def __init__(self, path):
        self.path = path
        self.next_read = 0.0
        self.sample = None

    def congested(self):
        now = time.monotonic()
        if now >= self.next_read:
            self.next_read = now + 1.0
            try:
                with open(self.path, encoding="utf-8") as handle:
                    payload = json.load(handle)
                sample = payload.get("slow_recovery") if isinstance(payload, dict) else None
                if not isinstance(sample, dict) or sample.get("version") != 1:
                    sample = None
                self.sample = sample
            except (OSError, ValueError, TypeError):
                self.sample = None
        sample = self.sample
        if sample is None:
            return False
        try:
            age = time.time() - float(sample["sampled_at"])
            return 0 <= age <= 3.0 and sample.get("congested") is True
        except (KeyError, TypeError, ValueError):
            return False
