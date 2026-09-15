# Host asynchronous preparation — 2026-09-14

Opt-in `SGLANG_AGENTIC_KV_P_HOST_ASYNC_PREPARE=true` requires the existing
Host-event progress and decoupled H2D modes (TP1, request-owned Mamba only).
Installed environments, dense Qwen3, TP>1, harness and sampling are unchanged.

Native FIFO admission registers an immutable prompt view in an existing bounded
H2D lane. The control worker materializes Host mappings, validates the existing
snapshot, claims it, and submits a full-workset allocation intent. Only the
scheduler services the allocator. The worker observes the grant, prepares the
existing ledger attempt and starts H2D without a second scheduler gate visit.
Radix binding and the live request remain scheduler-owned.

Cancellation during CPU preparation is deferred to the same worker, including
a new HTTP rid referring to the same request-generation. The existing physical
fence/abort path executes after CPU preparation quiesces. A failed claim or lease
attachment which releases the admission resource also retires the descriptor;
the original queue must select it again. It does not create an unbounded retry
queue or permission to recompute.

Ownership: D2P_HOST_OWNED remains authoritative until the original complete
Attention+Mamba fence and commit. Capacity shortage retains the Host claim and
bounded intent, not a partial workset. I/O exceptions retain the same attempt;
shutdown uses the existing supervised service-group cleanup.

Eight-invariant review:

1. Unique owner: unchanged claim/lease/CAS; registration is metadata only.
2. P→D Direct source release: unchanged.
3. P→D Host durable source release: unchanged.
4. D→P Host durable D release: unchanged.
5. Progress: CPU preparation moved off scheduler; no allocator/Radix in worker.
6. TP atomicity: new path cannot activate for TP>1; existing TP path retained.
7. Parent reuse: same snapshot/checkpoint and full copy fence, no new fallback.
8. Gate: lifecycle/cancellation/capacity tests plus independent audit are required
   before GPU execution. GPU smoke must check exact tokens and ownership counts;
   full SWE500 results, not this document, determine performance acceptance.

CPU test: `PYTHONPATH=python python -m pytest -q
python/sglang/srt/disaggregation/test_agentic*.py`.
GPU smoke: set `SGLANG_AGENTIC_KV_P_HOST_ASYNC_PREPARE=true` and a fresh `RUN_DIR`,
then run `bash validation/run_host_event_smoke.sh`.
Full launch adds `PD_FUSED_P_HOST_ASYNC_PREPARE=true` to the unchanged R18 settings.

Pre-GPU gate passed: 691 CPU regression tests; independent
`/root/audit_host_event_progress` GO (28 new preparation/fault/race tests).
This is a code-safety gate, not performance acceptance.

GPU smoke completed successfully:
`/tmp/pd-persist/qwen35-9b-tp1-host-async-prepare-smoke-20260914-r19`.
48/48 exact token matches; 32 cross-turn restores (23 Direct, 9 Host), no missing
or double route. All page64 parent checkpoints matched. Host offer = durable =
D source release = P restore release = 9, same snapshot sets; no nonterminal
payload before shutdown. P2D Host was not exercised. Supervisor exited 0 and
all owned GPU processes exited before the full run. No throughput claim is made
from this short synthetic test.
