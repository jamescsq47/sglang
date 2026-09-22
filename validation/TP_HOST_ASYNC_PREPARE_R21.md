# TP Host background preparation: R21 experiment

This is an opt-in preparation optimization on top of the R20 retry-fence
repair documented in `HOST_RECOVERY_R21.md`. It does not change routing,
workset sizes, Host capacity, tool deadlines, or recompute policy.

## Ownership and execution

- State: D2P_HOST_OWNED, with an eventual temporary destination workset.
  Host remains authoritative until the existing all-rank recovery commit.
- Rank0 still selects the bounded Host recovery pipeline. Each scheduler
  queues an immutable request view, without per-request filesystem access.
- The existing control worker performs metadata validation, Host claim,
  complete-workset intent, lease attachment and TP PREPARE acknowledgment.
- The original rank0 allocator plan remains the only page-allocation order.
  Worker preparation cannot authorize DMA: `start_allowed` stays false until
  the existing group START command. Radix bind and group commit are unchanged.
- A pending prepare acknowledgment is not reported as a prepared TP shard.
  Scheduler status reads retain one load reference rather than racing two
  dictionary lookups against worker cleanup.
- Capacity failure retains Host and the bounded admission descriptor. Retry
  returns to the existing epoch-fenced cleanup. Cancellation is serialized
  through the preparation worker, including cancellation before enqueue.
  No source or destination can be recycled before actual DMA/no-I/O proof.
- TP1 and other launchers remain unchanged. The new switch is
  `SGLANG_AGENTIC_KV_TP_HOST_ASYNC_PREPARE`, default false, effective only with
  TP>1 and multi-node configuration. The Qwen a10/a11 launcher opts in and
  records `tp_host_async_prepare=true` in its immutable experiment config.

## Acceptance criteria review

1. Unique owner: unchanged Host claim and all-rank commit.
2. P→D Direct release: unchanged.
3. P→D Host durable release: unchanged.
4. D→P Host durable release: unchanged.
5. Progress: preparation leaves the compute thread; no new waiting barrier.
   The existing control worker can still spend time in per-entry filesystem
   I/O. This change does not claim complete removal of control latency.
6. TP atomicity: original rank0 allocation/START/bind/commit commands preserved.
7. Reuse: no added fallback or implicit recompute, all physical fences retained.
8. CPU cancellation/retry/TP tests and independent review required before GPU.

## Evaluation plan

Same R20 workload: a10 P/a11 D, Qwen3.5-122B-A10B, TP8/EP1 per node,
c128 SWE Verified500 once with verifier; memory .8, Mamba ratio .5,
openai_tools, temperature .6/top_p .95/top_k20, 8192 per turn/64 turns,
native HiCache/Mooncake and congestion recompute disabled. Run-owned launcher
checks Direct/Slow smoke before evaluation and uses owned process cleanup.

Compare selected→prepared, H2D acknowledgment, final handoff and worker-stage
timings against pre-stall R20, alongside Host backlog, P/D Forward, running,
reuse and error counts. Do not infer network saturation from end-to-end waits.
The earlier QP creation failure is not established as fixed by this change.

## Validation progress

- Launcher unittest gate: 65 passed; shell syntax and `git diff --check` passed.
- First targeted run: 43 passed, 5 failed. Four new remote-ledger fixtures
  incorrectly used `/tmp` instead of an allowed control root; corrected to
  isolated `/dev/shm` temporary directories. The fifth used the old AST mode
  fixture without the new flag; updated to exercise both old TP1 and opt-in
  multi-node TP2/TP8 modes. No lifecycle assertion was removed.
- Independent static audit GO, conditional on final CPU gate. Existing
  scheduler-side blocking index mirror precedes lease publication, so moving
  the consuming view to the preparation worker requires no new CUDA sync.
- Final CPU gate: 893 passed, 2 skipped, 17 dependency warnings; 387.19 s
  dominated by NFS import waits. Launcher gate: 65 passed in 2.973 s. Exit 0.
- a10/a11 preflight passed, matching environment/model/data and cross-node
  control locking; both nodes had no competing GPU processes. Run directory:
  `slime/runs/dualpd/qwen35-122b-a10p-a11d-tp8-c128-r21`.
- Physical run pending; no performance claim yet.
- Additional fresh-process TP2/TP8 no-imported-record cancellation regressions:
  2 passed (264.37 s import-dominated). Original TP retirement blocks early
  CLEAR; terminal FAILED/drained/no-destination receipts permit source cleanup.
  No further engine change was needed. Independent final audit: GO.

## Physical run started

2026-09-19 15:48 UTC: launched the existing owned coordinator with
`bash tools/dualpd/qwen35_multinode.sh run --run-dir
/homes/siqic/dualpd/slime/runs/dualpd/qwen35-122b-a10p-a11d-tp8-c128-r21
--concurrency 128`. P and D supervisors started; initially importing/loading.
The launcher must pass readiness, Host prewarm and Direct/Slow smoke before
SWE500 begins. This is a running validation, not a completed result.
