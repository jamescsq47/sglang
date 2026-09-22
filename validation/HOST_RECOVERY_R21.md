# R20 failure diagnosis and R21 recovery-lifecycle repair

## Evidence and scope

R20: a10 P / a11 D, Qwen3.5-122B-A10B, TP8 per node, c128,
SWE-bench Verified 500 source-order once. Native HiCache/Mooncake and
congestion recompute remain disabled. No workload or memory change is planned.

- Evaluation started 2026-09-19 03:15:20 UTC. Last completed episode at
  04:34:18: 117 completed/verifier records, 79 resolved. This is a censored
  subset, **not** final accuracy or a completed performance result.
- At 04:34:19, P Host READ workers failed inside NIXL `add_remote_agent`:
  UCX `mlx5dv_devx_obj_create(QP)`, syndrome `0x19c6d6`, remote I/O error.
  A live link does not establish why QP creation failed. Resource exhaustion,
  connection churn and device/firmware failure are not yet distinguished.
- Last P Prefill 04:35:22, last D Decode batch 04:35:23. On inspection around
  14:06 both groups were idle; P/D attention KV metrics were 5.4% / 1.1%.
  Router remained alive and retried timeouts/500s. HTTP liveness was not useful
  workload progress.
- Four Slow entries remained ABORTING with no loader drain acknowledgments.
  Example `5660f0adb41b499ca4bac1a5d0c89025:57`: ledger epoch advanced past
  remote attempt epoch 7, which contained only one rank's no-I/O receipt.
  Earlier epochs had complete 8-rank drain proofs. Missing ranks had neither
  destination nor receipt in epoch 7; this was not proof of an in-flight DMA.
- Engine code exposes two premature retirement edges: the no-load retry branch
  acknowledges without publishing its remote no-I/O receipt, and failed-load
  cleanup removes its context before confirming the final retry ACK.
  The next attempt then encounters an unfenced previous epoch and is
  quarantined. Later HTTP cancellation cannot repair that lost context.
- Stopped via run-owned supervisors. Remote stop initially timed out; subsequent
  checks confirmed both nodes have no GPU compute processes. No GPU reset or
  name-based kill was used.

## Before-failure performance (not network-only timing)

| UTC window | Rank0 H2D completion samples | Selected to prepared | Selected to H2D ACK | Selected to handoff | Rank0 H2D wall | Rank0 Host READ worker |
|---|---:|---:|---:|---:|---:|---:|
| 03:20–03:30 | 983 | 1.382 s | 2.162 s | 3.186 s | 800 ms | 168 ms |
| 04:20–04:30 | 794 | 1.606 s | 2.673 s | 3.977 s | 1067 ms | 300 ms |

Stage times use one-second log timestamps; window cohorts differ slightly.
H2D wall includes Host READ preparation and completion publication, not only
DMA. These figures cannot be used as NIC saturation measurements. The P
control loop averaged about 6.7/7.0 ms in the two windows. Earlier stack samples
found scheduler per-entry shared locks, lifecycle marker operations and TP
synchronization even after historical scans moved to the watcher. Selected
latency does not include the longer HOST_READY queue wait. Fixing cancellation
alone must not be reported as eliminating those steady-state costs.

The worker column comes from `host_copy_ms`, not `elapsed_ms` (which is NaN
for RDMA). Even this worker time includes remote metadata preparation and
receipt operations; it is not pure DMA time. In the late window, roughly
0.77 s of the 1.07 s H2D wall time is outside the actual worker invocation,
and the complete selected-to-handoff interval is about 3.98 s. This supports
separating worker queue/progress, preparation and TP handoff from network
bandwidth; it does not prove how much of each wait can safely be overlapped.

## Ownership plan before edits

The affected state is D2P_HOST_OWNED plus a fenced, temporary P workset. Host
remains the source until normal all-rank commit. Each local recovery keeps the
exact group attempt/epoch until its local DMA completion or no-I/O proof,
owner-scoped lease retirement and retry/cancel ACK are durable. Only then may
the local load context/lane disappear. Next-attempt admission must not overtake
the old attempt's proof. Real unfenced posted READs remain quarantined.

Normal completion, Direct routing, P2D buffering and capacity admission do not
change. Capacity failure leaves Host intact. Partial failure/cancellation
converges through existing group retry/abort transitions. Shutdown retains
unfenced buffers until process teardown. No timeout substitutes for a fence.

## Acceptance gates

1. Unique snapshot owner: Host remains authoritative throughout failed recovery.
2. P2D Direct release: unchanged.
3. P2D Host durable/source release: unchanged.
4. D2P durable/source release: unchanged; destination retry cannot free Host early.
5. Progress isolation: failure of one recovery cannot consume all lanes forever;
   no new bulk scans or waits for another transfer path. Remote prepare captures
   the claimed ledger epoch before constructing its load; this adds a per-entry
   read at preparation while removing the mutable-epoch read at READ startup.
   This correctness repair does not eliminate existing scheduler/ledger I/O.
6. TP atomicity: exact group epoch, all-rank proof before retry; no independent
   per-rank route/recompute decisions.
7. Parent correctness: no new implicit recompute/eviction or relaxed DMA fence.
8. CPU failure/cancellation regression plus independent GO before any GPU run.

Implementation and test results will be appended after review. R20 is failed,
not an accepted full-run result. QP trigger remains a separate investigation.

## Additional checks

After engine teardown, a CPU-DRAM-only NIXL check (`CUDA_VISIBLE_DEVICES=''`,
`UCX_TLS=rc`, `UCX_NET_DEVICES=mlx5_1:1`) completed 200 metadata import/disconnect
cycles on a10 in 0.659 s. It allocated two temporary NIXL agents and 4 KiB of
Host memory, then exited. This is not a cross-node throughput or long-duration
leak test; it only shows QP creation was not persistently broken after teardown.
No firmware/driver setting or transport backend was changed.

Existing read retirement deliberately invalidates remote-agent metadata after
its final in-flight read to bound stale registrations. This also disconnects
the peer. Connection churn is a plausible contributor but is not proven to
cause this run's device error; retaining metadata indefinitely would introduce
new stale-key/capacity risks and is not part of this repair.

Added worker-only aggregate stage timings (one log per 64 successful reads):
descriptor preparation, group-attempt claim, destination publication,
NIXL preparation, transfer progress, receipt publication and retirement.
There are no new filesystem probes, CUDA synchronizations or scheduler calls.
These distinguish control latency from transport latency in the next run;
they are instrumentation, not a claimed throughput improvement.

## Implemented repair (GPU validation still pending)

- Remote recovery freezes its claim/epoch before preparing the destination.
  Neither cleanup nor a delayed completion can act on a newer attempt.
- A rank that never started READ publishes the existing bridge's no-I/O
  receipt before acknowledging retry or prestart abort. It cancels the frozen
  workset owner, not a potentially different HTTP retry request ID.
- Failed-load cleanup retains its context and lane until the old workset has
  actually retired and the retry ACK has committed. Lost ACK responses are
  idempotent; an old epoch ACK cannot acknowledge a successor epoch.
- Successful retirement clears the old record's attempt/claim before the next
  real prepare. A peer's ABORTING transition supersedes local retry, using the
  original physical-drain cleanup rather than waiting for another HTTP abort.
- A real cancellation regression found that `request_host_load_failure` did
  not accept remote RETRY_PENDING. That state now enters the existing ABORTING
  transition with unchanged owner checks and all-rank drain requirements.
  Local recovery's accepted states are unchanged.
- Normal Direct/Slow admission, lane counts, routing, workset sizing, eviction,
  workload settings and local TP1 execution policy are unchanged. No real
  unresolved DMA is freed merely because a timeout expired.

Regression cases cover TP2/TP8 partial connection failure, missing no-I/O
receipts, failed/lost retry ACK, actual next-epoch workset preparation, changed
HTTP request ID and cancellation through source cleanup. Read diagnostics are
also tested not to change successful transfers even if logging fails.

Independent review identified and corrected stale record epochs and missing
prestart-abort receipts before any GPU rerun. CPU validation temporarily
stalled in NFS `rpc_wait_bit_killable`; a stalled process is not a passing
test. Final gate results will be recorded separately below.

### Verification status

- Remote bridge suite, including diagnostic failure isolation: **20 passed**.
- First recovery suite run: **29 passed, 2 failed**. One failure was an
  incomplete test fixture, the other exposed the RETRY_PENDING cancellation
  predicate above. Both were corrected; this run is not reported as a pass.
- Lifecycle changes were authored by `multinode_hybrid_review` and independently
  reviewed by the main agent. Diagnostics were authored by the main agent and
  independently reviewed by `multinode_hybrid_review`. Static review GO is not
  a substitute for final fault-test results or GPU validation.
- Final complete CPU gate: **871 passed, 2 skipped**, followed by **64 launcher
  unittest tests passed**, plus shell syntax check; exit status 0. Command:
  `DUALPD_PYTHON=/homes/siqic/anaconda3/envs/pd_multi_node/bin/python PYTEST_ADDOPTS='--tb=short --durations=10' bash validation/check_multinode_cpu.sh`.
  Pytest took 249.11 s, dominated by the observed import-time NFS waits; launcher
  tests took 2.979 s. One preceding full-gate failure was an old hybrid-test
  fixture missing the now-required frozen attempt ID. Only the fixture was
  corrected; the final run above is clean and no fence assertion was removed.
- Independent static audit **GO**, fault-test gate **PASS**. No R21 GPU run has
  been launched. QP trigger and steady-state recovery performance are still
  unverified on real two-node hardware after this change.
