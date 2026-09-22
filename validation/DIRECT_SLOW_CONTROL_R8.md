# Direct admission / Slow handoff optimization (in progress)

## Baseline and scope

The accepted strict Direct-wait r7 ablation is stopped; both a10 and a11 GPU
process lists were checked empty before development. Its performance is not a
dual-path baseline. Compare the next dual-path run with `socket-full-r4` using
the same Qwen3.5-122B, TP8 per node, SWE Verified500 order, c128, memory 0.8,
Mamba ratio 0.5, tool 2s / Direct setup 1s and congestion recompute disabled.

r4 did not finish the dataset. Diagnostic counts (one logical snapshot, not
eight rank reports): Direct complete 2498; fallback 1841, of which 1733 have
explicit tool elapsed <=2s; 1227 fallbacks lack a P receiver-start record.
This last category is not proof of insufficient HBM. D2H wall mean was 0.267s;
Host durable-to-selection 76.86s includes possible tool waiting; selected-to-
prepared 0.85s, prepared-to-copy-complete 1.52s, copy-to-group-release 3.18s.
The fixed final 300s had 253 D2H completions, 169 recovery releases and 78
terminal eviction/recompute transitions. P2D Host completions were zero.

## Ownership boundaries retained

1. Direct: D retains its source until all physical shard fences and the existing
   group ownership commit complete. P reserves the entire workset in the native
   TP allocator epoch; no background thread mutates allocator or Radix.
2. Slow: Host remains pinned through actual remote READ completion and local
   Radix binding. Metadata completion may run in the Host control worker, but
   only rank0's native admission command makes every rank enter Prefill.
3. Capacity rejection, cancellation, failure, and shutdown retain existing
   exact lease/attempt identities and physical fences. No timeout is a fence.
4. TP1 retains its established handoff protocol; TCP TP>1 optimization must not
   introduce filesystem mailboxes or new synchronous RPCs on the scheduler.

## Measurements required

- Direct arrival-to-start and intent-to-grant / grant-to-start, success versus
  fast-tool fallback; count logical snapshots separately from shard events.
- Slow copy-to-group-handoff, write/restore/eviction rates and outstanding Host
  age; do not mistake source Host waiting for GPU DMA time.
- P/D Forward, D running, Attention/Mamba usage and wall-clock output tokens.
- No TP divergence, stale-attempt release, premature admission, unfenced DMA
  release, orphan lease or runtime NFS control.

CPU lifecycle/fault tests and an independent GO are required before launching.
This document records work in progress, not successful performance validation.

## Implemented changes

- Socket TP hybrid workset allocation constructs compact CPU Attention page
  IDs and Mamba snapshot slot IDs in one blocking mirror at the original native
  allocator boundary. This preserves index readiness; the NIXL metadata worker
  no longer performs a second GPU-to-CPU slot read while holding its control
  lock. Runtime Mamba slots are not transfer destinations.
- Socket TP Direct selection accounts for queued, not-yet-granted broker
  intents in its existing leader admission budget. Followers still execute
  rank0's existing grant; this does not introduce independent follower policy
  or claim a strict per-rank DMA semaphore during shard timing skew.
- Slow HBM-ready RPC uses the existing asynchronous control client, allowing
  other physical lanes to progress while one acknowledgement is pending.
- Native Slow COMMIT still hands the workset to the live Req on the scheduler.
  Host mapping cleanup and the exact claim/lease/read-epoch handed ACK then run
  in the existing Host control worker. A shared publisher reports completion
  directly; stale scheduler observations cannot erase that completed fact.
  Only rank0's native ADMIT makes the whole group runnable.
- PREPARE/START/BIND commands and bind ACK retain their existing protocol in
  this iteration. This is not a claim that all Slow scheduler rounds have been
  eliminated. GPU stage measurements decide the next bottleneck.
- Cancellation invalidates positive notifications, retains pending ACK context
  until drained, and uses original abort/physical-fence retirement. Failed or
  ambiguous ACKs retain the workset and are reported as errors, not success.

## Acceptance mapping (code/CPU scope; GPU evidence pending)

1. Unique owner: allocator publication and Host claim/lease/epoch guards remain.
2. P2D Direct release: no changes.
3. P2D Host release: no changes; preserve r7 cached physical terminal fix.
4. D2P Host source release: no changes to durable/source-release authority.
5. Independent progress: remove metadata GPU read under Direct lock and move
   handed cleanup off scheduler; physical-resource contention still exists.
6. TP atomicity: allocator epoch and native BIND/COMMIT/ADMIT remain; tests hold
   one shard ACK and verify no rank enters Prefill early.
7. Parent reuse: no new eviction/recompute policy, timeout or transfer length.
8. Gate: tests and independent audit recorded below before GPU execution.

## Frozen CPU gate

`DUALPD_PYTHON=/homes/siqic/anaconda3/envs/pd_multi_node/bin/python bash
validation/check_multinode_cpu.sh`: 1234 passed, 2 GPU-only skipped; 81 launcher
and diagnostic tests passed. Raw log: `/tmp/dualpd-r8-cpu-gate.log`.
Both repositories pass `git diff --check`. Independent `control_audit` verdict:
GO for same-config two-node TP8 c128 validation; independently executed 37
Direct and 105 Host checks. It verified CONSUMED forbids retry/eviction and HTTP
abort retains release_handed ordering. GO is not performance acceptance.

## GPU run

Started `qwen35-122b-swe500-tp8-c128-socket-full-r8` on 2026-09-21 UTC using
`bash tools/dualpd/qwen35_multinode.sh run --concurrency 128 --run-dir
/homes/siqic/dualpd/slime/runs/dualpd/qwen35-122b-swe500-tp8-c128-socket-full-r8`.
Preflight and both Direct/Slow two-turn token-equality smoke checks passed.
D2P Direct and Host both enabled, P2D unchanged. The workload started at
06:01:20 UTC. It was intentionally stopped through the owned coordinator at
06:09:21 UTC because Direct admission still failed the performance objective.
All model workers on both nodes stopped; GPU process queries were empty and
run-owned tool containers were removed. Logs and evaluation artifacts remain.

This is a diagnostic, interrupted run, not a completed SWE evaluation. Excluding
smoke, TP0 reported 877 Direct starts, 715 group completions, 625 Slow selections,
621 copies and 619 group releases. The final 300-second log window ended at
06:09:22: Direct arrival-to-start mean 610.4 ms (P90 966.1), intent-to-grant
279.2 ms, grant-to-start 143.5 ms. Slow selected-to-copy mean 1.56 s and
copy-to-group-release 2.33 s (P90 3, max 5). Log timestamps have one-second
precision; these completed-stage cohorts exclude pending/cancelled work.

An earlier complete 300-second sample over 402 TP8 Slow groups showed last-rank
copy-to-first-release 2.197 s, while rank copy skew was only 0.132 s. Thus the
remaining seconds cannot be explained by a single late shard's DMA alone.
Direct leader intent-to-grant averaged about 260 ms in the same inspection;
followers, which create the intent inside the native allocation epoch, spent
only 2.3–2.9 ms. Allocation misses were zero in that sample. The next change
must address native-epoch/control handoff waits, not increase capacity or
timeouts, and must retain physical fences and native allocator/Radix ownership.

Prometheus counter deltas from 06:02:08.962 to 06:07:43.873 UTC (334.91 s)
gave Decode 427.6 output tokens/s, P Forward 73.14% and D Forward 82.29% per
physical GPU (eight rank execution counters summed and divided by eight and
wall time). This is an early diagnostic window, not a finished evaluation or
a completed-trajectory-normalized comparison.
