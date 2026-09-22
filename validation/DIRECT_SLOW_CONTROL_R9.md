# R9: event-driven control completion (not performance accepted)

## Evidence and bounded scope

R8 smoke passed but its c128 workload was intentionally stopped after about
eight minutes: Direct arrival-to-start remained 610 ms/P90 966 ms with a 1 s
setup deadline. Slow copy-to-group-release improved to 2.33 s, but most of this
was after every physical rank had completed its copy. See R8 for exact windows.

The next iteration does not change tool/setup deadlines, concurrency, Host
capacity, workset sizes, congestion recompute, routing, or model parameters.
It removes avoidable control waits in the existing TCP TP>1 implementation.
TP1 and non-TCP control keep their established paths.

## Direct

Use the existing asynchronous control client for claim and completion control
operations so one pending RPC does not block progress for other Direct sessions.
The native TP allocator epoch still grants the complete workset. Admission is
not a transfer fence. All RPC results retain exact attempt/lease ownership;
cancelled or stale results cannot start a new transfer or free a live target.
An in-flight claim must be drained before retiring its local ownership context.

## Slow

Use the existing Host worker and TP progress publisher for the completed loaded
and bound ACKs, not merely the handed ACK. These are immutable completion facts
identified by control key, claim, workset lease and remote-read epoch. Invalidate
the context before retry or cancellation; old completions must never advance a
new attempt. Native scheduler BIND still mutates Radix, COMMIT still hands the
workset to the request, and ADMIT still requires the all-rank handed condition.
No additional scheduler collective or background allocator is introduced.

## Lifecycle and required gate

- Unique owner: unchanged physical fences and claim/CAS identities; async
  completion is not permission to transfer ownership early.
- P2D Direct and P2D Host: no protocol or release policy changes.
- D2P Host: source release still requires durable full-group completion.
- Independent progress: pending per-snapshot RPCs must not stall other lanes
  or scheduler Forward. Failure remains explicit/fail-closed.
- TP atomicity: one rank0 policy, immutable group commands, native ordered
  BIND/COMMIT/ADMIT; no follower makes independent admission decisions.
- Reuse: no new recompute/eviction path or changes to page/checkpoint semantics.
- Tests: delayed ACK, one-rank lag, cancel/retry identity, late completion,
  ownership teardown and TP1 regressions; independent audit GO before GPU run.

Work is in progress. CPU tests, independent verdict and GPU result must be
recorded after execution; this plan is not evidence of completion.

## Cancellation boundary identified during pre-run review

After native BIND but before COMMIT, the lease is `binding`, not `handed`.
HTTP cancellation must invoke the existing scheduler-owned
`rollback_bound_parent` before removing its pin. Generic `cancel_unstarted`
or `release_handed` cannot retire it. The hook is limited to TCP TP>1.
`test_agentic_tp_host_cancel_binding.py` executes native HTTP cleanup with real
workset brokers/CPU page allocators for TP2 and TP8, including duplicate cleanup
and all-rank suffix retirement: 2 passed. This proves page retirement, not Host
record cleanup; the separate pending/CONSUMED receipt cancellation tests must
also prove mapping, job, resident credit and lane quiescence before audit GO.

Host implementation is frozen: 21 new worker tests and 509 related checks
passed. Independent component review ran 119 tests and returned Host GO,
including cancellation after CONSUMED, one failed mapping-close retry, old
epoch callback isolation and phase-3 clamping before bound ACK. This is not
the combined GPU gate; Direct changes and full regression still require GO.

## Frozen regression gate

Full `validation/check_multinode_cpu.sh`: **1271 passed, 2 GPU-only skipped**;
launcher/diagnostics **84 passed**. Raw output `/tmp/dualpd-r9-cpu-gate.log`.
Direct focused gate: 60 passed, including actual multi-snapshot group progress
after one complete/bound/route Future fails (known and unknown outcomes).
The review-driven error handling is per session: definitive pre-CONSUMED
failure uses the original abort path; unknown outcomes retain resources; an
acknowledged CONSUMED followed by cleanup failure cannot become rollback-safe.
No new executor, allocator thread, timeout policy or recovery lane is added.

The initial full gate caught an extra legacy `ledger.get` in the new Host
CONSUMED cancellation branch. It now executes only for event-control ledgers;
the full suite above includes that regression. Independent combined GPU GO
was subsequently granted: the independent frozen Direct+Host gate passed 179
tests and diff-check, returning combined GO for same-config TP8 c128. This is
not a claim of GPU correctness or performance acceptance.

## GPU experiment

Launching `qwen35-122b-swe500-tp8-c128-socket-full-r9` with
`bash tools/dualpd/qwen35_multinode.sh run --concurrency 128 --run-dir
/homes/siqic/dualpd/slime/runs/dualpd/qwen35-122b-swe500-tp8-c128-socket-full-r9`.
Both nodes' GPU process queries were empty before launch. Same tool2s/setup1s,
memory0.8/Mamba0.5, both D2P paths, P2D unchanged, congestion recompute disabled.
Smoke and load-test results remain pending.

### Recorded outcome: stopped, not accepted

Both Direct/Slow two-turn token-equality smoke tests passed. The c128 workload
started at 06:36:01 UTC on 2026-09-21 and was intentionally stopped after about
three minutes. A repeated `tp_direct_bind_commit_retry` exception dereferenced
`entry.workset_lease.lease_id` after a legal native handoff cleared the lease.
The asynchronous control operation must retain its immutable attempt identity
across handoff; reading identity from a transferred resource is incorrect.
No evidence of wrong KV was found, but this is not a passing evaluation.
Both nodes' GPU process lists, run-labelled containers and coordinator PID
were checked absent after owned coordinator cleanup.

At 06:38:48, excluding smoke, TP0 had 318 Direct starts/267 completions and
229 Slow selections/225 copies/224 group releases. Completed-stage diagnostics:
arrival-to-Direct-start 579 ms (P90 967), intent-to-grant 257 ms,
grant-to-start 144 ms, claim 73 ms, metadata preparation 2.2 ms. Slow
copy-to-loaded-ACK 33 ms, loaded-ACK-to-bound-queue 518 ms,
bound-queue-to-ACK 154 ms, bound-ACK-to-P-release 937 ms,
P-release-to-handed-ACK 119 ms. These are interrupted-run cohorts, not full-run
throughput or identical cohorts suitable for adding into an exact critical path.

The P `shared_host_group_commit_release` event is local post-COMMIT cleanup,
not the source D Arena physical release timestamp. Source Host cleanup can
already follow authoritative CONSUMED plus remote-read fences. In particular,
RemoteHostDescriptor.close is a no-op, so the 937 ms is not evidence of slow
memory unregister. Native COMMIT currently initiates the asynchronous logical
manifest fence only when preparing COMMIT, adding a scheduler round.

A 20 s nonblocking py-spy sample of P rank0 found 55/332 main-thread samples
in Triton cache put/makedirs and 63/332 in compiler stacks. Sampling errors and
CPU stack residency prevent interpreting this as GPU idle percentage. The
launcher had no Triton cache override; the default home cache is on NFS.
The next run will explicitly use node-local compiler cache, a separately
recorded change, not proof that all of this time was NFS waiting.
