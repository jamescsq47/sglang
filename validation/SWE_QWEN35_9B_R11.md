# R11: TP1 Mamba H2D lane decoupling — no throughput gain

2026-09-11. Full 500-task SWE Verified evaluation, Qwen3.5-9B TP1,
2P:6D c500, same harness/sampling/static memory as R9. See H2D_DECOUPLING.md
for the ownership protocol, opt-in guard, R10 integration failure and 603-test
plus independent-audit GO validation. This is not a replenished steady workload.

| Common +300/+1500s window | R9 | R11 |
|---|---:|---:|
| P compute tokens/s total | 6173.23 | 5893.05 |
| D tokens/s total | 4059.16 | 3994.63 |
| P Forward fraction | 71.92% | 67.74% |
| D Forward fraction | 97.23% | 95.02% |
| D running/card | 30.85 | 30.89 |
| Host eligible waiting I/O, total | 91.24 | 62.58 |
| Physical lanes occupied/P, sampled | 3.1875/4 | 1.075/4 |
| P inflight queue/card | 5.88 | 8.59 |

Physical-lane release does reduce unnecessary occupancy, but it is not enough
to raise compute utilization. Resident credit is 8/P while physical lanes stay
4/P, an explicit bounded staging-budget change. No new fixed HBM pool, no
background allocator/Radix writes, no routing/timeout/recompute/harness change.
Keep default OFF; effective only for TP1 request-owned Mamba.

R11: 496 completed +4 environment errors; 148/500 resolved (29.6%, R9 32%).
Wall3538.043s, T450-completed1879.674s vs R9 1883.575s. Mean/P50/P90 task
latency1277.28/1357.36/1872.14s. Single stochastic evaluations are not evidence
that a 12-task score difference is caused by KV correctness or scheduler policy.

All500 per-task actual Prefill equals the fully retained page64 stable-prefix
counterfactual: actual10,532,772; ideal excluding alignment9,871,391;
boundary661,381 tokens. Output6,783,518 tokens;21,606 turns. No excess history
recompute, no Host eviction. Unique-snapshot conservation:

```
6196 Direct +15039 Host +269 app-final +102 length =21606
15039 Host durable =15039 D source release
14910 Host restored +129 final cleanup =15039
6196 Direct +14910 Host restored =21606-500
P2D Host:48 queued=durable=P source release=D restored=Host release
```

All owned GPU workers/containers/watcher exited; co-tenant GPU7 PID1868643
untouched. CUDA shutdown temporarily lagged but finished within the existing
cleanup process; no remaining experiment CUDA context after verification.

Full report and JSON/CSV summaries are in jamescsq47/slime main:
`examples/pd/runs-host/SWEBENCH_QWEN35_9B_H2D_DECOUPLING.md`.
Raw logs/traces remain local at
`/tmp/pd-persist/fused-qwen35-9b-tp1-swe500-2p6d-c500-h2d4-decoupled-q32-32-20260911-r11`.
Run source =9262db4fa3 +saved engine.patch. R10 is a stopped integration failure,
not a performance result. Do not infer P idle-time attribution from queue counts;
complete CPU/bind/bootstrap/Forward tracing remains a future diagnostic step.
