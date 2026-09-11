# Qwen3.5-9B request-owned KV/Mamba: R9 (2026-09-11)

This revision checkpoints the tested implementation **before** H2D lane-lifetime
decoupling. It includes request-owned stable Mamba checkpoints, atomic hybrid
workset admission, application-owned termination and late-final Host cleanup.
Ordinary Qwen3 behavior is not opted into request-owned Mamba mode.

## Configuration and scope

Qwen3.5-9B, TP1, 2P:6D on eight A100 GPUs, c500; the same 500 distinct
SWE-bench Verified tasks once, no replenishment. Static memory fraction 0.80,
Mamba:Attention pool ratio 0.5, page64, 4 H2D lanes/P, Q32/32 recorded but
congestion and fixed-failure recomputation **disabled**. Native HiCache and
Mooncake off; custom NIXL Direct plus Shared Host Arena return of the matching
Attention prefix and stable Mamba checkpoint. Host arenas: 128 GiB/P D2P,
32 GiB/P P2D. Same external Miles/OpenEnv-local fenced-shell harness; temperature
0.6, top-p0.95, top-k20, min-p0, thinking enabled, 8192 output tokens/turn,
64 turns, context131072. Shell timeout600s, verifier2400s, Docker2CPU/4GiB.

This is a **finite 500-task evaluation**, not a replenished 300+1200s steady-state
benchmark. The common middle window is start+300 through start+1500 seconds.
Original colocated and fusion engine versions differ; not a pure same-engine
PD ablation. R9 also includes cleanup/admission fixes and is not solely a
2-vs4-lane ablation of R6.

## Results

| Metric | R9 |
|---|---:|
| Ended / completed / infrastructure-failed tasks | 500 / 498 / 2 |
| Resolved / all tasks | 160 / 500 (32.0%) |
| Full wall time | 3635.25 s |
| Time to 450 completed tasks | 1883.58 s |
| Per-task latency P50 / P90 | 1362.26 / 1879.20 s |
| Full actual Prefill / Decode tokens | 10,821,829 / 6,725,985 |
| Actual Prefill / ideal unaligned increment per task | 21,643.66 / 20,281.21 |
| Actual Prefill minus page64-aligned necessary increment | 0 for every task |
| Middle P / D compute tokens/s, total | 6173.23 / 4059.16 |
| Middle D tokens/s/card | 676.53 |
| Middle P / D Forward fraction | 71.92% / 97.23% |
| Middle D running/card | 30.85 |
| Middle P / D active Attention KV fraction | 18.68% / 51.02% |
| Middle P / D Mamba pool fraction | 19.08% / 32.46% |
| Direct / Host restored generations | 7243 / 14428 |
| Host durable / restored + final reclaimed | 14538 / 14428 + 110 |
| Final outstanding Host snapshots | 0 |

Two failures are environment timeouts; no verifier infrastructure failures.
Stable-prefix reuse excludes old reasoning removed by the unchanged harness;
it does not imply reusing the full generated tail.

## Recovery bottleneck diagnosis (not a causal performance attribution)

Middle-window Host-durable requests whose next turn arrived but whose recovery
I/O had not started averaged91.24 across both P workers. Timestamp reconstruction
splits this into13.22 before P HTTP reception,0.71 in API dispatch,73.52 after
dispatch but before recovery selection, and3.79 from selection to I/O start.
Worker timestamps have one-second precision. Scheduler delivery, scanning and
lane admission cannot be separated further with these logs.

Space-allocation miss counters stayed126 on P0 and133 on P1 throughout the
available middle-window progress samples. H2D I/O wall time averaged153.79ms;
GPU event time46.45ms. Each4-lane pool was sampled fully occupied57.5%/65.0% of
the time, but lanes covered preparation, allocation and scheduler handoff too.
This motivates decoupling physical lanes from non-I/O waits; it does not prove
that all28.08% non-Forward time is caused by admission or is recoverable.

## Reproduction and retained data

Launcher in jamescsq47/slime:
`examples/pd/scripts/new_method/run_qwen35_fused_swe500_2p6d.sh`.
Set `MAX_INFLIGHT=500 PD_FUSED_P_H2D_MAX_INFLIGHT=4
PD_FUSED_CONGESTION_RECOMPUTE=false PD_FUSED_CONGESTION_HIGH=32
PD_FUSED_CONGESTION_LOW=32`.

Run retained locally at:
`/tmp/pd-persist/fused-qwen35-9b-tp1-swe500-2p6d-c500-h2d4-q32-32-20260911-r9`.
It contains the exact precommit engine.patch, source snapshots, launch settings,
traces, raw responses, metrics and verifier artifacts. Raw logs are not uploaded.

Comparison tables and offline analysis are in jamescsq47/slime main:
`examples/pd/runs-host/SWEBENCH_QWEN35_9B_TP1.md` and
`examples/pd/runs-host/SWEBENCH_QWEN35_9B_TP1_MIDWINDOW.md`.
