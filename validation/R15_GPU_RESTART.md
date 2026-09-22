# R15 a10/a11 restart — 2026-09-19 UTC

User confirmed both nodes recovered and authorized restarting the experiment.
Before launch: all 16 GPUs unused, PID1754686 absent, both nodes ~1.9TiB CPU
memory available; cross-node metadata visibility and exclusive flock verified.
CPU gate rerun: 831 passed, 2 skipped; 62 launcher tests passed. R15 independent
code audit previously GO. No engine/launcher changes in this restart.

Run: `dualpd/slime/runs/dualpd/qwen35-122b-a10p-a11d-tp8-c128-r15`.
a10=P, a11=D, Qwen3.5-122B-A10B BF16, each TP8/EP1, mem_fraction_static=.8,
mamba_full_memory_ratio=.5, c128, SWE Verified500 source-order once with
inline verifier. openai_tools, sampling .6/.95/20, 8192 per turn,64 turns,
81920 total tokens. Tool2s, Direct admission1s, congestion recompute disabled.

All 16 startup prewarm ranks completed before requests: P8GiB/rank ~3.7s;
D16GiB/rank ~7.5s. D initialized first; P cold model loading took ~5minutes.
Direct/Slow two-turn smoke passed: each cached8192 parent tokens,16 output
tokens exactly equal to its chunk-matched full-recompute reference. Slow
writer/loader/source-release ranks cover0..7, ledger consumed.
This smoke does not separately exercise P->D Host capacity backpressure.

01:21:30 UTC: workload loaded500 questions. Evaluation is running, NOT a
completed throughput or accuracy result. Per-node logs are local under
`/tmp/dualpd-multinode/qwen35-122b-a10p-a11d-tp8-c128-r15/`.
Coordinator uses owned component supervisors and existing fail/exit cleanup;
no pattern kill or GPU reset. NFS hard-mount kernel failures still cannot be
guaranteed recoverable by userspace SIGKILL. Do not stack another run on an
unreaped GPU owner.

Control from `dualpd/slime`:
```bash
bash tools/dualpd/qwen35_multinode.sh status --run-dir /homes/siqic/dualpd/slime/runs/dualpd/qwen35-122b-a10p-a11d-tp8-c128-r15
bash tools/dualpd/qwen35_multinode.sh stop --run-dir /homes/siqic/dualpd/slime/runs/dualpd/qwen35-122b-a10p-a11d-tp8-c128-r15
```
