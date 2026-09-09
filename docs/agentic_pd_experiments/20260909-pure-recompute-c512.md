# 2026-09-09 custom-PD pure-recompute c512

- Engine commit: `3aaba1a46a7aa5b1c5911cb3f1d6c436b243dc6c`
- Workload: Qwen3-8B, BrowseComp source-order n680, TP=1, 4P:4D, c512,
  temperature=0, 301.18 s warmup + 1200.01 s measurement.
- Memory: P `0.80 × 4`; D `0.80/0.80/0.80/0.60`.
- Scope: current custom P-to-D late binding, D Router and P-to-D Host staging
  remain enabled. `SGLANG_AGENTIC_KV_DISABLE_D2P_REUSE=true` removes parent
  metadata and disables both D-to-P Direct and D-to-P Host preservation.

## Result

- Decode: **1938.839 token/s total**, 484.710 token/s/D.
- Prefill compute: 43009.000 token/s total.
- Completion: 0.95665 agents/s; 1148 agents; zero failures.
- Prefill/agent: 44890 tokens; Decode/agent: 2024 tokens.
- P Forward: 99.99%/GPU; D Forward: 98.94%/GPU.
- P KV: 8.62%; D KV: 10.73%; P queue: 118.83/P; D running: 6.96/D.

Formal-window lifecycle evidence: 4090 finished D generations produced 4090
`reverse_reuse_disabled` records and 4090 D release records. D-to-P Direct
offers and D-to-P Host writes/restores were all zero. P-to-D remained enabled;
because P was the bottleneck and D retained spare capacity, P-to-D Host was not
needed in this run. No OOM, NIXL, Router 500, scheduler exception, or request
failure occurred.

Relative to the adopted full method (5148.410 Decode token/s), disabling only
D-to-P reuse reduced Decode throughput by 62.34% and increased actual
Prefill/agent from 14719 to 44890 tokens. The generic analyzer reports 5.45%
cached parent-prefix tokens from local/common Radix hits; reverse D-to-P reuse
is exactly zero.

Canonical Slime artifact:
`examples/pd/runs-host/current/ablations/browsecomp-qwen3-8b-4p4d-c512/pure-recompute-20260909-r1`.
