# TP1 capacity-only Direct recompute: formal trial did not meet performance target

Request: fast tools should receive Direct promptly when a complete workset is
available. Only a definitive complete-workset capacity refusal may cause
recompute. No claim, an unknown delay, or setup failure must use Slow; slow
tools also use Slow. Reuse existing control polling.

Source base: 3f508399c2cc8d1d7025cbee903ca9c7eba8d5fa.
New explicit opt-in: SGLANG_AGENTIC_KV_DIRECT_CAPACITY_RECOMPUTE=true.
It is mutually exclusive with the older unconditional Direct failure recompute
ablation flag. TP1-only allocator refusal/mirror changes do not certify TP>1.

Changes: atomic never-granted capacity refusal through the existing manifest
CAS; nonblocking pinned CPU mirror of Direct GPU page IDs with cancellation
fences; refresh the existing D manifest once when receiver metadata appears.
No P-to-D, Host DMA, Forward, batch-size, or memory-fraction changes.

Validation: independent GO; 299 CPU regressions, 52 lifecycle tests, one real
CUDA mirror/cancellation test. Formal run: Qwen3-8B, BrowseComp source-order
n680 cyclic, TP1 4P:4D, c512, temperature0, seed2026, 301.25s warmup +1200s
measurement. Ordinary cards mem_fraction_static=.80; GPU7 with search=.60.
D-to-P Host128GiB/P, P-to-D Host32GiB/P; native HiCache/Mooncake off;
tool/setup deadlines1s, D target1.0, P-to-D grace.5s.

| Formal measurement | Old mixed-policy reference | This revision |
|---|---:|---:|
| Decode token/s | 4888.225 | 4669.864 (-4.47%) |
| Fast-tool Direct success | 92.3729% | 89.7869% |
| Direct complete | 6649 | 6110 |
| Fast recompute | 424 | 550 |
| Fast-tool Slow | 125 | 145 |
| Receiver-started abort | 125 | 43 |
| P Forward/card | 97.037% | 96.624% |
| D Forward/card | 99.616% | 99.552% |
| P KV/card | 50.60% | 57.97% |
| D running/card | 48.44 | 45.21 |

Success denominator includes Direct + fast recompute + fast-tool Slow.
Mean Direct start latency did not improve:375.48->378.22ms.
Whole-run659 capacity refusals match659 recomputes by snapshot ID, with zero
missing/invalid evidence. P-to-D Host all4683 queued/durable/source-released/
restored/destination-released. D-to-P Host465 durable/source-released,396
restored/Host-released; pending-at-end items must not be labeled lost.

Performance acceptance FAILED. Do not promote this revision as a faster
default. The allocator currently refuses at its first real shortage instead
of retaining the opportunity within the original1s deadline. That may reject
transient shortages too early, but no counterfactual proof was captured; do
not attribute all additional recomputes to it without another controlled test.
Independent log audit matched tool elapsed for624 recomputes:622 finalized
within1s of tool return,475 within500ms, median approximately286ms. These
timings demonstrate early refusal, not proof that a later grant was possible.
No second revision or GPU run was launched. Matrix remains paused.

Full records in canonical Slime:
examples/pd/runs-host/current/qwen3-8b-tp1-browsecomp-c512-w300-m1200/current-method-capacity-direct-1s-20260909-r1/
including RESULT.md, AUDIT.md, capacity_comparison.json and metrics/logs.
