# Fast Direct failures, including returned claims, recompute

- Base: `971a683e03`, restored measured 4888 runtime.
- Runtime change: `408718770b`; existing DMA fence and P claim return remain prerequisites.
- Fast identity: reuse verified arrival timestamp even after marker cleanup.
- Tool >1s still Slow. Direct setup deadline 1s, not a DMA completion deadline.
- Reviewer: Parfit, GO; 289 CPU tests passed with the integration PYTHONPATH.
- Qwen3-8B, BrowseComp source-order n680, 4P:4D TP1, c512, t0,
  P .80 and D .80/.80/.80/.60, warm300 + measure1200 after arena prewarm.

## r1 — stopped before traffic

Profile loader rejected `SGLANG_AGENTIC_KV_FAST_DIRECT_FAILURE_RECOMPUTE`:
the profile had the key, but the Slime loader's explicit allowlist did not.
No measured throughput. Added that one existing key to the allowlist;
real profile loader then passed. No engine changes for this startup repair.

## r2 — completed, 300s warmup + 1200s measurement

Run: `/homes/siqic/slime/examples/pd/runs-host/current/qwen3-8b-tp1-browsecomp-c512-w300-m1200/current-method-all-direct-fail-recompute-1s-20260909-r2`

Baseline reference: 4888.225 token/s; fast Direct acceptance 92.37%,
formal Direct6649/recompute424/fast-Slow125/slow-tool89.
Do not use the old 94.01% denominator, which excluded fast-Slow outcomes.

- Decode 4668.150 token/s (-4.50% vs reference); completed2752, failures0.
- Prefill 38707.34 token/s; per-card P/D Forward98.10%/99.64%.
- Per-card P/D KV45.68%/65.02%, D running43.14; parent-prefix token reuse91.12%.
- Formal-window Direct6413, recompute518, Slow80, known fast-tool Slow0.
- Fast Direct success6413/(6413+518)=92.53%.
- P2D Host queued=durable=P-release=D-restore=Host-release=2375 in the window.
- D2P Host80 writes and66 recoveries in-window: boundary outstanding is not loss.
- The more aggressive recompute policy did not improve throughput. Keep this
  as a measured comparison, not the default recommendation.
- Workers finished shutdown and GPU memory was released by02:57UTC;
  shutdown grace warnings occurred after measurement, not during serving.
