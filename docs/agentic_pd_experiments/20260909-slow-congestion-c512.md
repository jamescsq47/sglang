# c512: advisory global Slow-recovery congestion

Status: audited; formal run pending. Base: b47459357f (all fast Direct failures
recompute, measured4668.150 token/s). No allocator or transport changes.

- Tool>1s still Slow. Fast Direct setup deadline1s remains unchanged.
- Count unique active parent generations: tool returned, Host durable, waiting
  for H2D worker admission. Cross-P/domain moves do not double count. Host data
  still waiting for a tool is excluded. io_inflight includes CPU preparation;
  this Q is not an actual-DMA waiting queue.
- Reuse the Router's existing ledger traversal/pressure publication. Two 1s
  samples at Q>=32 enable explicit recompute after fast Direct failure;
  Q<=8 restores Slow. Slow tools never become congestion-driven recomputes.
- D reads the advisory file at most once per second. Missing/stale>3s/malformed
  state falls back to Slow. The switch defaults OFF.
- No physical ownership from Q: existing P claim return, DMA fence, FAILED CAS,
  durable recompute route, and D release order remain required. TP rank0 decides.
- Independent audit Parfit: GO; independently reran446 CPU tests successfully.
- Eight design acceptance criteria checked; runtime conservation still pending.

## Formal configuration

Qwen3-8B, BrowseComp fixed source-order n680 cycling, c512, temperature0,
4P:4D TP1. P GPUs0/2/4/6 .80; D1/3/5/7 .80/.80/.80/.60, searchGPU7.
Host D2P128GiB/P, P2D32GiB/P;2 H2D lanes/P; globalHostrouting; no native
HiCache/Mooncake. D_TARGET1.0, P2Dgrace0.5s. Arena prewarm barrier, then
300s business warmup +1200s formal measurement.

Run: `/homes/siqic/slime/examples/pd/runs-host/current/qwen3-8b-tp1-browsecomp-c512-w300-m1200/current-method-slow-congestion-1s-20260909-r1`

Compare with4888.225 mixed historical exit and4668.150 all-fast-failure-recompute.
Report throughput, P/D Forward/KV/running, globalQ/mode timeline, Direct/fastSlow/
slow-tool/recompute, extraPrefill and reuse. Check CPU-preparation-to-DMA delay
and every outstanding generation. Similar throughput with less needless
recompute is acceptable; no claim of optimal thresholds before measurement.
