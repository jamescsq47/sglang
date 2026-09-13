# R13: trusted-harness token hash opt-out and startup Host registration

User-requested comparison against R11: Qwen3.5-9B TP1, 2P:6D c500, finite500
SWE Verified tasks, same fixed harness/sampling. Q32/32 recorded with policy
recompute off, H2D4/P and R11 decoupled resident credit unchanged.

Only changes: SGLANG_AGENTIC_KV_TOKEN_CONTENT_HASH=false at every P/D participant,
and mandatory all-context Host registration before workload traffic. Content
hash default remains on in the engine. The off value is a non-SHA sentinel, not
a forged content digest; mixed on/off peers fail their existing comparisons.
Disabling it explicitly trusts unchanged token serialization and cannot detect
same-length content changes. It does not prove numerical KV/Mamba correctness.

Ownership map: content guard runs during D snapshot publication and P Direct/
Host admission. No ownership state, fence, CAS, lease, physical buffer, timeout,
fallback or cancellation transition changes. Identity/generation, exact claimed
parent length, layout and Mamba checkpoint checks remain. Host registration
occurs before any request owns a snapshot. Failure/timeout prevents traffic;
existing launcher cleanup drains all CUDA users before reclaiming arenas.

All eight acceptance criteria: 1–4 ownership/release paths unchanged; 5 removes
hash work and startup registration from the business critical path but does not
claim elimination of all scheduler overhead; 6 same flag on every rank and mixed
mode fails existing digest comparisons; 7 trusted stable-prefix contract,
remaining guards and per-task actual/ideal prefix accounting still required;
8 relevant regression tests and independent GO required before launch.

Host budget remains 2*(128+32)=320 GiB shared physical backing. Each P registers
288 GiB (both D2P arenas + its P2D arena), each D320 GiB (all arenas), using the
existing320 GiB registration cache. Mapping totals across contexts are not
additional physical copies. All8 completion records are required; startup
completion report is host_register_prewarm.json. Outer startup wait3600s,
registration wait1800s. These are startup-only, not task/tool timeout changes.

Report full500 completion/accuracy/tokens and the historical300–1500s slice.
This finite workload is not a replenished closed-loop capacity benchmark.

2026-09-12 prelaunch gate:611 lifecycle/fault/TP/Mamba tests passed; four
launcher/barrier CPU checks passed, plus bash syntax validation. Independent
audit `/root/audit_hash_off_prewarm` returned GO after independently passing
11 hash/prewarm tests and reviewing all eight criteria. Check actual eight
registration records before accepting performance data. No engine changes
outside this isolated checkout; shared installed pd environment untouched.
