# TP1 Host-event refill — engineering experiment, 2026-09-13

Scope: the copied/fused engine only; installed `pd` and `pd_mamba_baseline`
packages and external dataset harnesses are not modified. Default is OFF.
Enable with `SGLANG_AGENTIC_KV_P_HOST_EVENT_PROGRESS=true` plus the existing
`SGLANG_AGENTIC_KV_P_H2D_DECOUPLED=true`. Both require TP1, Mamba, and request-owned
state. TP2 and dense Qwen3 retain their previous paths.

Previously H2D already ran on background workers. The new change is narrower:
after the existing safe-boundary completion sweep, consume at most 64 completion
or Host-ready hints, run one continuation of the original FIFO admission, service
its full-workset allocation intents, and start selected loads in the same boundary.
This continuation shares the initial admission budget; it does not add priority by
Direct/Slow/New type. Events are hints, not ownership or completion authority.
Missing/foreign/stale hints retain normal level-triggered discovery.

No GPU allocator or Radix operation moves to an unsynchronized worker thread.
Copy kernels, full Attention+Mamba fences, ledger CAS, routes, and fallback policy
are unchanged. This reduces an avoidable scheduler-turn delay; it does not remove
all scheduler scanning or prove that every non-Forward interval will disappear.

Ownership affected: preparation of D2P_HOST_OWNED -> fenced handoff -> P_HBM_OWNED.
Success still requires the original copy completion and claim commit. Failed or
cancelled work retains the original source and target fence/release handling.
Capacity shortages still leave metadata pending, not a partial hybrid workset.

## Pre-GPU gate

488 tests passed: event progress, H2D decoupling, resident Mamba admission,
request-owned Mamba, lifecycle, TP, TP Host commit/readiness/cancellation, final
Host cleanup. Independent `/root/audit_host_event_progress`: code GO.

Eight acceptance checks:

1. Unique physical ownership: unchanged; event cannot grant ownership.
2. P2D Direct release: unchanged.
3. P2D Host durable release: unchanged.
4. D2P Host durable source release: unchanged.
5. Progress: bounded continuation, no DMA wait/poll loop; existing I/O threads.
6. TP: new mode cannot activate for TP>1; collective/state logic unchanged.
7. Parent correctness: same complete KV+Mamba fence, no new recompute decision.
8. Gate: regression suite and independent code GO before GPU launch.

## Reproducible short validation

`RUN_DIR=<fresh-directory> bash validation/run_host_event_smoke.sh`

Uses GPUs P=0,6 / D=1,7, Qwen3.5-9B, TP1, .80 static memory, .5 Mamba ratio,
page/track64, context32768, deterministic sampling for token equality checks.
Small 16GiB D2P + 8GiB P2D arenas per P, fully pre-registered before requests;
content hashes OFF, native HiCache/Mooncake OFF, congestion recompute OFF.
Only the existing synthetic transport client is used, not a replacement SWE harness.
Eight clients each execute two three-turn chains (0s/3s simulated tools), then
recompute the same prompts without parent reuse for exact token comparison.
This is 96 model calls, including 32 actual cross-turn reuse attempts.

The launcher uses the existing dependency-ordered process supervisor and isolated
ports/control paths, never touching other experiment processes. Results retain
raw requests, client traces, metrics and final ledgers. GPU validation must still
account for queued/durable/source-release/consumed and every outstanding snapshot.
This short test is NOT a 300+1200 serving throughput acceptance run.

## Completed result

R2: `/tmp/pd-persist/qwen35-9b-tp1-2p2d-host-event-smoke-20260913-r2`
with `TEST_CLIENTS=16`: 192 model calls (96 trajectory + 96 reference),
96/96 exact token matches, 64/64 cross-turn restores (32 Direct + 32 Host).
Both P workers logged actual event-refill execution. All parent prefix hits
matched the expected page64 prompt checkpoint. Host offers = durable copies =
D source releases = P restore releases = 32 (identical snapshot ID sets).
No nonterminal Host entries remained BEFORE service shutdown; no missing/double route.
The per-snapshot event ledger retains192 P2D `rejected` metadata receipts for
unused Host offers, not payload. Both the registry and event files are checked.
P2D Host spill was not exercised by this low-load test (0), so its coverage is
regression tests, not this GPU run. P2D Direct and D generation releases were
both 192. Final application releases were128 (32 chain finals +96 references).

Selected->grant mean3.86ms, I/O start->fence mean23.89ms, fence->handoff mean9.86ms;
these are short-test stage timings, NOT an A/B speedup or full serving benchmark.
The final one-time INFO observation addition also passed36 focused tests.

R1 is retained for provenance:48/48 exact matches, but the old diagnostic omitted
reference-generation final ACKs under APP_OWNS_TERMINATION. Only the validation
client was fixed (no production SWE harness changes), then R2 repeated the test.

Offline strict accounting:
`python validation/summarize_host_event_smoke.py <R2-directory> --clients 16`
Full results in `smoke-summary.json`.
