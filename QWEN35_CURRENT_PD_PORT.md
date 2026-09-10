# Qwen3.5 on current agentic PD

## Frozen base and isolation

Working tree: `/homes/siqic/sglang-qwen35-integration`.
Branch: `codex/qwen35-current-pd`. Original tree:
`/homes/siqic/sglang-h100-integration` (not edited by this port).

Engine base: `921fbd46ab6d3baf68be66f097661a918dec6707`, plus the
two uncommitted static Host recovery changes captured in `d38015245f`.
Source worktree and existing pd/pd_mamba environments are not modified.
The port is implemented and bounded compatibility validation passes. TP1/TP2
multi-turn return, TP2 all four Direct/Host paths, and Qwen3 old/new token
equivalence pass. No high-concurrency or formal throughput result is claimed.

## Scope and ownership contract

Preserve current Qwen3 control plane, deadlines, congestion policy, TP protocol,
registered-window DMA, and allocator semantics. Hybrid hooks run only when a
Mamba pool exists. Adapt proven pd_mamba payload/checkpoint logic, not its old
scheduler or old Decode manager. No whole upstream SGLang upgrade.

All four P2D/D2P Direct/Host handoffs treat Attention pages and matching Mamba
conv/temporal slots as one snapshot. The old owner retains both resources
until every component's physical fence completes. Destination allocation
reserves Attention workset plus Mamba active/checkpoint slots atomically.
Capacity failure rolls back the entire grant. Cancellation, timeout, failure,
and shutdown must settle all component DMA before releasing source/destination
resources; TP ranks make one grouped path/commit/release decision. Host durable
requires every component written, and Host removal requires complete load and
bind. Existing explicit eviction/recompute paths remain explicit, not silent
partial cache hits. Reference-counted Radix ownership remains authoritative.

P2D must carry current active state plus the appropriate page checkpoint;
D2P must carry the checkpoint matching the advertised Attention prefix.
For unchanged token histories only the page tail is recomputed. SWE histories
that remove prior reasoning require an explicit stable-prefix mode with
separately reported suffix recompute, never a claim of complete tail reuse.

## Acceptance gates

1. Unique composite owner at every generation transition.
2. P2D Direct complete+bind releases P resources.
3. P2D Host durable releases P resources without waiting for D capacity.
4. D2P Host durable releases D resources without waiting for tools/P.
5. Direct/Slow/control/Forward progress remain decoupled.
6. TP grouped claim, failure fences, commit and release.
7. Exact prefix/checkpoint correspondence and separately counted recompute.
8. Relevant regression/fault tests and independent audit GO before GPUs.

Validate Qwen3 unchanged behavior and Qwen3.5 all four paths, digest equality,
multi-turn reuse, cancellation, resource return, TP1 then TP2. Final serving
performance acceptance requires 300 s warmup + 1200 s measurement; smoke tests
must not be reported as formal performance. External harness stays unchanged.

## Validation progress (2026-09-10)

- 453 CPU tests pass: integration safety, TP lifecycle/failure tests, lifecycle
  metadata, prompt checkpoints, hybrid DMA fences and allocator rollback.
- Independent `audit_current_mamba_port` audit: GO for isolated small GPU
  smoke only. TP2 now has the checks below; formal performance is unvalidated.
- Fixed audit findings: complete-extent/mapping reference accounting;
  Attention-only event cannot commit the composite transfer; bounce buffers
  retained before partial submissions; preallocated P2D Host publication;
  short-prefix Radix release and native tombstone-ancestor deletion semantics.
- Added opt-in exact conv/temporal hashes alongside Attention hashes. Enabled
  only for correctness diagnostics, disabled for throughput measurement.
- `validation/run_smoke_services.sh` uses the isolated engine overlay and an
  isolated launcher copy. Explicit `extra_buffer` is required on this base;
  its native `auto` selects `no_buffer`, unlike the newer donor environment.
  Native pd/pd_mamba environments and shared Slime scripts remain untouched.
- Smoke r1 rejected incompatible launch strategy before serving; r2 rejected
  control ledger outside `/dev/shm`. Both exited and GPU memory was reclaimed.
- r3 exposed the old Mamba cache's split full/state capacity API; corrected
  the hybrid-only capacity query. r4/r5 exposed launcher flags that require
  multiple P domains; the isolated single-P smoke now disables those flags.
- `validation/check_host_payload.py`: four physical GPU→Host→GPU checks pass,
  covering complete Attention+conv+temporal payload, 1/2 state slots, and
  registered-DMA requested on/off. This is not full serving validation.
- r6 P→D active+checkpoint hashes match on real Qwen3.5-9B, but first reverse
  snapshot explicitly failed: native prebuilt cache insertion clears tracking
  length even though Req ping-pong slots remain alive (Radix forks a copy).
  The hybrid-only fix preserves that length after successful native insertion;
  exception/dense/feature-off cases are covered. Synthetic continuation chat
  boundaries were also corrected; the production harness was not changed.
- r6 is a failed diagnostic, not an accepted result. All owned model/router
  processes were stopped and GPU memory reclaimed before r7.
- TP1 r7: fast/Host slow each three turns, both 9-token and 87-token output
  variants pass; 12/12 outputs equal full-recompute references token-for-token.
  All 32 observed per-transfer Mamba hash pairs match. Reverse cache reuse was
  observed, not inferred from HTTP completion. Persisted diagnostics:
  `validation/results/tp1-r7/` (gitignored; no formal throughput claim).
- r7 long-prompt diagnostic exposed a launcher conflict: deterministic Triton
  uses 4096-token truncation alignment, but the smoke had chunk=2048. Prompts
  above that chunk could not be admitted. Isolated smoke now uses chunk=4096;
  engine scheduling policy is unchanged.
- TP2 r1: first transfer/reverse import worked, then native idle-only Radix
  sanity failed while a legal TP grouped-release owner still held a lock.
  Fixed hybrid-only idle gating using the existing atomic ownership ledger
  (TP pending release does not increment the native token counter). P broker
  leases likewise denote non-idle work. Empty ownership restores all checks.
  Failed logs: `validation/results/tp2-r1-failed/`.
- First-sampled-token completion now enters the existing manager only for
  tagged agentic Mamba requests. It excludes the uncomputed sampled token,
  respects response-pending gating, and uses normal terminal finalization.
  Dense and feature-off branches are unchanged and covered by regression.
- Qwen3 r1 did not complete the reverse path: its first Direct setup reached
  the deadline with diagnostic hashing enabled; cause/comparative baseline
  not established. This is not a passed Qwen3 regression. Logs retained in
  `validation/results/qwen3-r1-incomplete/`.
- TP2 r2: 6/6 long-prompt multi-turn outputs (about 4.7k prompt, 87 output)
  match full-recompute references; another 6/6 first-token completion outputs
  match. Both TP ranks have matching observed Mamba transfer hashes. Final
  P/D loads return to zero requests and zero used tokens. This covers P2D
  Direct, D2P Direct and D2P Host, not yet P2D Host.
- Qwen3 new-tree r2 and original-tree r1: all six prompt arrays and all six
  output arrays match across trees. Each tree also passes 6/6 full-recompute
  references. Hash diagnostics were disabled for this comparison; no original
  files or installed environments were changed. Records:
  `validation/results/qwen3-r2/`, `validation/results/qwen3-base-r1/`.
- P2D Host probes in TP2 r2 deliberately made D unavailable via an isolated
  diagnostic router. First probe hit the inherited Host metadata restriction
  on `return_logprob=True`; the diagnostic client now uses native output IDs
  with logprobs disabled. Second probe exposed the lazy arena adapter's missing
  two-state layout before materialization. Both probes eventually fell back
  to Direct: correct outputs do NOT establish Host-path correctness.
- Lazy Host fix: retain immutable CPU state-index metadata without mapping or
  CUDA work; construct the full layout and install indices in the copy worker
  before publication. Failure closes the unpublished mapping without unlinking
  the arena. Dense construction and D2P default one-slot layout are unchanged.
  Real memfd-arena and TP preparation-rejection rollback tests pass; independent
  review GO received before starting TP2 r3.
- TP2 r3 actually completed P2D Host on both ranks: queued, durable, P source
  release, D restore and Host consumed/release all observed. However subsequent
  D2P recovery passed GPU slot indices to the CPU-only Lazy setter and failed;
  the entire r3 is therefore a failed diagnostic, not acceptance. Logs retained
  in `validation/results/tp2-r3-failed/`.
- D2P setup now mirrors the slot vector once in the background H2D worker.
  Failed-load retry settles the complete fence and old workset before clearing
  its destination binding; durable Host data/mapping remains reusable with a
  new lease. Real arena tests cover in-flight refusal, new-slot retry and stale
  callback isolation. No scheduling/deadline/ownership policy changed.
- Final TP2 r4 passes. Long-prompt test: 4729/4852/4975 prompt tokens, 87
  generated tokens each; fast and slow trajectories each three turns, 6/6
  exact against full-recompute references. Additional first-token completion
  test: 899/935/971 prompt tokens, one output token, another 6/6 exact. Combined
  64/64 observed per-rank conv/temporal digest comparisons match, no missing
  sources and no engine exception markers. `audit_current_mamba_port` reviewed
  real logs and returned bounded compatibility GO.
- Actual P2D Host snapshot `p2d:2015185672659118382`: each rank staged 4729
  tokens / 128991232 bytes. Both ranks queued, completed durable and released
  P at 04:44:26 UTC; D restore and Host consumed/release followed at 04:44:30.
  This is proof of release-before-D-capacity, not a Direct fallback. Both D2P
  Direct and D2P Host also complete with matching state hashes.
- After tests, saved P/D `/get_load` each reports zero requests, zero physical
  used tokens and zero transfer/prealloc queues. Both saved Host ledgers have
  empty `entries` and `relays`. All owned model/router processes were stopped;
  `nvidia-smi` shows no compute processes. This is not a dump of every internal
  allocator field; do not equate an HTTP zero-running count alone with proof
  of complete ownership reclamation.
- Accepted r4 artifacts: `validation/results/tp2-r4/`, including both trace
  JSONs, per-rank hash comparison, final loads and pre-shutdown control records.

## Acceptance summary and limits

All eight design criteria were reviewed at the tested scope: composite
Attention+state ownership and rollback have CPU fault tests; Direct and both
Host source-release transitions have physical GPU evidence; H2D Host release
follows full load/bind; I/O remains in the inherited independent workers;
TP2 ranks transfer/commit/release together; multi-turn outputs match fresh
Prefill references; every engine change passed tests and independent review
before GPU validation. No new routing, queue priority or deadline policy was
introduced. Performance isolation itself still needs a loaded benchmark.

For the long trajectory, both fast and slow continuations report cached token
counts 4800/4928 at prompts 4852/4975; the uncached logical suffix is 52/47
tokens, not a full-history Prefill. Padding and GPU computation must be counted
separately in a throughput experiment.

Validated models: Qwen3.5-9B (TP1/2) and Qwen3-8B (TP1 regression). This does
not validate TP>2, speculative decoding, arbitrary hybrid layouts, SWE prompt
rewriting/stable-prefix mode, overload/retraction throughput, or a production
300+1200-second run. Keep `return_logprob=False` for this P2D Host implementation;
its inherited metadata limitation is not removed by this port.

## Reproduce compatibility checks

Do not install this tree into a shared environment. Launch it via the isolated
overlay; the installed dependencies still come from `pd`:

```bash
cd /homes/siqic/sglang-qwen35-integration
RUN_DIR=/tmp/q35-check-unique \
  PREFILL_GPUS=0,1 DECODE_GPUS=2,3 \
  PREFILL_GPU_GROUPS=0,1 DECODE_GPU_GROUPS=2,3 \
  PREFILL_TP_SIZE=2 DECODE_TP_SIZE=2 \
  bash validation/run_smoke_services.sh
```

Only use idle GPUs and fresh paths/ports. Defaults are Qwen3.5-9B, TP1,
page64, Mamba `extra_buffer`, exact-prefix mode, 8192 context, deterministic
generation, and 0.60 static memory on every GPU. `--long-output` crosses a
Decode checkpoint; `--archive-repeats 256` crosses a Prefill chunk.

```bash
PYTHONPATH=$PWD/python:/homes/siqic/slime/examples/pd \
  /homes/siqic/anaconda3/envs/pd/bin/python -B validation/check_multiturn.py \
  --run-dir /tmp/q35-check-unique --long-output --archive-repeats 256
```

`run_host_route_probe.py` is diagnostic-only: it temporarily pauses this
deployment's idle primary router, exposes another port, and forces a native
Host offer by reporting D unavailable. It must never target another agent's
router or a deployment carrying real traffic. Stop the probe before stopping
the servers; verify the primary router resumed. Shut down the exact inner
`validation/run_pd_servers.sh` controller, then verify its GPU children exit.

`check_multiturn.py` is a synthetic token-exact compatibility test, not a new
production agent harness. It disables logprobs and uses `/generate` output IDs.
Result JSON and logs are kept under `validation/results/` and gitignored.
