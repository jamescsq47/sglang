# Request-owned stable Mamba checkpoint (2026-09-10)

User-authorized specialization of the agentic design for the unchanged SWE
fenced-shell harness. All ordinary Qwen3/native behavior remains opt-out.

Enable lifecycle + custom storage + Mamba prompt checkpoint +
`SGLANG_AGENTIC_KV_MAMBA_REQUEST_OWNED=true`. Native HiCache/Mooncake are off.
The experimental Mamba:Attention memory ratio is **0.5**, not the 0.9 of the
previous collocated run. Static total memory fraction stays 0.80 on every GPU.

## Ownership and resources

* P_HBM_OWNED: preserve the longest page-aligned checkpoint before the thinking
  opener. P chunked Prefill retains native scratch. Insert and lock the new
  checkpoint before pruning obsolete unlocked Radix states; shared references
  and in-flight sender locks remain authoritative.
* P->D Direct/Host: unchanged full composite fence and native release. No new
  timeout, credit, reservation, retry or routing policy.
* D_HBM_OWNED: one active state plus one immutable imported checkpoint. No
  rotation while the generation owns this checkpoint. Attention sharing is
  unchanged. Return only the Attention prefix matching the checkpoint.
* D->P/Host: both components retain their existing fence and ownership protocol.
  Source state is freed by native request retirement after transfer success or
  Host durable. No unowned checkpoint is retained for speculative future reuse.
* Native Radix pruning frees only unlocked state. Internal nodes retain shared
  Attention; dead, unreferenced leaves use native KV cleanup. It is deliberately
  NOT triggered by generic dec_lock_ref (temporary COW unlocks are not handoff).
* D capacity estimate reserves three slots/request: one active, one private
  frozen checkpoint, and one native locked Radix copy of that same checkpoint.
  The Radix copy can be shared across requests but remains locked throughout
  Decode; it is not merely transient headroom. This version removes the second
  private ping-pong slot and unowned historical checkpoints, not every duplicate
  of the live checkpoint. P retains the native five-slot estimate for old/new checkpoint overlap
  and active plus ping-pong scratch. Physical allocation still checks availability.
* Native D retraction now backs up active AND frozen state alongside Attention,
  restoring the same checkpoint boundary before reuse. An invalid retracted
  checkpoint cannot resume tracking or be advertised as reusable.

The ordinary request-generation protocol still owns cancellation, timeout,
source/receiver DMA drain and TP rollback. A failed handoff never triggers raw
free. Final-answer/cancel paths retire snapshots through the same protocol.

## Correctness scope and acceptance

1. Unique owner: unchanged ledger/lease/CAS; reclaimed cache must be unlocked.
2-4. P2D Direct, P2D Host, D2P Host source releases retain existing DMA fences.
5. Normal forward/transport queues unchanged. Extra CPU copying is only on the
   pre-existing synchronous native retraction path, not ordinary delivery.
6. All TP ranks choose the same enabled mode; existing all-rank rollback fences
   apply to the reduced physical state slots too.
7. SWE serialization removes old reasoning. Reuse is of the agreed stable
   Prompt prefix, NOT the full generated tail. Report visible-response/tool/tail
   recomputation separately; the <=63-token tail claim applies only to alignment
   within an otherwise identical reusable prefix.
8. CPU lifecycle/fault/shared-prefix tests and independent GO precede GPU.

## Experiment

Canonical launcher:
`/homes/siqic/slime/examples/pd/scripts/new_method/run_qwen35_fused_swe500_2p6d.sh`.
Qwen3.5-9B TP1, P GPUs 0/4, D 1/2/3/5/6/7, c256, all 500 distinct Verified
instances once; unchanged external Miles/OpenEnv-local harness, temperature .6,
top_p .95, top_k20, min_p0, 8192 tokens/turn, 64 turns, context131072, page64.
Docker limits2CPU/4GiB, shell600s/verifier2400s. This is a finite evaluation, not
300+1200 closed-loop steady-state acceptance. Record accuracy, milestones,
latencies, tool times, exact/cached/theoretical Prefill, Decode, both pool
occupancies, Direct/Host/fallback counts and state ownership.

The fusion engine and original colocated baseline use different SGLang versions;
do not label the comparison a pure same-engine PD ablation. Do not modify either
shared installed environment. Current implementation/validation status must be
read from run artifacts; this design document does not claim a GPU pass.

## Resident-admission fix (2026-09-11)

Formal c256 exposed a TP1 P-side progress deadlock: early Direct binding had
already handed78 worksets to requests, reserving312/315 Mamba slots. They were
still behind16 older, unallocated metadata waiters. Re-sorting original arrival
times every tick defeated queue rotation; the bounded scan never admitted the
resident requests to native Prefill, so their state slots could never retire.

Only TP1 Mamba requests with `gate_complete`, `workset_backed`, reserved runtime
states and a `handed` lease now drain independently of NEW I/O scan/admission
caps. This is P_HBM_OWNED delivery to its existing compute consumer, not another
KV ownership transfer or allocation. Native Prefill scheduling, the existing
parent pin, suffix allocation, P2D fences and source release remain unchanged.
Cancels still use the original metadata/native queue cleanup and release_handed;
capacity-failed/in-flight/partially bound requests cannot take this path.
There is no change to Qwen3 admission, TP group decisions, ratio or timeouts.

All eight design criteria remain unchanged except improved progress (criterion5).
The old function fails both 78-behind16 fast/slow regressions; patched function
passes8 tests, including duplicate ticks, non-Mamba/partial-state rejection,
TP-local readiness exclusion and preserved NEW admission budget. Engine agentic
suite518 and native Mamba cache unit tests4 pass (522 total).

Extra old `slime/examples/pd/tests/test_mamba_*` fixtures target the separate
sglang-agentic-mamba API and produce8 compatibility failures under this fusion;
they are not silently treated as passing. CPU-only Mamba numerical kernels also
are not built into this CUDA environment (3 missing-operator failures). Neither
API nor CPU kernels are modified by this admission fix. This is not a claim
that every test in every checkout passes.
