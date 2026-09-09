# pd_mamba: Qwen3.5 agentic PD compatibility branch

Snapshot published 2026-09-09. This branch contains the SGLang engine changes
used by the isolated `pd_mamba` environment. It is separate from the `pd`,
`pd_baseline`, and node-specific development branches. Historical port notes
are in `AGENTIC_PD_MAMBA_PORT_STATUS.md`; they are chronological diagnostics,
not a statement that all later fixes were absent.

## Scope

- Composite Attention KV plus Mamba temporal/conv state snapshots.
- P-to-D active state and page checkpoint transfer; matching checkpoint
  recovery for multi-turn D-to-P reuse.
- NIXL Direct and Shared Host staging, including ownership, cancellation,
  physical completion fences, and source release.
- TP lifecycle tests and the existing compatibility scheduler/control plane.
- Optional stable prompt-prefix checkpoints for histories that remove old
  reasoning when reserialized. Default off.

This is the older compatibility implementation, not a merge of the concurrently
developed `pd` branch's latest capacity-based routing or DMA optimizations.
Validated launch configurations used 2s tool/2s Direct waits, file-backed Host
arenas (32 GiB D-to-P and 8 GiB P-to-D per P), and P-ready request cap 8 /
token cap 25% of the P KV pool.
Native HiCache/Mooncake is not the custom lifecycle's storage fallback.

## Pair with the external harness and launcher

The harness, datasets, router entrypoint, and experiment scripts remain in
[`jamescsq47/slime`](https://github.com/jamescsq47/slime), under `examples/pd`.
They are not copied into this SGLang repository. In particular, the matching
`launch_mamba_late_binding_router.py` entrypoint loads the compatible router;
do not substitute a newer `pd` router with a different ledger API.

Point the existing compatible environment/launcher at this checkout using
`SGLANG_OVERLAY_ROOT=/absolute/path/to/sglang/python` and
`PYTHONPATH=$SGLANG_OVERLAY_ROOT:...`, or install this checkout editable into a
dedicated compatible environment. This branch does not include a conda
environment, model weights, dataset downloads, or raw experiment logs.

## Snapshot policy

BrowseComp's native tool-role history uses the default full page-aligned
snapshot policy:

```bash
SGLANG_AGENTIC_KV_MAMBA_PROMPT_CHECKPOINT=false
```

The unchanged Miles SWE harness sends shell observations as user messages.
Qwen3.5's template can then remove prior reasoning, invalidating a complete
generation-end prefix. Enable the following consistently on P and D for that
case, together with the launcher-managed agentic lifecycle:

```bash
SGLANG_AGENTIC_KV_MAMBA_PROMPT_CHECKPOINT=true
```

P captures the real page-aligned state before its trailing thinking opener;
D continues normal Decode on the active state but preserves that checkpoint.
The return snapshot contains only the compatible prompt prefix. Untransferred
suffix recomputation is intentional and must not be described as full generated
tail reuse. Token digest checks remain mandatory; retraction invalidates a
frozen checkpoint. This mode rejects speculative decoding/lazy extra buffers.
Neither policy modifies the external harness or model input tokens.

## Validation and limitations

Focused checkpoint, Mamba, lifecycle, and TP suites: 370 tests passed. Run with
the compatible environment's Python and this checkout on PYTHONPATH:

```bash
python -m pytest -q -p no:cacheprovider \
  python/sglang/srt/disaggregation/test_agentic_prompt_checkpoint.py \
  python/sglang/srt/disaggregation/test_agentic_mamba.py \
  python/sglang/srt/disaggregation/test_agentic_kv_lifecycle.py \
  python/sglang/srt/disaggregation/test_agentic_tp.py
```

Qwen3.5-9B/TP1 SWE first-100 evaluation completed: 99 graded, one tool cleanup
failure, 29 resolved. All 12,575 Direct returns in the subsequent BrowseComp
c512 run were bound successfully; that run completed 300+1200 s, but exposed
Host eviction, a 600 s Decode-capacity timeout, and a high 2048-token turn-limit
termination rate. These are known capacity/harness limitations, not a claim
of an error-free serving system. State transfer was not byte-hashed for every
request in those formal runs. No full 500-task correctness claim is made.

Before further engine changes, retain snapshot ownership/fence invariants,
run relevant failure/cancellation/TP tests, obtain an independent audit, and
validate the actual transfer-to-bind path on GPUs. A completed DMA alone is
not proof that the following Prefill reused the snapshot.
