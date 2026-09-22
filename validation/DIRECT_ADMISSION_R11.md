# R11: completed Direct handoff must not reoccupy an I/O slot

Status: implementation and audit pending; no performance acceptance.

## Root cause and bounded correction

Socket TP admission counts active grants without a receive object. Native
handoff legitimately removes the completed receive, but its TP control record
remains until all ranks report phase5 and clear. That cleanup-only record must
not become an unmaterialized grant again. It delays fresh Direct admission
despite the previous snapshot no longer using a receive slot.

Use the existing `local_admitted` fact, established only after successful native
handoff, to exclude such records from socket TP grant slot accounting. Publish
that fact and remove the receive atomically under the existing poll lock.
Do not exclude merely failed, terminal or cancelled attempts: they still count
until existing physical-fence retirement. TP1 and non-socket paths unchanged.

No changes to physical allocator, Host recovery, deadlines, routing, capacity,
model/workload, cache configuration or release policy. Same R10 c128 setup.
Local Triton cache now contains R10 kernels, so compilation warmness is another
comparison caveat; record stage timings rather than attributing all throughput
changes to this accounting fix.

## Gate and eight invariants

1. Unique snapshot ownership: unchanged; admitted is an existing handoff fact.
2. P2D Direct release: unchanged.
3. P2D Host durable release: unchanged.
4. D2P Host durable/source release: unchanged.
5. Independent progress: cleanup metadata no longer blocks fresh receive slots;
   no additional polling or thread and no new synchronization boundary.
6. TP atomicity: native grant/BIND/COMMIT/ADMIT and all physical fences retained.
7. Parent reuse: no new recompute, eviction or transfer-length policy.
8. CPU regressions + independent GO precede any GPU run. Tests must cover actual
   native handoff while group status remains4, eventual clear, fresh admission,
   pending claim, reserved-but-not-started grant and unfenced cancellation.

R10 had many fallback snapshots with no P-start record, but it is not proven
that this bug explains all of them. GPU validation must separately measure
Direct success, failed-start cohorts and Slow post-copy handoff latency.

## Frozen gate and launch

Full `validation/check_multinode_cpu.sh`: **1332 passed, 2 GPU-only skipped**;
launcher/diagnostics **86 passed**. Raw log `/tmp/dualpd-r11-cpu-gate.log`.
Independent control_audit **GO**, with 74 Direct tests including real native
handoff-to-Req and subsequent TP clear. Existing 30 s stats add `slots_used`
and `admitted_cleanup`; no extra control RPCs or per-tick logging.

Starting `qwen35-122b-swe500-tp8-c128-socket-full-r11` with the same Bash
launcher and `--concurrency 128`. R10 owned cleanup verified on both nodes.
GPU correctness and performance outcomes remain pending.

## Interrupted GPU diagnostic

Direct/Slow two-turn token-equality smoke passed. Workload began at
07:20:41 UTC on 2026-09-21; verified coordinator PID 309931 received SIGTERM
at about 07:30 UTC. Cleanup completed, and both nodes' GPU compute-process
queries were empty. This is not a completed 300+1200-second benchmark.

Matching post-warmup one-minute arrival cohorts (R10 07:10--07:11,
R11 07:26--07:27) settled to:

| Outcome | R10 | R11 |
|---|---:|---:|
| Eligible fast arrivals | 143 | 159 |
| Direct group complete | 43 | 89 |
| Fallback without any rank start | 80 | 47 |
| Fallback after a rank start | 20 | 23 |

No pending outcomes remained in those cohorts. Fallback declined from 69.9%
to 44.0%; this is an observational comparison, not a controlled full-run
throughput claim. Successful arrival-to-start remained about 541 ms; failed
started attempts averaged about 1013 ms. Accounting cleanup helped, but the
Direct objective is not met.

Final 300-second P log window ended 07:30:25 UTC. Completed Slow cohorts had
copy-to-P-group-release mean 1.258 s/P90 2 s; all-eight-copy-to-all-release
mean 1.271 s. Copy-to-loaded ACK averaged 27 ms, loaded-to-bound-queue 442 ms,
bound-queue-to-ACK 157 ms, bound-ACK-to-release 684 ms. These P handoff events
must not be interpreted as the source Host extent's physical-free timestamp.

Stop reason: snapshot `ce2b18c1a6a748c181e477356aa1b762:6` had no P receiver
start on any rank. TP4 observed SLOW_FALLBACK at 07:26:14, and rank0 requested
abort; its Direct 8320-token lease remained active across ranks through
07:30. D had already completed D2H/source release at 07:26:15; P selected
Host recovery at 07:27:12 but did not start its copy. This exposes an
unfinished cancellation/retirement boundary, not a reason to omit failed
attempts from capacity accounting or release without a physical fence.
