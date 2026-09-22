# R15: finish unbound cancellation on its existing I/O worker

Status: implementation frozen; full CPU gate and independent audit GO passed.
Runtime performance acceptance remains pending.
R14 is stopped, cleanup recorded, both GPU process lists empty.

## Evidence, not a bandwidth hypothesis

R14 final settled post-warmup cohort had 61/163 final Direct admissions.
Separate 09:16--09:17 cohort: 96 successes, 77 failures with no rank start,
37 failures after a start. Started failures spent 391 ms before intent versus
158 ms for successes; native-plan wait was 325 versus 248 ms; post-grant wait
149 versus 139 ms. Router arrival-to-publish averaged about 10--12 ms, not
the principal lost interval. These are diagnostic cohorts, not a full benchmark.

The real broker/admission/drop path reproduces transport occupancy 3 -> 4 -> 5:
a physically terminal receiver ceases to occupy a slot, a new intent is
accepted, then dropping the old receiver makes its still-active cancellation
context look like a never-started grant. No fifth physical DMA was started.
Example `ba17ec5e4b9046519718df94aa091620:5` drops at 09:17:31 and the following
stats report five slots; group abort completion follows at 09:17:32.

Four cancelled owners overlap around 09:16:09--10.4 while later arrivals
expire without any P Direct start. Thus retained cancellation metadata can
block transport admission even though destination memory is available.

Do not infer that cancellation explains every failure. The later settled
09:19:10--09:20:10 cohort had 61 successes and 39 started failures. All 61
successes have TP0 timing (pre-intent 95 ms, native plan 221 ms); 31 of those
39 failures have TP0 timing (pre-intent 159 ms, native plan 542 ms). Native
allocator-epoch delay remains material. The offline cohort helper now reports
these outcome-specific timings and their sample counts; failures without a
TP0 timing record are not assigned zero latency or excluded from success rate.

## Chosen minimal lifecycle change

Reuse the existing cancellation and full rollback receipts, not a new
transport namespace/FSM. A cancellation that has **never entered bind** can
complete its existing local rollback on the I/O worker:

1. Validate the current generation, wire attempt, owner and real lease identity.
2. Install the original retirement/start tombstone under the broker lock.
   `begin_bind` and `begin_io_attempt` use that same lock: if bind won first,
   leave rollback to the native scheduler.
3. Drain pending/unknown claim and metadata operations and the real transport
   fence. Close the receiver only after it cannot perform a later write.
4. Confirm no binding/handed/consumed ownership can have been entered, then
   publish the original local rollback completion. This must be a true full
   unbound rollback, not merely a local DMA completion masquerading as one.
5. Existing all-rank rollback confirmation returns transport admission credit
   without waiting for the final native clear. Keep the active cancellation
   owner and physical pages until the unchanged native retirement permits free.

Already-bound requests retain native Radix/Req rollback. Unknown control
results, partial DMA, disconnection, stale attempts and shutdown cannot be
interpreted as completion. Successful Direct transfer, Host recovery, model
parameters, deadlines, concurrency, I/O caps and recompute policy are unchanged.

## Required tests and eight-invariant audit

Use real broker/native/mailbox logic for TP2/8: occupancy rebound, all-unposted
cancel, posted follower, unknown claim, metadata callback, actual bind/cancel
lock competition, bound ownership, old attempt/new lease, late native command,
disconnect, and new allocation while old pages still await native retirement.
TP1 and legacy compatibility remain mandatory.

Unique ownership, TP atomicity and reuse must preserve the original exact
identity/fences; P2D Direct/Host and D2P Host source release are unchanged.
No added runtime file operations, threads, synchronous scheduler RPC or polling
protocol. Only unbound cancel work moves to the existing worker. Independent
actual-code GO plus the full CPU gate is required before another GPU run.

Runtime acceptance remains better settled Direct success and timely Slow
recovery without stalled ownership, not simply a passing smoke or fewer logs.

## Pre-run verification

Full CPU gate `/tmp/dualpd-r15-cpu-gate-final.log`: **1467 passed, 2 GPU-only
skipped**, plus **90 launcher/diagnostic tests**. Independent control audit:
**149 passed**, actual-code GO. Author's overlapping focused suite: 534 passed;
do not add these as unique tests. Both worktrees pass `git diff --check`.

The review-found ordinary-poll teardown race is covered: cancellation marks
strict receiver cleanup before yielding for in-flight DMA; failed `clear()`
retains the receive entry and cannot publish rollback completion. Bind/cancel
uses the same broker lock and exact owner/lease checks. Frozen late allocations
cannot reopen a superseded owner. Actual HBM retirement remains native-owned.

Frozen scheduler SHA256:
`205b46268453e58f51a252072537ef9cfac988239167d96a286f2f73ad29de7a`.
Before launch both a10/a11 compute-process lists and ports 23900--23905 were
empty. Next run: `qwen35-122b-swe500-tp8-c128-socket-full-r15`, unchanged R14
configuration. No CPU result is a claim of GPU performance or long-run safety.
