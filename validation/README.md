# Optional engine diagnostics

This directory retains diagnostic scripts for payloads, digests, multi-turn
execution and Host-event progress. They are development helpers, not the public
experiment launch interface or a current-release acceptance suite. Inspect each
script's arguments and environment assumptions before use; legacy smoke helpers
do not replace the full benchmark measurement window.

For installation and supported experiment commands, use the
[DPD application](https://github.com/jamescsq47/DualPD#installation).
For engine component boundaries and validation limits, see
[engine boundaries and validation](../docs/DUALPD_CLEANUP.md).

Focused automated tests remain under
`python/sglang/srt/disaggregation/dualpd/tests/` and its `router/tests/`.
Historical development reports were removed from the published working tree;
they remain recoverable in Git history. Their old results are not acceptance
evidence for later code changes.
