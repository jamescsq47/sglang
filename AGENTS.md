# SGLang DualPD extraction

The user explicitly requested this standalone checkout. The original
sglang-src/pd checkout remains the comparison source; do not modify it.
Read the companion DualPD AGENTS.md before changes to request-generation
lifecycle or transport semantics: ../DualPD/AGENTS.md in the sibling workspace
layout, or ../../AGENTS.md when this checkout is the 3rdparty/sglang submodule.
Public imports remain
sglang.srt.disaggregation.agentic_* even when sources live under dualpd/.
Preserve native baseline behavior and all eight ownership/fault invariants.
Run lifecycle, capacity, cancellation and TP tests and obtain an independent
audit GO before GPU validation. Record GPU verification separately from CPU
tests and source-equivalence checks.
