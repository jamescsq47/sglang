# Agentic PD modification gate

Before modifying files in this directory for the custom agentic PD pipeline,
read the complete design source of truth at:

`/homes/siqic/slime/examples/pd/docs/AGENTIC_PD_DESIGN_INVARIANTS.md`

All agentic P→D/D→P Direct, Shared Host Arena, TP coordination, allocator,
Radix ownership, and scheduler changes must preserve its eight acceptance
criteria. Map every change to an explicit request-generation ownership
transition and define success, failure, timeout, cancellation, and shutdown
behavior before editing.

Do not run a GPU experiment until relevant lifecycle/fault tests pass, all eight
criteria have been checked, and an independent audit returns GO. If context was
compacted, re-read the design document rather than reconstructing the intended
architecture from recent patches.
