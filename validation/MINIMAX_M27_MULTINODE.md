# MiniMax-M2.7 ordinary GQA adapter — 2026-09-17

No engine model/kernel/scheduler/allocator/lifecycle change is needed for this
checkpoint. Existing `models/minimax_m2.py:MiniMaxM2ForCausalLM` is selected by
native ModelRegistry. MiniMax has62 full Attention layers,48Q/8KV heads,dim128;
its MoE FFN introduces no additional persistent recurrent KV state.

Remote launcher in sibling Slime explicitly allows `model_family=minimax_m2`,
validates full Attention/head/expert/FP8 geometry, and uses TP8+EP8. Pure MoE
TP8 would split intermediate1536 into192, incompatible with the128 FP8 block;
EP8 keeps full experts and uses the native implementation, not custom padding.
Attention TP8 stays rank-to-rank. P and D must have identical TP/model/layout;
each TP/EP group stays within one host. No DP attention, cross-host EP group,
MTP/speculation, native HiCache or Mooncake in custom PD.

Existing MHA pool/Direct/Host adapters are unchanged. Hybrid Mamba/MLA remains
rejected: the paused Qwen3.6 investigation never modified production code.
CPU layout regression covers TP1/2/4/8, all62 layers,K/V and discontiguous pages;
it checks byte placement only, not GPU numerical output or RDMA functionality.

Ownership review:1 unique owner unchanged;2 Direct P release unchanged;
3 P Host durable release unchanged;4 D Host durable release unchanged;
5 no new runtime I/O or Forward dependency;6 unchanged TP group fences;
7 ordinary page-aligned KV semantics unchanged, actual reuse awaits remote tests;
8 CPU regression plus independent launcher audit precede remote GPU execution.
Failure/timeout/cancel/shutdown retain existing all-rank physical fences; model
validation rejects incompatible layouts before worker launch.

CPU probe of pinned model revision d494266a4affc0d2995ba1fa35c8481cbd84294b:
official config class loads in the current environment; RoPE theta5000000 and
rotary dimension64 recognized; tokenizer/harness multi-turn rendering agrees.
First remote evaluation: native colocated TP8/EP8,mem_fraction_static0.8,c64,
SWE-bench Verified500 once with inline verifier. Entry point:
`../slime/tools/dualpd/minimax_swe.sh`; instructions in `MINIMAX_M27.md` there.
This finite quality evaluation is NOT a300+1200s PD throughput acceptance.

Hardware/GPU/model-weight loading, multi-node Direct/Host correctness and
performance are NOT verified here. No H100 or local GPU run was performed.

Validation: existing protocol regression741 passed/2GPU skipped; new GQA CPU
layout4 passed; SWE harness39 passed; new launcher6 passed. Independent agent
audit GO for remote colocated validation; corrected failure cleanup to always
drain every owned supervisor even when one fails, without touching other jobs.
