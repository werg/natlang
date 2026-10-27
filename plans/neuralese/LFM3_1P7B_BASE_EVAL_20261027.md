# LFM3-1.7B base on the held-v3 cadence subset (2026-10-27)

Owner request: quick evaluation of the newly released `LiquidAI/LFM3-1.7B` on natlang tasks.

- Model: `LiquidAI/LFM3-1.7B-GGUF@b3f2ae5d73cd3c91bf40e8ba752c648aa05dffdf`, `LFM3-1.7B-BF16.gguf`
  (sha256 `39158a79...5e9f68`). Architecture `lfm3`: conv + full/sliding(256) attention, partial RoPE on full
  attention, per-head sigmoid output gate, sandwich norm, 128k vocab, reasoning on by default, XML tool calls.
- Serving: upstream llama.cpp `d05706b` (has LFM3 model + template support; our fork and the vLLM image do not),
  `llama-server --jinja -np 16 -c 1048576`, server defaults (reasoning preserved across turns). Smoke: XML tool
  calls parse into OpenAI `tool_calls`, reasoning into `reasoning_content`; ~250 tok/s single stream.
- Harness: `scripts/held_task_budget_probe.py --spec-from training/neuralese/recipes/neuralese-direct-v2.json`
  (packet `natlang-held-task-packet-v3-20261012`, 16-case cadence subset), greedy, 7200 s / 1000 requests / 1000
  turns, 8 workers; the same path as the LFM2.5-2.6B GGUF reference in DECISIONS.md "Task-eval budgets".
- Placement: one-GPU CoreWeave sandbox `dfa72ee1` (RTX PRO 6000, 2 CPUs, 40 GiB), because DGX had 2.7 GiB free
  beside production training. Stopped after collection. Outputs: `runs/lfm3-1.7b-heldv3-20261027/`.

**Result: 13/16 task successes** (2 semantic_mismatch: case 0 workbench_tickets, case 16
relational_multihop_qualifier; case 5 edit_stream unresolved because the current runtime refuses it as a
quarantined curriculum case, `source_review_pending`, with zero model requests: 13/15 scorable). No budget
exhaustion, invalid action or repetition. Probe wall time 363 s; slowest task 350 s (case 32, 127 requests).

Comparison on the same subset: LFM2.5-350M base 0/16, LFM2.5-2.6B GGUF 5/16 (unbounded budgets), Mellum2.1 base
11-13/16. Base-model task outcomes only; no training or Neuralese capability is implied.

Native adaptation not yet done: the shared native port (`model/lfm2_port.py`, serving, GGUF export, tokenizer/role
markers) does not cover sliding-window KV splicing, partial RoPE, the attention output gate, sandwich norm or the
LFM3 tokenizer/tool format.
