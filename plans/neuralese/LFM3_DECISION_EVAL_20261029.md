# LFM3 decision-grade behavior eval (2026-10-29)

Purpose: decide whether LFM3 SFT materially improves task behavior over the base (owner: if not, reset and go
straight to RL/self-distillation). Entry point: `scripts/neuralese_data/lfm3_checkpoint_gate.py`.

## Why the old gate was not decision-grade

Two executions of the same base GGUF gave 33/38 and 26/38. Root causes, each verified:

1. **Different harness.** `gate-base-v1` ran against an older `ts-host/dist` (system prompt `0522800a…`, tool surface
   `b7bda28d…`); the repeat and the step-142 run used the 14:18 rebuild (`8628b031…`, `695dfc9f…`). The 33 vs 26
   compares two different prompts; step 142 (29/38) had been gated against the old-prompt base. The first model
   request of a case already differed (5124 vs 5314 prompt tokens).
2. **Batch-dependent greedy decoding.** llama-server with `-np 8` and continuous batching: llama.cpp selects the BF16
   matmul kernel by the number of tokens in the batch (MMVF for 1 column, MMF for a few, cuBLAS beyond; see
   `ggml-cuda/mmvf.cu` `ggml_cuda_should_use_mmvf`), and prompt-cache reuse carries KV computed under other batch
   shapes. Same 8 cases, same harness, run twice: 7 of 7 byte-identical first requests (modulo random tool-call ids
   and per-run seeds, neither of which reaches the rendered prompt or greedy sampling) produced different replies;
   0 of 8 conversations stayed identical; successes 7 vs 6. Minimal repro (`-np 8`): one request alone repeatedly
   is stable until the server has served a concurrent batch; afterwards the same request gives a different reply.
3. **Wall-clock budget.** `case_seconds 3600` ended 3 base cases (88, 89, 1) as budget exhaustion after 108–143
   requests in the slower run, while the faster run finished 88/89 successfully: outcome depended on server speed.
4. A `ts-host/dist` rebuild during a run invalidated a step-142 execution (`eval.invalid-dist-rebuild`).

## Fix

- `-np 1 --no-cache-prompt`: one sequence at a time, every reply recomputed from scratch with batch-1 decode.
  Verified: same 8 cases twice → 65/65 requests identical, 65/65 replies identical; minimal repro stable; a later
  run under concurrent RL GPU load with the frozen runtime reproduced the same requests/tokens per case.
  Separate server processes (base and trained in parallel, training on the GPU) change speed, not numerics.
- Count budgets only: 64 model requests (successful cases have used at most 51), wall budget 48 h (never binding;
  a run where it binds is refused).
- Frozen TS runtime per run, identified by the digest of its complete inventory, identical for both arms; runs with
  mixed harness identities are refused; cached base runs are keyed by binary, GGUF, serving, flags, budgets,
  runtime, packet and cases.
- The export's serving settings (chat-template kwargs, producer-marker logit bias) apply to both arms.
- All 128 cases (16 shapes × 8 worlds).

Cost: single-stream decode is ~60 tok/s on the idle GB10 versus ~280 tok/s aggregate for 8 batched slots. With
the DGX RL learner on the GPU (21:30) three single-sequence servers got ~8 tok/s each; a 128-case run (≈1.3M
completion tokens for the base) would take well over a day. The gate therefore has two modes:

- `--serving deterministic`: exact replay, for an idle GPU.
- `--serving batched` (default): 16 slots, continuous batching, count budgets, frozen runtime; replies vary with
  load, so `--repeat-base` executes the base twice and the report gives the base-vs-base flips and both paired
  comparisons. McNemar stays valid under this noise (under H0 the discordant directions are exchangeable); the
  noise only costs power.

A batch-invariant llama.cpp mode (one per-column kernel path regardless of batch size, plus attention) would give
determinism at batched throughput; not implemented.

## Decision rule (recommended)

Paired flips on identical cases: b = base fails / trained succeeds, c = base succeeds / trained fails; exact
two-sided McNemar p on (b, c).

- **improved**: b > c, p < 0.05, and no shape loses net ≥ 3 of its 8 worlds → SFT is materially helping.
- **regressed**: c > b, p < 0.05.
- otherwise **no-significant-change** (or improved-with-shape-regressions).

Continue SFT only on **improved** against both base executions (batched mode); on no-significant-change or
regressed, reset and go to RL/self-distillation.

## Measurements

See the end of this file (filled in as runs complete).
