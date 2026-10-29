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

## Measurements (2026-10-29)

Build used for every number here: llama-server from the fork worktree
`dgxops/claude-recovery-20261028/data/llama-actor-worktree` at `b89457f32` (server fingerprint `b12005-b89457f32`,
binary sha256 `cb57d2b2…91b9`); GGUFs converted by the fork at `85a2e716` (the pin now moves to `a6401bbd3`).
Base GGUF `LFM3-1.7B-BF16.gguf` sha256 `39158a79…5e9f68`. Step 142: checkpoint `bf13a50b…`, GGUF `844707d2…`
(`lfm3-eval/gate-fresh-step142/gguf`). Step 608: checkpoint `b48d4d9c…`, GGUF `e17b5c89…`
(`lfm3-eval/decision/step608/gguf`). Harness: system prompt `8628b031…`, tool surface `695dfc9f…`.

- Determinism: `lfm3-eval/det/e1a`, `e1b` (`-np 8`, 8 cases twice): 7/7 identical first requests, different
  replies; 0/8 conversations identical; 7 vs 6 successes. `e2a`, `e2b` (`-np 1 --no-cache-prompt`): 65/65
  requests and replies identical, 8 and 8 successes, 649 s and 635 s wall. A frozen-runtime rerun of
  cases 40 and 97 under RL GPU load reproduced the same request counts and completion tokens.
- Throughput: single sequence ~60 tok/s on the idle GPU, ~8 tok/s per server with three servers next to the RL
  learner; batched 16 slots ×4 servers next to RL ~4 tok/s per slot (~250 tok/s total).
- Old-gate data, same harness, 38 cases, batched with the 3600 s budget: base 26/38, step 142 29/38, gained
  {16, 24, 104, 105, 120}, lost {9, 57}, McNemar p = 0.45. That is no significant difference; it is not the
  decision eval.
- The full 128-case run (step 142, step 608, base ×2) was stopped after 14 minutes by an owner decision that gave
  the DGX GPU to RL. No full arm completed. Partial outputs are in `lfm3-eval/decision/`.

## Rerun (one command; do it on a cloud sandbox GPU, not the shared DGX)

    PYTHONPATH=training/neuralese:. NATLANG_HASH_CACHE=$(mktemp -d)/h.json \
      .venv-neuralese/bin/python scripts/neuralese_data/lfm3_checkpoint_gate.py \
      <step142 export dir or .pt> <step608 export dir or .pt> --out <dir> --serving deterministic --repeat-base \
      --llama-server <fork build>/bin/llama-server --base-gguf <LFM3-1.7B-BF16.gguf> --eval-root <dir>/root

Use `--serving deterministic` on a dedicated GPU: four single-sequence servers run in parallel, and on a
datacenter GPU each should decode far faster than the GB10's 60 tok/s. With `--repeat-base` the base-vs-base flips
must be 0, which checks determinism on the full packet. Use `--serving batched` only when GPU time is short. Record the
llama-server build and converter revision with the result. Rebuilding with the `a6401bbd3` pin changes the
numerics, so it needs its own base run; the base key includes the binary hash, so this happens automatically.
