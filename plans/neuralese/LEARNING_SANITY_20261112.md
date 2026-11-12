# Learning sanity checks (2026-11-12)

Question (owner): is the failure to improve LFM3-1.7B in the data/harness/tasks, or in the learning plumbing (trainer,
checkpoint write-back, export to SGLang, actor weight sync, GRPO math)? Cheap checks, each pass/fail with numbers.
Work dir: `/home/werg/data/lfm3-rl-dgx-20261029/ops/sanity-1112/`.

## Check 1: does what the trainer learned reach the served model? PASS

Fine-tuned checkpoint: Experiment 0's `ft2` (sha 980a9bf1, base 5ff07638 + 148 successful episodes, 3 epochs, LR 1e-5,
FP32-master AdamW). 24 of its own training chains (`exp0_sft.chains_of(turns.pkl)`, <= 12k tokens, seed 0; 37,616 sampled
target tokens) scored on their exact sampled tokens with `scripts/neuralese_data/sanity_logprob_paths.py`:

| path | base 5ff07638 NLL | ft2 980a9bf1 NLL |
|---|---|---|
| trainer (`exp0_sft.load_model` + `lfm3_fast_train.token_logprobs`, BF16) | 1.0261 | 0.7200 |
| SGLang actor (`serve_native_checkpoint --engine sglang --sglang-projection vendor`, FP8 KV, prefill logprobs) | 1.0277 | 0.7226 |

- The served ft2 shows the full trained drop (-0.305 nats/token against the base; trainer -0.306). Training-chain NLL
  0.975 -> 0.696 in the run's own log matches.
- Per-token trainer vs SGLang: mean signed +0.002/+0.003, mean |d| 0.044 (base) / 0.033 (ft2), p99 |d| 0.41 / 0.28. That is the
  FP8-KV + vendor-GEMM noise floor and is the same for both checkpoints, so nothing is dropped or mixed up in between.
- The base was scored on Experiment 1's live base actor (`exp1-actorbase`, same engine flags); ft2 on a fresh actor
  exporting into `sanity-1112/ft2-hf`.
- Weights: all 287 backbone tensors present; 40-83% of elements changed in the matrices, the interface rows unchanged
  (frozen by design). Some RMSNorm weights are bit-identical to the base: their accumulated FP32-master change stayed under half
  a BF16 ulp (2^-9 near 1.0).

Conclusion: checkpoint write-back, the native -> HF export and SGLang serving carry the SFT update exactly. Experiment 0's
flat task success is not a serving bug: the model memorised its chains (NLL -0.3) without that changing outcomes.

Note for GRPO (LR 1e-6, BF16 copies of FP32 masters for both learner forward and actor): on the base weights only 2.1% of
elements have half a BF16 ulp below 1e-6, 16.5% below 1e-5, 88% below 1e-4. An Adam step moves an element at most about LR,
so one step changes about 2% of the served weights, and the policy only moves once about 10-100 aligned steps have
accumulated. Not lost (the masters keep it), but delayed and quantised; check 4 measures it directly.

## Check 2-5

Pending GPU windows (an FP32-master AdamW fine-tune needs about 32-40 GB; the DGX is GRPO's from 18:00).
