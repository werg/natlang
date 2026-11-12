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

## Finding: Experiment 0's training data was half guidance-authoring calls

Decoding every invocation's prompt in `exp0-positive-control/turns.pkl` (the 148 w5 successes ft2 trained on):
319 of 551 invocations are `selectGuidanceFacts(consumer, facts: GuidanceFact[], budgetTokens)` calls.
They hold 1.17M of the 2.19M sampled target tokens (54%), and every episode has them.
Their facts carry the privileged answer ("The verified result of this program is no.", "A verified reference solution:
1. return_result {...}").
Their samples are long `<think>` passages that probe with `read_code` (rejected: no-such-function) and with
`eval console.log(JSON.stringify(facts.slice(...)))`.
These are w5's memetic/JOLT+ guidance calls. They never occur in the eval harness (sampling config `guidance: false`).
So more than half of exp0's SFT signal was off-distribution and favoured probing loops. That fits flat training-task
success, a slightly lower held score and loop rate 0.03 -> 0.11.
GRPO and Experiment 1 collect with `guidance: false`.
Checks 2/3 use Experiment 1's base successes (guidance off): `sanity-1112/c2/exp1-success.pkl`, 136 episodes, 1,465 calls.

## Check 4: GRPO with a trivially learnable reward: PASS (at LR 1e-5; at 1e-6 nothing moves in 4-5 steps)

Setup: `ops/sanity-1112/check4.sh` (`grpo.py --synthetic-reward short-calls`), own child actor (held-v2 engine).
The reward is minus the mean sampled tokens per call / 1000; an episode with no lowerable call gets -10.
Settings: 32 slots, 3-4 groups of 8 per step, `--max-model-requests 4`, KL beta 0.02, rho cap 2.
Probe: 4 fixed exp0 chains, 5,739 tokens.
Phase a: LR 1e-6, 5 steps. Phase b: `--resume` from a's step-4 FP32 state at 1e-5, steps 5-10. 19:06-19:58.
Code: 7114f1540 (a), 828f7ab1b (b).

| step | LR | mean sampled tok/call | reward | BF16 changed: step / vs start | KL | rho mean (pre-cap max) | abs log rho | probe NLL learner / actor after sync |
|---|---|---|---|---|---|---|---|---|
| 0 | - | - | - | - | - | - | - | 0.8806 / 0.8838 |
| 1 | 5e-7 | 990 | -1.00 | 0.88% / 0.88% | 0.0 | 0.9995 (2.5) | 0.028 | 0.8805 / 0.8852 |
| 2 | 1e-6 | 1226 | -1.24 | 1.15% / 1.24% | 5.4e-4 | 0.9995 (4.9) | 0.036 | 0.8804 / 0.8839 |
| 3 | 1e-6 | 1186 | -1.30 | 1.27% / 1.91% | 4.2e-4 | 0.9998 (3.9) | 0.029 | 0.8809 / 0.8837 |
| 4 | 1e-6 | 612 | -1.17 | 1.04% / 2.18% | 5.9e-4 | 1.0005 (2.4) | 0.038 | 0.8807 / 0.8821 (idle: 0.8824) |
| 5 | 1e-5 | 1446 | -1.31 | 6.8% / 6.9% | 5.4e-4 | 1.0003 (3.0) | 0.032 | 0.8799 / 0.8824 |
| 6 | 1e-5 | 546 | -0.68 | 8.1% / 11.2% | 0.0061 | 0.9999 (4.3) | 0.053 | 0.8952 / 0.8981 |
| 7 | 1e-5 | 562 | -0.90 | 8.1% / 14.6% | 0.035 | 0.9995 (6.6) | 0.061 | 0.9205 / 0.9199 |
| 8 | 1e-5 | 284 | -0.63 | 8.2% / 17.7% | 0.085 | 0.9984 (4.1) | 0.048 | 0.9628 / 0.9670 |
| 9 | 1e-5 | 94 | -0.93 | 8.1% / 20.7% | 0.051 | 1.0000 (3.2) | 0.058 | 1.0153 / 1.0176 |
| 10 | 1e-5 | 122 | -0.97 | 7.7% / 23.0% | 0.59 | 0.997 (9.8) | 0.108 | 1.0254 / - |

Phase a's step 5 was not kept: b resumed from the step-4 state. The step 5 above is phase b's.

- **The GRPO math learns.** Sampled tokens per call fell from 600-1450 (steps 1-5) to 546, 562, 284, 94, 122 within 5 steps
  at 1e-5. Advantage sign and direction are right.
- **Reward mean is noisy.** It mixes in -10 for episodes with no lowerable call, and different tasks per step. The length is the
  clean signal.
- **Weight sync works.** From step 6 on the actor's probe NLL tracks the learner's within 0.001-0.004 at every sync, through a
  +0.14 nat drift.
- **Rho is healthy.** Mean 0.998-1.0005; |log rho| 0.03 rising to 0.11 only as the policy moves fast.
- **KL behaves as expected.** It grows with the drift (5e-4 -> 0.59 by step 10: the run was collapsing towards empty replies).
- **At LR 1e-6 the policy barely moves.** About 1% of BF16 elements flip per step, but most flip back: only 2.2% differ from the
  start after 4 steps. The learner probe moved +0.0001; the idle actor moved -0.0015, which is within the FP8-KV noise.
  At 1e-5, 8% flip per step and accumulate (23% after 6 steps).
  For GRPO's real run at 1e-6: a policy signal needs on the order of tens of consistent steps to reach the served weights.
- **KL floor.** At step 1 (identical weights) KL was exactly 0 here, but 5e-4 in grpo-v1-smoke (longer chains). The KL floor
  of about 5e-4 at 1e-6 is numerical (reference pass no-grad vs policy pass with grad/recompute), not drift.
- **Learner speed** next to a 32-slot actor: 1.7-3.1k fwd tok/s, against about 600 next to a 48-64-slot actor.
- **Actor probe noise.** Scored during collection, the actor probe varies by about +-0.0015 (FP8 KV, batching). The idle actor
  is deterministic: 0.88384 twice on the start weights.
- **Export sanity.** The run's SGLang export equals the base weights bit for bit (all learner tensors; only the producer table
  is extra), so in-place sync writes into the right layout.

## Checks 2, 3, 5

Pending GPU windows (an FP32-master AdamW fine-tune peaked at 39-54 GB in ft2's run; the DGX is GRPO's from 18:00).
