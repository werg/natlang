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

## Check 2: overfit one episode end to end: PASS

Data: 1 successful base episode of `ext:decisions:inbox-n4:inbox-sms-spam-train-109:078c21fd` from Experiment 1's collection
(guidance off; 12 calls, 6,874 target tokens; base pass rate 2/5).
Training: `exp0_sft.py train` from the base, 50 epochs, LR 2e-5, `--batch-loss-tokens 3000` (61 steps). NLL 0.96 -> 7.9e-5.
Eval: served with the held-v2 engine (SGLang vendor, FP8 KV), `exp1_collect.py` on that task.
`exp1_collect --temperature` was added for the near-greedy runs; `build_queue` refuses T=0.

- T=1: 8/8 task successes. Each episode has 7-12 of its 11-12 calls token-identical to the trained episode.
- T=1e-3: 2/2 successes, 12/12 calls identical.
- So the training format and the harness format agree: loss mask, chat template, tool-call parsing, the sampled reply's token
  boundary. Files: `sanity-1112/c2/` (sft23.sh, eval-t1, eval-greedy).

### Finding: the harness history does not replay the sampled tokens (16% of call pairs)

In Experiment 1's base successes, 201 of 1,281 consecutive call pairs of one invocation have a next prompt that does not extend
`prompt + sampled ids`. Breakdown:
- 86: the same text re-tokenized non-canonically, e.g. the sampled ` invoke` (19390) comes back as ` inv` + `oke` (1072, 4247).
  It also happened in the overfit episodes: identical sample, split tokens in the next prompt.
- 67: assistant text between `</think>` and `<tool_call>` dropped, or a second tool call of one reply dropped.
- 48: a difference inside the earlier prompt (history rewritten).

Training stays consistent: each call's exact prompt is used. But the model conditions on token splits it never produces and on a
history that differs from its own outputs, and learner chaining breaks there. Cause not traced: the history is presumably
re-encoded from streamed or normalized text. Owner: harness.

## Check 3: artificial SFT goal (marker phrase): PASS

Data: 200 calls from 88 tasks of Experiment 1's base successes (guidance off). Each sampled reply gets
"Zebra protocol engaged. " prepended (right after the prompt's `<think>`). Built by `sanity-1112/marker_chains.py`.
Training: from the base, 2 epochs, 11 steps at LR 2e-5; loss 0.83 -> 0.57.
Eval: held-v2 engine, `exp1_collect` on 24 tasks x 2 samples (16 tasks not in the marker set), at most 6 model calls each.
Measured with `sanity-1112/marker_rate.py`: does the policy turn's reasoning start with the marker?

- Base, same measure: 0/54 turns.
- Fine-tuned: 53/54 turns (98%); 11/11 episodes open with it. By task: 24/24 on tasks not in the marker set, 29/30 on trained
  tasks.
- Stopped at 11 finished episodes to free the GPU; the verdict was already clear.

So a supervised behaviour learned in `exp0_sft` reaches real harness episodes, and it generalises to unseen tasks.

## Check 5: task-level overfit (RFT on 4 tasks): partial lift, not ~100%

Data: 5 clean base successes (guidance off) from Experiment 1's collection, on 4 tasks with base pass rate 0.4-0.5:
design-brief-gates 008, decisions inbox-n7-119, codebase-ops select-leaf, auth line-support-desk.
Training: `exp0_sft` from the base, 8 epochs, LR 1e-5, 34 steps; loss 0.67 -> 0.09.
Eval: held-v2 engine, 8 samples per task at T=1. Three long line-support-desk episodes were cut at the window end and are
unscored.
Baseline: Experiment 1's base pass rates (4-5 samples per task, other seeds; no paired base run fitted the window).

| task | base (exp1) | fine-tuned |
|---|---|---|
| auth line-support-desk billing-flag-0001 | 2/4 | 5/6 |
| codebase-ops select-leaf length-3 | 2/5 | 7/8 |
| decisions inbox-n7 sms-spam-119 | 2/4 | 2/8 |
| design-brief-gates 008 reconstructNeeds | 2/4 | 4/8 |
| total | 8/17 (47%) | 18/30 (60%) |

- Moderate fine-tuning on own successes (loss 0.09) lifts two tasks and leaves two flat. Contrast check 2: driving one episode
  to NLL 1e-4 gives 8/8.
- Small n and unpaired, so no claim of lift. But a training loss near memorisation does not carry a task to ~100% at T=1:
  outcomes hinge on a few high-entropy decision tokens that a low average NLL does not pin down.
- So the plain SFT-on-successes signal is weak per task even when the plumbing is right. It is not a bug.

## Summary

The nuts and bolts work. Trainer, write-back, export, serving, weight sync and GRPO math all pass (checks 1-4).
Supervised behaviour reaches the harness and generalises (check 3); a fully overfit episode reproduces verbatim and succeeds
8/8 (check 2).
The failure to improve comes from data and objective:
- Experiment 0 trained 54% on off-distribution guidance-authoring calls.
- The harness rewrites the sampled history in 16% of call pairs.
- At GRPO's LR 1e-6 the BF16 policy barely moves: about 2% of elements in 4 steps, mostly flipping back. GRPO's real run now
  uses 3e-6.
- Moderate SFT on own successes lifts per-task success only partly (check 5).
