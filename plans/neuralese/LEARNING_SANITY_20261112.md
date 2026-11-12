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

## Check 4: GRPO with a trivially learnable reward (setup)

Runs in GRPO's envelope right after its 3-step smoke (`ops/sanity-1112/check4.sh`, unit sanity-c4a, own child actor).
`grpo.py --synthetic-reward short-calls` replaces the task outcome with minus the mean sampled tokens per model call / 1000
(every group has spread). The task outcome is still logged as `task_reward_mean`.
`--probe-turns` logs the NLL of 4 fixed exp0 chains on the learner every step and on the actor after every sync.
Every step logs `bf16_changed_step_fraction` and `bf16_changed_vs_start_fraction`.
Plan: 8 steps at LR 1e-6. If neither the reward nor the probe has moved, resume at 1e-5.
Pass: `mean_sampled_per_call` falls and the reward rises within a few steps, the actor probe NLL tracks the learner probe
NLL across syncs, KL and rho stay sane.

## Checks 2, 3, 5

Pending GPU windows (an FP32-master AdamW fine-tune peaked at 39-54 GB in ft2's run; the DGX is GRPO's from 18:00).
