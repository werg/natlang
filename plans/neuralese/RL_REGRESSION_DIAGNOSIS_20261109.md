# RL dev16 regression: problem statement and diagnostics so far (2026-11-09)

Brief for an independent in-depth review. Goal: find **any** bug in our RL implementation (collection, scoring,
objective, optimizer, checkpointing, evaluation) that could explain why RL updates make the model worse, including
causes we have not considered. Treat everything below as claims to verify, not settled facts.

## The symptom

Lineage `ops/bf95-lowlr-1105/w4` (data root `/home/werg/data/lfm3-rl-dgx-20261029/`), interleaved JOLT+ RL on LFM3-1.7B,
starting from the step-92 checkpoint (`ops/bf95-lowlr-1105/rollback-step92/checkpoint.pt`). Protected dev16 (16
single-sample cases, evaluated on Pop by `scripts/neuralese_data/pop_behavior_eval.py`; evidence under
`ops/pop-behavior-eval/evidence/<label>/detail.json`) declines monotonically with RL updates:

| checkpoint | what | dev16 task_success |
|---|---|---|
| step 67 (`parent-67-71c3eb45594d`) | SFT parent, re-evaluated on runtime v67 | 12/16 |
| step 92 | RL parent (rollback point) | 13/16 |
| step 97 (`raw-97-b90de36dd4e6`) | after w4 RL round 0 (5 updates) | 11/16 |
| step 102 (`raw-102-3d846629174f`) | after w4 RL round 1 | 9/16 |
| step 110 (`raw-110-2ba78f75e16e`) | after SFT window 000000 (8 updates) | 10/16 (16/16 cases scored, not yet marked complete) |

An earlier lineage from the same parent showed the same shape (steps 96/98/99: 12, 10–11, 11). With ±1.7 binomial
noise per point, any single step is ambiguous, but the trend is consistent across lineages and checkpoints.

Failure character (per-case comparison 92 → 97 → 102), important because it differs from the 11-05 regression:
- Failures are `semantic_mismatch`, **not** loops: 0 looping cases at 97/102 and 1 budget exhaustion at 110.
- At 102 the model runs **longer** than before (about 19 steps vs 11–12) and returns **confident wrong answers**:
  - `live_inventory`: reserves `[]` and reports a shortage;
  - `relational_multihop`: drops 3 of 8 items;
  - `route_replan`: quiesces as "blocked" (an honest-stop-shaped answer that is wrong).
- Lost cases stay lost at later checkpoints.

Training-round outcomes (`w4/round-00000N/score.json`, 128 student rollouts per round, 16 tasks x 8):

| round | task_success | loop rate | unresolved | groups with spread |
|---|---|---|---|---|
| round 0 | 57 | 0.12 | 22 | 8 |
| round 1 | 40 | 0.10 | 24 | 7 |

Round 1 also had 20 `infrastructure_error` rollouts, from the Pop actor OOM at 02:04, since fixed (ec68d4312).

## Setup facts relevant to review

- **Config:** `w4/config.json`, frozen runtime `runtime-native-corrected-mix-v67-1109` (code = main at 7fc570ec9).
  - `policy_loss`: `jolt-plus/1`, `beta_s 1.0`, `beta_t 0.5`; outcome `score-centered/1`, `importance: tis`, `k 128`.
  - Teacher = privileged JOLT+ context (OPSD-style), `jolt_teacher_plan crisp`.
- **Actors:**
  - DGX runs `serve_native_checkpoint.py --engine sglang --sglang-projection vendor`. That is a fast engine, so
    learner logprobs are recomputed and TIS-corrected.
  - Pop runs a second actor (`scripts/neuralese_data/pop_rl_actor.py`) at weight 0.4 with 4 workers. Each rollout
    records `policy.endpoint`.
  - The actor-vs-learner replay check covers one episode per round only.
- **Optimizer:** `shared-muon-sr-in-backward-v1`; backbone LR 1.56e-6, row LR 7.8e-5, weight decay 0.1.
  - Muon params use Kahan compensation (`muon_rounding: kahan`).
  - AdamW-routed params (norm gains, conv kernels, interface rows) still use **stochastic rounding** in BF16.
- **Checkpoints** (`natlang.neuralese-native-checkpoint/1`) contain:
  - `backbone_state`: 287 BF16 tensors;
  - `interface_state`;
  - `training.optimizer`: Muon momentum and compensation, plus AdamW state.
- **Code:**
  - `training/neuralese/natlang_neuralese/rl/`: `production.py`, `collection_generations.py`, `task_rollouts.py`,
    `native_stage.py`, `interleave.py`, plus the policy-loss code it calls.
  - `scripts/neuralese_data/pop_rl_actor.py`.

## Diagnostics so far

1. **Weight delta 97 → 102** (scratch scripts, not committed):
   - 2.57% of backbone elements changed, 86% of those by exactly 1 BF16 ulp; total delta L2 0.205.
   - The largest relative changes, about 1e-3, are in the `post_operator_norm` / `post_ffn_norm` weights. These are
     AdamW-routed, so they get stochastic rounding.
   - The update is tiny and mostly ulp-quantized. A behaviour change this large from it is surprising.
2. **Retracted hypothesis:** "stochastic rounding of the backbone is a random walk". Muon already uses Kahan (the
   10-29 fix). Stochastic rounding remains only on AdamW-routed params, which are among the most-moved tensors
   above, so it is not fully ruled out.
3. **Random-direction control (queued on Pop, label `raw-97rand`):**
   - The checkpoint is step 97 with the *real* 97→102 per-tensor delta values scattered to random positions (seed
     1109); delta norm 0.212 vs 0.205.
   - Path: `ops/diag-random-1109/checkpoint.pt`, sha 6ad98d7d9d75…
   - If it also drops to about 9, then the update's size or noise is the problem, not its direction. If it stays at
     about 11, training is pushing in a harmful direction.
4. **Direction test, complete** (`ops/diag-direction-1109/REPORT.md`; 78 episodes, 1.12M sampled tokens):
   - **Rebuild:** episode set, token totals, ordered-identities hash and deferral lists match the learner's receipts,
     and `rl-native-graphs.jsonl` is byte-identical. The per-episode advantages can't be checked, because the
     receipt doesn't store them.
   - **Δlogp (step 97 → 102) is the same for every advantage class:**
     - positive-advantage student episodes −0.00056 ± 0.00013;
     - negative-advantage −0.00054 ± 0.00011;
     - zero-advantage −0.00057 ± 0.00010;
     - teacher −0.00068.

     The positive-minus-negative gap is −2e-5 ± 1.7e-4, the advantage correlation is −0.045 (p 0.60), and its sign
     flips across the five update batches. **The outcome signal is not detectably present in the update.**
   - **The update is a common drift, independent of advantage:** rare sampled tokens gain probability (+0.055 for
     behaviour logp < −5) and mid-probability tokens lose it.
   - **The only directed movement is toward the privileged twin.** KL to the twin falls 1.9%, and the correlation
     with the twin gap is +0.11.
   - **Loss magnitudes:**
     - the twin-KL term is about 0.005–0.02, while the outcome term is 1e-5–1e-3;
     - on six probed episodes the outcome gradient was 35–60% of the KL gradient's norm and roughly orthogonal to it.

     So the JOLT+ teacher term dominates. Combined with the common drift, RL as configured is mostly "move toward
     the privileged-context policy plus noise". That fits the confident-wrong failures: a privileged-context policy
     answers without needing the information the student lacks.
   - **Actor agreement with the step-97 learner:**
     - the DGX native engine matches to 2e-8 mean (max 1.9e-6);
     - Pop and SGLang-sampled episodes differ by 0.0147 nats mean (max 0.58; 3.5% of tokens > 0.1). This is not
       biased beyond the k3 estimate, and the guard rejects none of them.

     Pop's per-token mismatch is the same size as the whole 5-update movement (mean |Δ| 0.0195).
   - **Caveats:**
     - the Δ spans five updates, which dilutes each one;
     - the 40 sampled decision actions were not measured;
     - graphs were run unpacked.

## Known defects (real, possibly contributing)

- **Budget-exhaustion loops are unscored** (`task_rollouts.py` ~595–618, `budget_prefix_capture_unavailable`).
  Loops never receive a negative signal. 13 of round 1's rollouts were affected.
- **`unresolved` (24) and `infrastructure_error` (20) rollouts are unscored.** Together with the item above, about
  45% of round 1 carries no reward, which skews which behaviours the centred advantage sees.
- **Over-long episodes are deferred out of the update.** The fix (40758e0da, truncation to the longest
  import-closed prefix) is built but not deployed. Long episodes are disproportionately the failing ones, so the
  learned signal is biased toward short episodes.
- **AdamW-routed params still use stochastic BF16 rounding** (see diagnostic 1).

## Suspects not yet tested (please extend this list)

1. **Wrong-signed or misaligned policy gradient.** Possible causes:
   - an advantage sign error;
   - group-centering applied across the wrong grouping;
   - TIS weights inverted (actor/learner ratio upside down);
   - a target mask shifted by one token;
   - advantages attached to the wrong episodes after shuffling, packing, or dropping unscored/deferred rows;
   - prefix aliasing (`native-prefix-physical-alias-v4`) mixing rows between episodes.
2. **JOLT+ teacher term (now the lead suspect, see diagnostic 4).** The privileged teacher may teach the student to answer confidently without the
   information. This would match the confident-wrong failures. Check its weight (`beta_t 0.5`) relative to the
   outcome term, and whether privileged facts leak into student targets.
3. **Reward asymmetries:**
   - honest "blocked" stops may earn credit, matching `route_replan` quiescing as blocked;
   - `execution_efficiency` cost terms may reward early wrong answers over long correct ones;
   - the `score-centered/1` math with `k 128`.
4. **Actor mismatch.** Possible sources:
   - the Pop actor (different machine, engine path and memory cap) sampling from different numerics;
   - recovered Pop blocks (`w4/recovered-pop-stores`) carrying wrong policy ids;
   - an SGLang `vendor` projection differing from the native learner on Neuralese spans.

   TIS clips this, but systematic bias would survive.
5. **Interleaved SFT/CE rows inside RL updates.** Check whether `ce_supervision_policy`, `continued_native_graphs`
   or `pending_focus` contribute loss during RL updates, and whether that mixture itself degrades dev16. Step 110
   (after the SFT window) did not recover.
6. **Checkpoint/optimizer state.** Check that resume restores Muon momentum and compensation for the right
   parameters (name/order mapping), and that the AdamW state for interface rows isn't misbound after the
   `interface_state` split.
7. **Task coverage.** RL tasks may be far from dev16's semantic cases, so pressure toward short episodes and
   confident answers transfers badly. In that case the cause is the data, not a bug.
8. **Evaluation drift.** Runtime v66 vs v67 python runtimes (`-py34fcf6…` vs `-py7bcae0…` labels). Steps 97 and 102
   were evaluated on the older runtime and the parent was re-run on the new one (12/16). Confirm the eval is
   deterministic and the comparisons are like-for-like.

## Planned next tests

- Find out why the outcome term is 10–1000x smaller than the twin-KL term. Is `score-centered/1` / `k 128`
  normalisation or token-mean averaging shrinking it? Is that scale intended?
- Explain the advantage-independent common drift: rare tokens up, mid-probability tokens down. Candidates are the
  twin KL, entropy-like effects of the k3 estimator, TIS weights, and the AdamW stochastic-rounding norm-gain noise.
- Random-direction control result (`raw-97rand`).
- An independent plain-PyTorch reference gradient on one round-1 batch, compared with the learner's gradient
  (cosine, sign per parameter group).
- A null pipeline round (LR 0, bit-identical weights). If dev16 still moves, the eval or checkpoint path is at fault.

Production (`bf95-lowlr-w4-r12`) is stopped until a cause is found. The resume point is step 92 or a corrected parent.

## Finding and fix in flight (2026-11-09 ~15:30)

**The privileged facts reveal the answer.** `rl/privileged_context.py:case_facts` puts into the twin's context:
- "The verified result of this program is …" (`expected_result`);
- the expected stop status;
- expected files and effects;
- reference solutions;
- minimum steps and decisive evidence.

The student OPSD term, `beta_s * KL(pi(.|s) || sg pi(.|s,c))` at every sampled position, therefore trains the student
toward a policy that *knows the answer*. The student learns to emit answers without first acquiring the information.
This is the classic privileged-information leakage of on-policy self-distillation, and it predicts exactly the observed
confident `semantic_mismatch` answers.

With the outcome term not detectably present in the update (diagnostic 4), this pull is the only directed component.

**Run w5** (`ops/bf95-lowlr-1105/w5`, control `interleave-v49/control-w5`):
- same as w4 but `policy_loss.beta_s = 0`, in both the config and the recipe's `rl_policy_loss`;
- restarted from step 92;
- runtime frozen from main at 2f027b77f, which includes the long-episode truncation 40758e0da.

The teacher term (`beta_t`) and the outcome term are unchanged.

**Prediction:** if leakage is the cause, w5 holds dev16 at about 12–13 over its first two rounds where w4 fell to 11
and 9. If w5 also declines, the cause is elsewhere: the common drift, noise, or an outcome-signal bug.

**Follow-up regardless of the result:** the outcome signal is too weak to see.
- About 45% of rollouts are unscored.
- Only 7–8 of 16 groups have outcome spread.
- The update is dominated by an advantage-independent drift.

Scoring budget-exhaustion and unresolved rollouts as failures, and checking the outcome term's normalisation, are the
next fixes. A redesigned student twin should get process hints only (attempt feedback, verified-failure warnings),
never answer facts.

**Update:** 7be84ffbd makes budget-exhausted episodes train as scored failures. In the w4 round-1 data all 14 were
affected:
- 8 were already captured;
- 4 lost their prefix to `decision_action_frame` turns being ignored;
- 2 lost it to an omitted oversized action record.

w5 was restarted onto runtime v69-1109, which includes this fix, during round 0 collection. So w5 tests "no student
OPSD term + loops penalised" together.
