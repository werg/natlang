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
4. **Direction test, in progress** (`ops/diag-direction-1109/`, `analysis.json`):
   - Method: rebuild the round-1 objective and measure per-token Δlogp (97 → 102) against advantage, split into
     student and teacher tokens.
   - Interim, **10 of 78 episodes** measured:
     - Positive-advantage student tokens moved *down* (episode-mean Δlogp −0.0005 ± 0.0005).
     - Teacher tokens also moved down (−0.0007).
     - Correlation between advantage and Δlogp is −0.16 (n=7, p≈0.7).
   - No negative-advantage episodes are in the sample yet.
   - Queue check: all 78 episodes have an endpoint record and none mismatch.
   - Not significant yet, but **the sign is the opposite of what the objective should do**. A sign error, an
     advantage/reward misalignment, a token-mask misalignment or a logprob-to-token offset would all look like this.

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
2. **JOLT+ teacher term.** The privileged teacher may teach the student to answer confidently without the
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

- Finish the direction test on all 78 episodes, including negative-advantage episodes and the Pop vs DGX endpoint
  split.
- Random-direction control result (`raw-97rand`).
- An independent plain-PyTorch reference gradient on one round-1 batch, compared with the learner's gradient
  (cosine, sign per parameter group).
- A null pipeline round (LR 0, bit-identical weights). If dev16 still moves, the eval or checkpoint path is at fault.

Production (`bf95-lowlr-w4-r12`) is stopped until a cause is found. The resume point is step 92 or a corrected parent.
