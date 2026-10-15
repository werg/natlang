# Adaptive Neuralese reinforcement learning

Owner objective: improve both models throughout the production recipe, exploiting
new data and fixes immediately. RL is a stage of that process, not a short probe
that counts as completion. One shared harness, reward, replay and learner serves
LFM and Mellum; only backbone loading, chat templates and resource sizes differ.

## Natural-language functions

`chooseTrainingWork(taskOutcomes, pendingWork, modelProgress) -> TrainingWork[]`:
1. Read named train tasks and their exact verifier outcomes.
2. Continue mixed-success groups with policy-gradient updates. For groups with no
   successful samples, request more samples or successful teacher trajectories;
   train successes through the existing RFT stage. Keep easy tasks available for
   regression checks while allocating useful work to harder tasks.
3. Incorporate newly admitted train tasks promptly; never collect held tasks for
   training. Decide whether a concrete defect needs a restart or continuation.
4. Keep rollout queues bounded by actual learner consumption and free memory.
   Adjust worker placement from observed rollout and update throughput.
5. Preserve useful state when changing recipes. Base KL is telemetry only.

`acceptRollout(batch, policyIdentity) -> AcceptedRollout[]`:
1. Require training split, finite checked rewards, and actual policy turns.
2. Exclude infrastructure failures from both reward and group normalization.
3. Keep the sampled reasoning, raw tool-call arguments and IDs. Do not replace
   the sampled action with the harness's interpreted/normalized action.
4. Bind behavior scores to policy version and sampling settings. A later policy
   score must never masquerade as the behavior probability.
5. Reject unknown or unreplayable behavior; recollect or send independently
   verified successes to RFT rather than inventing missing probabilities.

## Crisp mechanisms and reuse

- `task_rollouts`: exact train-split enforcement, collector queues and the shared
  `classify_case` reward. These are host boundaries and exact verifiers.
- `grpo`: group statistics, clipped importance weights and existing HTTP gradient
  replay/optimizer. Arithmetic and transport are crisp; no duplicate reward.
- `stage`: existing whole-trajectory RFT uses the shared direct trainer, optimizer,
  continuation checkpoints, task evaluation and data cache.
- RL production coordinator: explicit collect/score/update/persist transitions,
  bounded parallel subprocesses and atomic state are durability/plumbing. The
  work-selection policy above remains explicit, separately configurable.
- Full-policy objective: differentiable replay reuses `GradSession` and shared
  training infrastructure rather than a per-backbone trainer or copied harness.

No new prompt or runtime error is introduced by replay fixes. Research and
sampling interventions are not production qualification or measured model lift.
