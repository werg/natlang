# Neuralese RL operations

Updated 2026-10-15. The objective is useful, comprehensive training of both models,
with new data and fixes incorporated as they become usable. RL extends the shared
production trainer and Natlang harness. A diagnostic update is not production
completion or evidence of model improvement.

## What runs where

Start with one controller per model and colocate its actor and learner on that
model's training sandbox. The controller starts the actual checkpoint server,
collects parallel harness trajectories, durably saves the complete block graph,
stops its actor, and runs the shared learner. The next collection loads the new
checkpoint and verifies its identity. This uses the two existing training GPUs
without requiring a third allocation. Ordinary production SFT remains useful
while the mixed checkpoints needed for value RL are prepared.

Use DGX for development, exact-reference checks, and local learner validation.
Harness orchestration is mostly CPU work and can run on DGX or an available CPU
sandbox. GPU model servers must remain close to their model state. The cloud
supervisor owns allocation, transfers and process lifetime; each local workload
runs through the memory ledger.

With additional GPU sandboxes, use a separate actor fleet for each model and
spread harness workers over its endpoints. Publish the same checkpoint to every
replica and verify its SHA before collecting. An external-fleet configuration
requires concrete supervisor commands for publication; a placeholder command is
not a deployment. Keep complete trajectories on their actor until all referenced
and transitive `.nzb` blocks have been copied and verified. Transfer bundles,
checkpoints and manifests through the already qualified private artifact path.
Independent sandboxes do not imply a shared filesystem, NCCL or RDMA connectivity.

The implemented controller uses synchronous rounds and parallel collection.
It does not overlap learning with collection of the next policy version. Increase
actor concurrency or replicas from measured valid trajectories per second, learner
utilization, memory and transfer time. The current native server does not inherit
vLLM or SGLang continuous batching simply because several HTTP workers use it.
Record model-load, collection, block transfer, learning and checkpoint time before
buying more replicas. Continuous actor/learner overlap is a subsequent extension
requiring explicit per-turn policy provenance and recovery of in-flight groups.

## Learning and data

- Collect only admitted train tasks with their exact Natlang task oracles. The
  protected held packets remain evaluation data.
- Use actual sampled token IDs, stopping tokens and behavior probabilities.
  The initial GRPO contract is unconstrained temperature 1. Unknown guided-mask
  laws or uncaptured legacy turns are not silently treated as on-policy samples.
- Mixed-success groups use token-level GRPO through `rl.full_policy` and the
  existing direct trainer, MuonSR, checkpoints and task evaluation. The objective
  uses asymmetric clipping. Base-relative KL and agreement remain telemetry.
- Flat failure groups need successful verified trajectories or continued declared
  SFT/mixed training. Flat success groups supply no policy-gradient signal; keep
  them available for regression checks as the policy changes. Infrastructure
  failures are unscored, never fabricated negative rewards.
- Text RFT uses the existing whole-context rendering. Every distinct selected turn
  must appear in the epoch; task-level balancing must not make turns alternatives.
  Value-bearing RFT preserves the producer graph rather than flattening blocks
  into text. When both kinds are selected, the controller runs complete text and
  value substages with checkpoint/optimizer continuation between them.
  Managed actors and text RFT explicitly use `full` prompt rendering. The earlier
  local 48-update LFM diagnostic accidentally trained `program-header` text and
  served `full`; its checkpoint/resume checks remain useful, but it does not
  validate matching actor/learner text prompts. The default and generated
  commands now agree, and data receipts record the actual training prompt source.
- Preserve checkpoint projections and optimizer groups. A direct checkpoint with
  no NeuraleseProjection first enters the declared shared mixed stage. The older
  standalone trajectories checkpoint schema is not a substitute for that path.
- LFM and Mellum share tasks, exact verifiers and useful text successes. Their
  latent widths differ: value graphs cannot be copied across widths as if they
  were interchangeable teacher text.

The registered `rl-verified-cross-model-seed-20261015-v1` contains four genuinely
successful Mellum trajectories on two train tasks. Corresponding LFM failures
supply selection-rate evidence only. This is RFT seed data, not clean-base
on-policy GRPO data or a measured improvement. Add newly verified successes
through normal corpus admission and update the controller configuration at a
round boundary.

## What we reuse from current RL systems

[Prime-RL's architecture](https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/docs/overview.md)
separates inference, CPU environment orchestration and learning, with a pull-based
rollout transport. We reuse those role boundaries and learner-driven queueing;
our first implementation persists bundles and rounds through existing durability
helpers. Its inference documentation also stresses preserving the processed
sampling law. We retain actual sampled probabilities and block coordinates.

[verl's async trainer](https://verl.readthedocs.io/en/latest/advance/v1_async_trainer.html)
describes colocated and separate actor pools, token-preserving partial rollouts,
version-aware replay and restoring in-flight groups. Those are useful next steps
when switching or idle time becomes the bottleneck. Current Natlang rounds do not
implement those async guarantees; importing a version-lag default would not make
that implementation exist.

[DAPO](https://dapo-sia.github.io/static/pdf/dapo_paper.pdf) motivates token-level
loss, asymmetric clipping and filtering groups without reward variation. These
fit the existing full-policy objective. Verified RFT or real mixed SFT supplies
useful work when the current tasks have no policy-gradient spread.

[slime](https://github.com/THUDM/slime) separates rollout generation from training
using SGLang and Megatron. We reuse the separation principle. Replacing the native
port with a generic rollout server would also require implementing our learned
value channel and producer-gradient replay there. That is a concrete integration
project, not a prerequisite for using the working native pipeline now.

## Validation and limits

Both actual full-size models have passed finite controlled full-model MuonSR
updates, real-width writer/reader replay with gradients into the writer, durable
block round trips, and wired Natlang collector fixtures. The fixtures force tool
choices to exercise the transport; their outcomes are not task-reward gains.
Ordinary student-root collection scored four LFM rollouts with zero successes,
and four Mellum rollouts with four successes. Those small samples identify useful
next work, not model-wide performance.

The Mellum wired fixture exposed unsupported Flash attention masking on cached
multi-query sliding-window reads. The shared fix uses offset-aware Flex masks;
CPU and GPU forward/gradient references and the actual Mellum fixture pass.
Parallel BF16 likelihood replay also differs slightly from single-token serving;
actual behavior probabilities are retained. Same-step replay is exact and the
corresponding FP32 comparison localized the difference to shape-dependent BF16
arithmetic. No replacement of old probabilities or artificial ratio-one claim is
permitted.

The earlier projected-channel RFT diagnostic on parent6257 was discarded after
the shared-state replacement. Its run artifacts have been removed; they provide
no qualification for current training. Track actual shared-state progress in
`SHARED_STATE_MIGRATION_20261015.md` and the live coordination status.
The controller also passed focused restart,
publication, policy-binding and new-task admission checks. The standing full-check service
validates integrated main; focused checks resolve concrete changes without
repeatedly occupying a second full-suite allocation.

Run commands and configuration generation are in
[the RL configuration guide](../../training/neuralese/rl-configs/README.md).
