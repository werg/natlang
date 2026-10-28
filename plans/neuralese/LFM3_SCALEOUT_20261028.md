# LFM3 scale-out: local first, then DiLoCo SFT on 4 sandboxes, then multi-machine RL (owner, 2026-10-28)

Owner 2026-10-28: "first iron out the kinks on this machine, then transition to DiLoCo SFT with 4 sandboxes, then
figure out a good pattern for swift RL, using several machines in concert where useful". This authorizes four
one-GPU learner sandboxes (plus the CPU DiLoCo coordinator) for the LFM3 SFT stage once the local phase passes.
CoreWeave credit remains opaque (AGENTS.md): bounded lifetimes, measured throughput, usage monitoring, private
full-state exports before long runs.

## Phase 1: local on DGX (now)

Exit criteria, each measured on the real path:
1. Port qualification (`LFM3_PORT_20261028.md`): HF-reference logits/gradients, one-pass graph vs cached execution,
   native serving vs llama.cpp on the same GGUF, native held-v3 reference.
2. Output-only objective (workstream E) declared in the LFM3 recipe; per-category loss telemetry visible.
3. LFM3 data twins for the first stage's sources; idiom audit decides the allocation.
4. A short local SFT run from LFM3 on the recipe's first stage: step time and memory at the real batch geometry,
   actual parameter updates (incl. FP32 sparse marker rows), save/restore/continue preserving optimizer/RNG/exposure,
   reply/action CE falling separately from reasoning CE, development-packet outcomes measured through native
   serving before and after. Only after this does anything go to the cloud.

## Phase 2: DiLoCo SFT on 4 sandboxes

Reuse `train/diloco` + `scripts/coreweave/diloco.py` (`DILOCO_COREWEAVE_20261027.md`; never yet run on CoreWeave).
LFM3-specific sizing to settle with the phase-1 numbers:
- Sync volume: 1.7B trainable FP32 = ~6.8 GB per full sync (~3.4 GB BF16 wire). Choose `local_steps` so a round's
  transfer and merge are a small fraction of its compute, using the existing fragment plan (streaming fragments
  overlap compute). If an adapter-first stage trains only adapters/new rows, sync is small; full-backbone stages
  need the measured choice.
- Learner memory: full FT with AdamW FP32 masters ~27 GB + activations on a 96 GB GPU; set the RAM minimum from the
  measured host footprint (not the 2 Gi default shape).
- Outer optimizer: short stages showed Nesterov overshoot on few rounds; start with averaging or low momentum and
  compare held/dev against the local run at equal exposure.
- Bring-up: 2-learner smoke for one or two rounds (identity, merges, export to a continuation checkpoint, restore on
  DGX), then 4 learners. Exports: coordinator root and learner-0 full state privately exported at a cadence; DGX
  keeps the verified recovery copy.
- Selection: DGX evaluates each exported global version on the development packet through native serving while the
  learners continue (DGX is the evaluator/selector, not a learner).

## Phase 3: multi-machine RL pattern (to settle after phase 2 measurements)

Default candidate: **DiLoCo-RL**. Each sandbox runs the existing native whole-graph RL round loop (its own actor
serving the current local policy, collector, verifier, GRPO learner) on its own stable task shard, and the DiLoCo
coordinator merges pseudo-gradients every H updates. Rollouts stay on-policy within a sandbox with no weight
broadcast over WAN, and it reuses both existing mechanisms. Needs: the RL learner calling `DilocoLearner.after_step`,
curriculum shards that keep outcome variance per shard, and a global curriculum view.

Alternative if learner-side compute dominates: asynchronous actor/learner (K actor sandboxes serving published
policy versions with captured behaviour log-probs and bounded staleness, one learner consuming groups continuously).
Costs a weight broadcast per version (small only for adapters) and importance correction.

Score centering (TRAINING_INSIGHTS_20261028.md section 5) changes the trade-off: it stays stable under quantized
samplers and, with truncated IS, under staleness of tens of steps. That makes the asynchronous actor/learner pattern
cheaper than assumed (actors need not wait for every version, and can serve the deployed NVFP4/FP8 policy), and it
also protects DiLoCo-RL against the drift between each sandbox's serving engine and its trainer.

Choice by measurement: rollout seconds vs update seconds per group, fraction of zero-variance groups, sync cost,
and measured trainer/sampler drift.
