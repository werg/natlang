# Explicit native CE mass continuation

The admitted raw native ordinary/pair converter uses `.05` per context token
and `1` per assistant token. This is not a five-percent allocation to context.
The actual first-12 TRAIN-row audit found median assistant mass of 8.87%/9.44%
on ordinary FULL LFM/Mellum, 32.7%/34.2% on Qwen V5, and 14.6% on native paired
graphs. Existing segmented whole trajectories already allocate about 96% to
assistant targets. Evidence: `/home/werg/data/dgxops/native-supervision-masses-audit-20261020.json`,
SHA `c067c0891050e0a12135c59d92c238c5675346a3267d9cf44afa105a942e9960`.
These are small descriptive samples, not population percentiles or task-lift
evidence. The retained18006 gold support probe did not establish that this alone
caused the protected-task failures.

Historical ordinary sources can still contain unproductive assistant history at
their context coefficient. This repair preserves existing exclusion masks and
reduces total context mass; it does not invent additional failure masks. Any
new masking needs authenticated trajectory outcomes and a separate declared
continuation, keeping failed actions in the actual input history.

The new opt-in policy reuses `balanced_position_weights` with the exact suffix
and original zero exclusions. It rescales each node to its original total mass.
For coefficient `.05`, target mass is at least `1/1.05`, while every previously
positive context position remains positive. Graph/node multiplicity, causal
imports, forced markers, completed row credits and optimizer/RNG are preserved.
Already segmented trajectory weights remain bitwise unchanged. Whole-graph
GRPO likelihood/reward/advantage is a separate objective and is unchanged.

`natlang.native-ce-supervision-policy/1` binds every selected original source
SHA to its exact producer encoding and coefficient. Ordinary rows require their
authenticated `conversion.source_suffix_start`. Historical witnessed-pair rows
lack explicit suffix coordinates: the declaration explicitly selects their
original `{0, .05, 1}` encoding and verifies exact classes, a single suffix per
node and the witnessed producer/consumer provenance. This never infers targets
by approximate weight thresholds. A new eligible source absent from the policy
fails with an explicit extension-needed error. Do not disable that coverage
guard in production. Unsupported source encodings require a real provenance
conversion, not a guessed suffix.

## Prepare and deploy through the existing owner

1. Read the two **actual active recipe** training-source bindings. Use those
   exact source manifests, including newly admitted KEEP views and other ready
   additions. The materializer recursively verifies existing manifest bindings;
   it classifies source intent from its first row, and the shared trainer
   verifies the selected exact encoding on every consumed row. It does not
   admit data, fabricate completed exposure or claim a full-source scan.

   ```sh
   .venv-neuralese/bin/python scripts/neuralese_data/build_native_ce_supervision_policy.py \
     --source "$LFM_ACTIVE_TRAIN_MANIFEST" --source "$MELLUM_ACTIVE_TRAIN_MANIFEST" \
     --legacy-pair-context-weight 0.05 --out "$CE_POLICY_JSON"
   ```

2. Bind the returned policy path/SHA as the `ce_supervision_policy` input role
   in BOTH native SFT recipe continuations. The CLI receives
   `--ce-supervision-policy "$CE_POLICY_JSON"`. Use the existing verified
   full-state lineage handoff at an update boundary. Do not start a new lineage
   or add `objective_continuation` merely for this algebra change: original
   source bindings/schedules and completed row visits remain valid. Remaining
   scheduled visits use the declared corrected objective. Any decision to add
   explicit repeat exposure is a separate recipe policy, not hidden credit.

3. Shared semantic config uses the same bound object:
   `"ce_supervision_policy": {"path": "...", "sha256": "..."}`.
   Each streamed pair carries its authenticated container SHA outside its JSON
   content. Actual control/consumer CE and distillation use the same weights;
   phase gradient/capacity checks execute that same selected objective. VAE
   reconstruction/alignment and policy-RL terms are unchanged. Provider input
   closure transports the declaration as an ordinary immutable dependency.

4. Checkpoint options record the exact policy binding and transition history,
   with the preexisting global update, and preserve all optimizer moments/RNG
   and exposure. A later shared-trainer stage inherits the exact bound policy
   unless an explicit replacement is supplied; the original SHA is verified
   again. DPO/GRPO retain that future-SFT setting but bypass CE reweighting for
   their separate objectives. The full shared checkpoint's
   config history records semantic configuration changes through its existing
   mechanism.

5. Periodic/final **held CE retains original corpus weights** to keep historical
   held telemetry comparable; it is not the new training objective. Policy
   history records `held_ce=original-corpus-weights`. Actual protected task
   execution, numerical/runtime checks and genuine per-row readiness remain
   mandatory; no new task-quality certificate follows from this repair alone.

No frozen R14/v8 numerical package or active learner was edited during development.
Both cloud adoption and exact source/recipe bindings remain the cloud operator's
responsibility following independent source review. After adding new eligible
data, regenerate the policy and bind that explicit extension in the same
adaptive continuation. Keep all usable prior data and completed credits.
