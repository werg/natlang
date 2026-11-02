# Zero-SGD source adoption

This is a source and CE metadata correction, not a training update, task result,
or completed SFT stage. The operator must select the actual current earned full
parent and independently retain it. Its carried SFT frame must have the exact
prior source, held source, CE policy and zero exposure for each corrected row.
The successor recipe must declare the typed source correction, prior source,
corrected shared train/foundation source, and corrected CE family map.

Freeze a runtime containing the adoption implementation and run its commands
under `scripts/memory_ledger.py` with CPU-only settings and a suitable memory
budget. The output filesystem must retain at least 40 GiB beyond the full-state
write. Stage into a fresh directory:

```
PYTHONPATH=training/neuralese:. python scripts/neuralese_data/adopt_native_source_zero_sgd.py prepare \
  --parent /absolute/retained-current-full-checkpoint.pt --parent-sha256 EXACT_SHA256 \
  --prior-recipe /absolute/earned-source-recipe.json \
  --successor-recipe /absolute/corrected-source-recipe.json \
  --stage-id native_joint_sft_v4 --output /absolute/fresh-adoption-directory
```

The preparation reads and verifies the complete parent, applies the trainer's
source and CE admissions, writes a full checkpoint, then independently reloads
and compares every state and tensor byte. It writes `proof.json` only after that
check. Preserve both the full parent and this immutable child: central replay
rederives the correction from both, so neither is a disposable rolling slot.
The proof records zero optimizer updates and zero new visits.

After independently reviewing the proof and current controller ownership, bind
its exact SHA and publish under the controller lock:

```
PYTHONPATH=training/neuralese:. python scripts/neuralese_data/adopt_native_source_zero_sgd.py publish \
  --state /absolute/controller/state.json --proof /absolute/fresh-adoption-directory/proof.json \
  --proof-sha256 EXACT_PROOF_SHA256 --output /absolute/fresh-publication.json
```

Publication independently repeats the full-state derivation. It preserves the
controller phase and all earned counters, invalidates stale actor readiness,
and rejects an active update or a distinct recoverable partial checkpoint.
An SFT continuation uses a separately reserved rolling slot initialized from
the verified adopted checkpoint; do not overwrite either retained proof input.
Only after this publication may the 13,802-row append use the corrected source
as its actual predecessor. Recheck the current protected-world union and qualify
actual live runtime, capacity, data controls and parent before launching any
learner. Seven over-context additions remain pending under their separate proof.
