# Held quality after CE mass continuation

The original held `loss` stays the corpus-weighted token mean, exactly as before.
It is useful historical telemetry but can be dominated by long context. Native
SFT's optional `best-checkpoint.pt` was selected using this raw CE. The actual
native recipe result, continuation and publication use the newest full
`checkpoint.pt`, retaining optimizer/RNG/exposure. Native stage gates check
completed exposure and numerics; runtime/task gates require their real receipts.
They do not compare raw CE to a base or earlier score. Semantic completion checks
finite held loss and exact held coverage/source, not a raw-CE improvement.

With declared CE supervision active, native saves no longer nominate a best
export from raw CE. Existing historical best files/metrics are not relabeled or
used as a continuation. Newest full recovery state is still saved/published.
Task outcomes and separately measured target/declared CE inform the supervisor;
no new automatic metric threshold, rollback, early stop or qualification exists.

The CE policy can now carry a separate `held_sources` list, with exact held leaf
SHA, original encoding and coefficient. Extend a prospective immutable policy:

```
python scripts/neuralese_data/build_native_ce_supervision_policy.py \
  --source /actual/admitted/train-source-manifest.json \
  --held-source /actual/protected/test-source-manifest.json \
  --legacy-pair-context-weight 0.05 --out /new/immutable/ce-policy.json
```

These are command syntax examples, not staged deployment paths. The materializer
checks explicit train/test separation and source declarations are disjoint.
Existing policies without `held_sources` remain valid and report the additional
metrics as unmeasured. Already segmented trajectory sources keep their exact
weights and are explicitly retained, not guessed into a new target encoding.
Each actual consumed held row verifies its raw binding before decoding the
explicit suffix or exact witnessed-pair encoding. Malformed encodings fail.

Held `additional_ce` reports `declared` and `target`, computed from the SAME actual
graph and vocabulary readout as raw CE. `declared.loss` averages the normalized
original graph losses, matching native CE training's graph-level normalization;
`target.loss` uses only authenticated positive suffix targets. The separate
`token_weighted_loss` exposes cross-graph mass effects. Zero exclusions stay
zero. Original held `loss`, boundary metrics and per-class raw metrics retain
their original semantics. Reports include exact held source/policy bindings,
observed declaration coverage and counts; absent/uncovered sources are visibly
unmeasured. No held row receives training credit or enters a training iterator.

Semantic held evaluation must first exclude TRAIN-only CE declarations from its
raw objective. Otherwise a protected held SHA absent from train declarations
raises a source-coverage error. Explicit held declarations now additionally
measure producer control and adapted-consumer CE per model on their existing
forwards/readouts. They do not replace raw VAE/reconstruction/distillation loss,
change adaptation, or add extra graph forwards. All modes/RNG are restored.
No declared-target CE is invented during VAE-only phases with no CE supervision.

Focused qualification uses both canonical tiny ports, exact full-vocabulary
oracles, unequal graph masses, zero masks, partial chunks, original row bindings,
immutable train/held declaration tests, and genuine paired joint/distill forwards.
It verifies readout/forward counts, raw objective equality and no parameter,
gradient, RNG or exposure changes. This is local mechanism qualification; no
full-model quality result or cloud adoption is claimed.

Detached telemetry does not change intrinsic parser/index source files or the
seven-file index-policy identity. It uses the existing post-binding
`NativeGraphSource.set_ce_policy` seam: TEST examples dispatch only to
`held_sources` measurements, never TRAIN reweighting. A dataclass subclass carries
optional measurements while original semantic row bindings stay identical.
No reindex/rekey or million-row parse is required merely for this reporting.
