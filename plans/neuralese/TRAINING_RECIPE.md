# Native Neuralese training recipe

Current training uses positional KV and hybrid-state spans. Neuralese tokens use
ordinary token embeddings with a learned low-rank embedding transform. Producers
export state at recorded positions; consumers import it with the matching
positions. The shared trainer runs teacher-forced graphs in one graph forward
pass and preserves gradients through producer and consumer dependencies. The
projected-value and recurrence training path is retired.

The executable recipe is `natlang_neuralese.train.recipe`. Current examples are
in `training/neuralese/recipes/native-state-production-*.json` and the shared
semantic curriculum recipe. Resolve a recipe by ID or path with:

```sh
python -m natlang_neuralese.train.recipe resolve native-state-production-mellum-20261015-v1
```

Execute a native recipe with explicit recipe and output paths:

```sh
python -m natlang_neuralese.train.recipe \
  --recipe training/neuralese/recipes/native-state-production-mellum-20261015-v1.json \
  --out /path/on/local-nvme/native-mellum-lineage --device cuda
```

The runner resolves and hashes every admitted input, freezes the exact Python
package, records stage plans and receipts, enforces native exposure/readiness
gates, and forwards stop signals to the active stage. A normal restart requires
the same recipe, input identities, device, and frozen runtime. `--plan-only`
validates the launch or handoff without writing. `--handoff` preserves the
lineage while accepting explicitly supported native operational changes and
source continuations; each native trainer verifies checkpoint, optimizer, RNG,
and exposure custody before continuing.

## Recipe and data rules

Recipes declare native stage kinds, dependencies, typed parameters, an exact
`identity_qualification`, and content-addressed input bindings. Native SFT uses
`native_graphs` and `held_native_graphs`; RL uses complete captured rollouts and
the held native graph set. The source binding retains split, producer and every
consumer occurrence, actor/checkpoint/runtime provenance, and the full
producer-to-consumer dependency graph. A continuation that appends train data
must bind the exact prior train and held manifests; the held set and all
previously trained rows remain fixed. New admitted data is reflected only after
verified exposure receipts, not on download or recipe registration.

Native epoch extension and supported backend, precision, or resource changes
use the same checkpoint continuation. The handoff records old and new values;
optimizer moments, step counters, RNG, and exposure stay under the trainer's
exact resume checks. Changes to the backbone, held data, or objective require a
new lineage or a specifically implemented continuation policy.

The shared recipe contains both model lines and the resumable joint semantic
stage. That stage becomes eligible when its native participants have completed
the actual producer/consumer foundation and passed its readiness checks. A
recipe declaration, gradient test, or stored guidance artifact alone does not
establish task capability. Measure task outcomes on distinct development and
confirmation invocations and retain the exact serving actor and checkpoint
provenance.

## Joint learned optimizer lane (after LFM main training)

Owner decision 2026-10-27: the joint learned optimizer (plans/neuralese/JOINT_LEARNED_OPTIMIZER.md) and the tunes of
its 51 supplied programs run after the LFM production recipe is done, on its final checkpoint. Everything is
prepared: the lane declaration, specs and runbook in `training/neuralese/learned-optimizer/` (README there),
persisted program splits in `/home/werg/data/program-supply-20261027`, and the operation directory
`/mnt/external/natlang-learned-optimizer-20261027`. It is a lane beside the recipe, not a stage in the production
recipe JSON: it trains no backbone weights and never advances production lineage exposure. Its order is case export
and teacher collection for the supplied programs, their program tunes, then `run-after-lfm-main.sh <final
checkpoint>` (bank, corpus of bank-basis fits and joint targets, outer training, meta-examples), then tuning the
optimizer program itself on exact hindsight/counterfactual labels of its own stages. Mellum follows with its own
exact checkpoint once its production recipe is done.

## Individual-program adaptation lane

The [tuning operations handoff](../TUNING_OPERATIONS_HANDOFF_20261026.md) is an
active companion to production training. Follow the actual workloads and
receipts in [program tuning operations](../PROGRAM_TUNING_RUNS_20261026.md):
grouped-world splits, teacher SFT of program adapters and native soft states,
whole-graph RL from the actual served release, development selection and an
untouched confirmation set. Keep program exposure separate from production
lineage exposure. Continue saved adapter and optimizer histories when adjusting
the recipe; use reply-only supervision when long repeated prompts overwhelm the
action targets. A recipe adjustment must not reuse a completed data-only rounds
journal and silently skip the new training.

Use the existing distillation export for verified training-world executions,
then ordinary source admission to feed both production recipes. Exported rows
are initially not admitted; development and confirmation worlds stay protected.
Neither saved adapters nor launched RL sampling establish capability lift.
Repeat useful program mechanisms on Mellum with its own exact checkpoint and
reader identity; incompatible LFM state is not a Mellum adapter.

## Precision and runtime choices

Native training owns its parameter masters and optimizer state. Backend and
precision changes are explicit native stage parameters, validated against the
restored model and hardware. FP8 execution is only enabled for the native
trainer's supported expert backend and exact precision recipe; its checkpoint
and receipts include the backend state required for continuation. A precision
setting or hardware profile is not a task-quality claim. Continue to evaluate
held loss, actual task outcomes, numerical behavior, and memory/runtime
constraints on the executing checkpoint.

See `NATIVE_STATE_SFT_DECOMPOSITION_20261015.md`,
`SHARED_STATE_MIGRATION_20261015.md`, and the native recipes for the detailed
trainer, readiness, and production-lineage contracts.
