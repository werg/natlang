# Shared local and CoreWeave RL configuration

Generate configuration from the actual full mixed checkpoint. The generator
inherits the parent's optimizer, precision and training options, together with
operational controls from the declared production stage; keeps a whole
epoch of selected data; and uses the same controller and learner for both models.
The recipe and stage must describe the actual selected checkpoint's production
path. It requires checkpoint NeuraleseProjection state and the existing direct trainer
schema. Family-member KD is currently unsupported in RL; use a declared full-model
continuation rather than silently omitting that objective.

Build the normal `ts-host` runtime first. On DGX, all builds and execution go
through `scripts/memory_ledger.py`; training/GPU units use high priority. On a
CoreWeave sandbox use its prepared Python environment and the same source commit.
Stage data, heads, pinned base, full checkpoint and required identity/monitor
sidecars through the existing artifact transfer mechanism before starting.

The following is the CLI shape for each model. Replace the paths with the
artifact locations on the host that will execute it; these are arguments, not
published ready-made checkpoints:

```bash
PYTHONPATH=training/neuralese .venv-neuralese/bin/python \
  -m natlang_neuralese.rl.production configure \
  --model lfm \
  --production-recipe training/neuralese/recipes/neuralese-mixed-lfm-qad-r11-ph-coreweave-20261015-v1.json \
  --production-stage direct_sft \
  --checkpoint /path/to/lfm-mixed/checkpoint.pt \
  --heads /path/to/lfm/canonical-raw-heads.pt \
  --base /path/to/pinned-lfm-qad-base \
  --cases runs/decision-rich-train-20261005-v2/cases.jsonl \
  --held-text-data runs/natlang-target-union-r11-full-ready-20261015-v1/full/lfm25/text-data.jsonl.zst \
  --runtime ts-host --out /path/to/lfm-rl-loop \
  --config /path/to/lfm-rl.json --port 8097

PYTHONPATH=training/neuralese .venv-neuralese/bin/python \
  -m natlang_neuralese.rl.production configure \
  --model mellum \
  --production-recipe training/neuralese/recipes/neuralese-mixed-mellum-r11-full-coreweave-20261015-v1.json \
  --production-stage direct_warmup \
  --checkpoint /path/to/mellum-mixed/checkpoint.pt \
  --heads /path/to/mellum/heads-raw-token.pt \
  --base /path/to/pinned-mellum-base \
  --cases runs/decision-rich-train-20261005-v2/cases.jsonl \
  --held-text-data runs/natlang-target-union-r11-full-ready-20261015-v1/full/mellum21/text-data.jsonl.zst \
  --runtime ts-host --out /path/to/mellum-rl-loop \
  --config /path/to/mellum-rl.json --port 8098
```

For example, start the generated LFM loop on DGX under an appropriate measured
memory claim:

```bash
python3 scripts/memory_ledger.py run \
  --unit neuralese-rl-lfm-production --priority high --budget-gb 24 \
  --env PYTHONPATH=training/neuralese -- \
  .venv-neuralese/bin/python -m natlang_neuralese.rl.production run \
  --config /path/to/lfm-rl.json
```

Choose Mellum's claim from measured full learner memory, including MuonSR state,
gradients and checkpoint/evaluation peaks; BF16 weight size alone is insufficient.
Do not run two unaccounted model copies on DGX. The managed actor stops before
learning and reloads the updated policy on the next round. On CoreWeave the
existing supervisor launches the same command inside its owned allocation.

The generated JSON is operational configuration, read again at round boundaries.
`rows_per_round`, `group`, `workers`, admitted `cases`, `teacher_rollouts` and
`trainer_parameters` can evolve with evidence and available data. Existing
harness controls are configurable. A finite `--rounds` invocation is useful for
validation; it does not declare production completion.

For a separate actor fleet, remove `actor_command`, set `endpoints` to reachable
replicas of the same exact policy, and supply actual supervisor argv arrays for
publication/reload. The controller requires an `after_update` publication command;
verify the requested checkpoint on every endpoint. Preserve actor stickiness for
live value state and archive complete transitive block bundles before releasing
actors. Credential handling belongs to the existing authenticated transport;
never put access tokens or expiring artifact URLs in tracked configuration.

State and round directories distinguish collected tasks from actually trained
tasks/tokens. An interrupted collection, incomplete learner checkpoint or failed
publication must retain its round and resume without losing data or counting an
update twice. Held task success, held CE, finite gradients and functioning
transport remain checked; base similarity is telemetry.

See [operations, research decisions and validation limits](../../../plans/neuralese/RL_OPERATIONS_20261015.md).
