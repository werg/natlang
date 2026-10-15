# Shared-state RL operations

The executable path is native actor → existing natlang collector/verifier →
exact sampled graph admission → shared `native_state_sft` full backbone update
→ existing VAE/joint/distillation refresh → new shared actor checkpoint and
bound semantic adapter. It replaces the old projected RL deployment path.
No new prompts are introduced: the existing native runtime is activated.

## Prepare each model configuration

After the two full native production recipes and initial shared semantic stage
have produced their actual export, run the following for each participant.
Use its exact participant name from `semantic-adapter.json`, the current
production recipe and its native stage, admitted whole-trajectory graph seed,
held graph file and train-only task packet. The adapter checkpoint must be the
same immutable shared parent for both learners.

```sh
python -m natlang_neuralese.rl.production configure \
  --native-state --model lfm --native-participant lfm \
  --checkpoint /shared/joint/checkpoint.pt \
  --semantic-adapter /shared/joint/semantic-adapter.json \
  --semantic-refresh-config /shared/semantic-training.json \
  --production-recipe training/neuralese/recipes/CURRENT-LFM-NATIVE.json \
  --production-stage native_full \
  --base /models/PINNED-QAD-LFM \
  --seed-native-graphs /data/lfm/train-native-graphs.jsonl \
  --held-native-graphs /data/lfm/test-native-graphs.jsonl \
  --cases /data/train-cases.jsonl --runtime /home/werg/natlang/ts-host \
  --out /runs/native-rl/lfm --config /runs/native-rl/lfm.json --port 8097
```

Repeat with `--model mellum`, its exact participant/base/recipe/graphs, distinct
output/config and port. The factory inherits actual native precision, shared
optimizer, FP8/checkpointing, task-evaluation cadence and numerical tolerances.
QAD LFM remains FP32; Mellum uses BF16 and the shared triton-expert memory path.
The factory does not create a new lineage or substitute weights. The actor
uses the exported tokenizer with its four role markers; captured actions are
never retokenized with the original unextended base tokenizer.

The paths above are deployment bindings, not claims that artifacts already
exist. The production recipe's `native_state_rl` handler can also admit a
sealed rollout file and continue the exact shared trainer directly. Its inputs
are `rollouts`, `held_native_graphs`, `continue_from`, optional `rl_block_bundle`;
parameters extend `native_state_sft` with `rl_tokenizer`, `rl_mode` and
`native_participant`.

## Run both learners and publish together

```sh
python -m natlang_neuralese.rl.native_rounds \
  --config /runs/native-rl/lfm.json --config /runs/native-rl/mellum.json \
  --out /runs/native-rl/joint --device cuda
```

Run DGX commands under its memory ledger with measured claims; do not launch a
second full validation suite. The same controller runs inside a qualified
sandbox or controls existing sandbox actors/learners through argv hooks. Edit
`actor_command` and `learner_command` to the qualified provider exec/transfer
bridge when controller and GPU reside on different machines. These commands
allocate no sandboxes. Persist model checkpoints, bundles and controller state
outside ephemeral process storage. Root/cloud supervision owns actual resource
placement and artifact transfer; an untested networking route is not ready.

The coordinator starts the two sealed-round jobs concurrently. Each collects
at one immutable policy, stops its managed actor after archiving all state,
and completes its full selected epoch with the shared optimizer. The two
participant updates retain the same parent proof. The shared refresh merges
both updates, preserving each learner's optimizer and current weights; it
retains learned VAE/optimizer state and reruns required VAE, joint/control and
behavioral phase exposure for every changed pair. Old semantic model state
never overwrites a newer learner. Only the new actual joint checkpoint binds
the atomically exported adapter. External `after_update` hooks receive the
whole deployment manifest and must refresh both actor bindings. Managed actors
load and attest it on their next lease.

An interrupted learner resumes its own rolling checkpoint and row cursor.
Changing graph packing or decay preserves moments and RNG. A publication retry
does not count or train the same stage twice. Adding tasks or data does not
freeze an empty selection. Infrastructure is unscored and repaired, never
negative training. No-spread failures use the actual admitted native graph
seed source; existing external teacher text is not replayed as student tokens.
Downloaded or collected data is counted separately from trained exposure.

This is parallel sealed synchronous RL with a shared refresh boundary. It does
not claim asynchronous actor/learner overlap or stale cross-checkpoint KV
interchange. KV, hybrid state and embedding identities are tied to actual
weights. Whole-trajectory graphs use one packed masked forward, not recursive
producer reruns. `rl/native_stage.py` contains admission only, not another
trainer. Ordinary native training remains useful while RL preparation runs.

## Validation scope

Focused tests cover actual native LFM shared-parent admission → full optimizer
update → checkpoint → unchanged resume; shared full-backbone/interface RFT and
GRPO optimizer updates on both canonical tiny ports; writer/reader one-forward
replay; real VAE/joint/distillation refresh after a changed participant; parallel
learner merge and idempotent publication. Fixture rewards do not establish
natlang task success. Full-size task quality and complete production exposure
remain actual run outcomes to measure after the required native stages finish.
