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

## Automatic start after the shared curriculum

The existing `native_readiness` controller accepts `rl_bootstrap` as an exact
`{path, sha256}` binding to a `natlang.native-rl-bootstrap/1` plan. `rl_out`
defaults to its output directory plus `/rl`; `rl_device` defaults to `cuda`.
The controller launches RL only after the full shared completion verifier passes,
and forwards stop requests to the same supervised child. Inspection with
`--once` starts neither training nor RL. Failed shared qualification never
launches RL merely because a subprocess exited successfully.

The plan contains `models` with at least two distinct exported participants.
Each has `model` (`lfm` or `mellum`), optional exact `native_participant`
(otherwise selected unambiguously from the actual export's architecture),
`production_stage`, `base`, `runtime`, and `port`. The fields `cases`,
`production_recipe`, `seed_native_graphs`, and `held_native_graphs` are exact
`{path, sha256}` file bindings. Optional `semantic_refresh_config` is likewise
bound; its default is the actual qualified completion's curriculum. Optional
`commands` holds existing argv hooks, never shell strings. The implementation
computes real learner configs from the completed export, not future checkpoint
hashes. Restart verifies the declared artifacts and preserves useful edits to
already generated controller configs.

The existing admitted S7 task source is
`/home/werg/natlang/runs/decision-rich-train-20261005-v2/cases.jsonl`, SHA-256
`6f2a470a8de79d4fb9966fc5a5157ebdf16d3c58a02ae034322a2bdfcde20dca`
(72 train rows, registered source `clef-rich-source-20261005-v1`). Its actual
sandbox mirror must be staged and verified before use; declaring a remote path
is not a transfer. Native graph and full-pair curriculum bindings come from
the admitted production conversion manifests. With no useful selected tasks,
the continuous controller closes actor leases and checks for new work every
minute. This wait does not count as production completion.

## Colocated round bridge and graceful interruption

Run readiness and the round coordinator on the Mellum sandbox with the actual
shared assets resident. Provider account credentials remain on the DGX. Each
model plan may set `provider_queue` to
`/work/native-production/shared/bridge-ipc` and `provider` metadata identifying
its existing sandbox. Set the shared `refresh_command` to the same queue-client
argv and `refresh_provider` to the qualified Mellum sandbox metadata. The
bootstrap generates `round_command` automatically from `provider_queue`:

```
python -m natlang_neuralese.rl.native_bridge --queue /work/native-production/shared/bridge-ipc --request {request} --result {result} --stop-file {stop_file}
```

`native_bridge.py` defines the request/result protocol. Requests bind a unique
attempt, immutable configuration/state snapshots, actual parent checkpoint,
semantic adapter, participant and output namespace. The local-file queue writes
`requests/<request_id>.json` provider envelopes. A credentialed host bridge reads
these envelopes, allowlists the existing participant sandbox, stages all declared
assets and runs this exact worker remotely:

```
python -m natlang_neuralese.rl.native_bridge --request REQUEST_PATH --result RESULT_PATH
```

The worker reuses the shared learner and semantic refresh. The host bridge must
fetch every returned `artifacts` entry to the coordinator namespace, verify its
SHA/size, and publish the result JSON **last**. The coordinator independently
verifies the inventory, exact attempt, parent and participant update bindings.
Queue inspection uses `--queue PATH --list-queue`; no SDK or account credential
is required by the sandbox. The concrete provider worker remains a required
execution binding, not something this generic queue client supplies.

Stop requests name the exact request. The queue client stays alive while the
host forwards TERM to the exact worker and waits for checkpoint and artifact
drain. Local learner children likewise receive TERM and are awaited; parallel
rounds drain both learners before returning. An interrupted shared refresh saves
its optimizer state and never publishes an incomplete adapter. Resume uses the
returned durable state and checkpoint. Do not kill a coordinator while leaving
its remote children untracked.

Transfer actual full-state artifacts, not weight-only estimates: measured native
SFT checkpoints are 47,738,195,645 bytes for Mellum and 2,003,763,339 bytes for
QAD LFM. No production joint checkpoint existed when this protocol was added;
its size is measured from its eventual artifact inventory. Reuse immutable
parents and stream cloud-to-cloud rather than downloading full state to DGX.

Managed native actors use `--store-dir {out}/actor-store` inside each sealed
round. The existing TensorStore preserves exact native block bytes across actor
restart, so interrupted archival can fetch previously produced blocks. Reuse the
same store only for that round and immutable policy; later rounds have distinct
output directories. Existing custom actor argv must include this option before
using interrupted collection recovery. Do not remove the store while collection
or a pending archive still references it.

## Foundation-to-production deployment binding

`train.native_deployment` prepares the actual two-sandbox controller from the
current cloud release record. Run it on Mellum after the cloud owner mirrors
both portable recipes, launch receipts, declared input closures and pinned bases.
It verifies the release recipe SHA and foundation source identity; it does not
need or invent the future shared checkpoint or adapter. Use the **current**
release record after every source withdrawal/continuation, not a stale recipe.

```
python -m natlang_neuralese.train.native_deployment \
  --release /work/native-production/shared/deployment/live-release.json \
  --curriculum /work/native-production/runtime/training/neuralese/recipes/native-state-semantic-curriculum-rest27-foundation-20261015-v1.json \
  --out /work/native-production/shared/deployment \
  --cases /work/native-production/data/decision-rich-train-20261005-v2/cases.jsonl \
  --runtime /work/native-runtime/e4dd6aabd3b0 \
  --participant lfm=lfm2:f40d7a5bf651a03618ae8c06bda24a180e22fa79de964e55862c2c4984249da6 \
  --participant mellum=mellum:bc700045baee1c7ea1869675c38efedd59d9db09975ea0b38cff8db152d6148f
python -m natlang_neuralese.train.native_readiness --config /work/native-production/shared/deployment/controller.json
```

The cloud owner publishes the current release at this agreed path and stages
the curriculum before invocation; the builder rejects missing assets. The
`--runtime` argument names the repository root containing `ts-host`, whereas
the generated task runtime points at its `ts-host` directory. It writes `controller.json`,
`curriculum.json`, and `rl-bootstrap.json`, all using existing shared interfaces.
The generated controller waits for both complete **finite communication
foundations**, not all ordinary CE. Before transferring optimizer ownership it
also requires each original trainer's successful `control/exit.json` after the
exact hash-bound launch receipt. Mirror LFM's terminal receipt and all its bound
training/readiness/checkpoint artifacts into the same absolute namespace on
Mellum. A readiness JSON alone is insufficient. The old recipe supervisors
must remain terminal; restarting one would create a divergent optimizer branch.

Initial shared VAE/control/distillation and later shared refresh run on Mellum's
96GB GPU. The actual all-phase adaptive schedule publishes qualified progress
while retaining the entire cohort's remaining semantic obligations. CE frames,
row credits, optimizer, FP8 state and RNG survive the transition. Between RL
rounds `native_supervised.interleave` advances mandatory CE using the existing
recipe sources and checkpoint quantum, then merges both current participant
updates and refreshes the adapter before either actor uses changed weights.
New communication is raw-trained in both participants before joining semantic
adaptation; it is retained as mandatory pending work until then.

Participant CE and sealed actor/learner jobs run concurrently on their own
H100 LFM and RTX PRO 96GB Mellum sandboxes through the existing provider bridge.
The joint semantic trainer itself is single-device: the LFM GPU is not used by
that kernel during Mellum refresh. There is no distributed semantic-training
claim. Concurrent independent CE against the old parent would fork the model
and optimizer and is therefore not scheduled. Keep the adaptive refresh quantum
useful and bounded by its actual declared schedule; do not claim an unmeasured
full-size memory fit from CPU fixtures. The provider's actual round, stop/drain,
artifact ferry and full-size joint memory qualification remain deployment
requirements, independently of local file-binding tests.

Base relocation is explicit. Each release record may provide
`base: {path, inventory_sha256}`; otherwise its resolved recipe must declare the
actual base path. The builder compares the resident inventory with the exact
participant's inventory hash. It never guesses a namespace path. It writes the
verified alias into `curriculum.models[participant].base`, which the shared
checkpoint loader verifies against the checkpoint's original inventory. The
cloud owner has verified aliases `/work/native-production/models/lfm25-qad`
and `/work/native-production/models/mellum21` on both sandboxes and will include
them explicitly in the live release. A historical `/work/production/public-base`
path is not assumed to identify the same model on another sandbox.


## Existing reasoning-boundary row upgrade

Fresh native models learn sparse overrides for existing tokenizer `<think>` and
`</think>` IDs. No vocabulary, spelling, role-marker or prompt change occurs.
Old checkpoints retain their exact old sparse-row contract on ordinary load.
For an explicit adaptive upgrade, declare `learn_reasoning_boundaries: true`
(or use `--learn-reasoning-boundaries`) in the shared native trainer. This adds
separate reasoning input/readout rows initialized from the loaded pinned rows.
The readout adds a zero-initialized correction against the frozen original row,
so the upgrade initially preserves logits exactly even across GEMM shapes.

All sparse learned rows use FP32 master parameters while input/readout casts
retain the backbone activation dtype. At BF16 production LR 1e-5, independent
Mellum probes found nonzero reasoning-readout gradients but no changed row
elements after 16 updates. The explicit upgrade therefore also promotes old
BF16/FP16 sparse role rows to FP32 with exact numerical values and records that
narrow parameter-schema change. Existing optimizer tensors are mapped by
parameter name and preserved (old sparse moments are exactly upcast);
only the added groups have fresh history. Weights, RNG, row credits and pending
mandatory stages continue. The checkpoint records the sparse IDs, learned row
content and upgrade history. Native identity and portable composites cover the
new rows, including an explicit old-parent composite upgrade. Shared SFT,
semantic training, RL and serving use the same model/restore path. New actor and
semantic adapter bindings must reference the actual updated checkpoint; never
relabel an old adapter after changing weights. Both-port exact migration and
learning tests are mechanism evidence, not full-size task-quality claims.
