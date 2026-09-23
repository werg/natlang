# Staged training pipeline

The repository has one staged production recipe. It builds a source and native
curriculum, then trains one adapter through the general code, coding, and teacher
stages in sequence. The recipe does not launch baseline runs, ablations, or
comparison runs, and creating the recipe does not launch any work.

The base model defaults to `LiquidAI/LFM2.5-350M`. The teacher collector defaults
to model id `Ternary-Bonsai-2-27B` at `http://127.0.0.1:8081`. Both can be
configured. The recipe preserves the configured split registry across stages and
chains adapters: coding starts from the general checkpoint adapter, and teacher
starts from the coding checkpoint adapter. Every stage has its own optimizer and
resumable checkpoint. Intermediate merged models are skipped.

Student selection applies to every rendering and training stage: each uses the
selected model's tokenizer/chat template and the same base revision. Use a
separate recipe and run directory for each student; rendered token sequences
and adapters are not interchangeable between models. The pipeline is not tied
to the default model, but a particular architecture still needs compatible
Transformers/PEFT support and sufficient training hardware. Set quantization,
LoRA target modules, context length, batch sizing, and the free-VRAM gate for
that model using the recipe options; the default settings are not a promise
that every model fits an 8 GB GPU.

## Create and run a recipe

The workspace `.venv` does not contain PyTorch. Use the existing
`natlang-train-kernels` image for Python preparation, corpus rendering, and
training. The runner and recipe creator themselves use the system Python or
`.venv/bin/python`; the image is selected when the recipe is created. Before
running the pipeline, the existing `ts-host` build must include
`dist/native/runtime.js`; `freeze-runtime` snapshots that built tree and fails
if it is absent.

```sh
.venv/bin/python scripts/create_training_pipeline.py \
  --output runs/production-recipe.json \
  --image natlang-train-kernels

.venv/bin/python scripts/run_training_pipeline.py \
  runs/production-recipe.json runs/production-2026-09
```

Recipe generation accepts `--model` and optional `--revision` for the student
base, `--teacher-model` and `--teacher-server` for collection, and `--image` to
select the existing Python training image. `--init-adapter` starts the first
phase from an existing adapter. Repeat `--train-arg` to pass extra trainer
options; for example:

```sh
.venv/bin/python scripts/create_training_pipeline.py \
  --output runs/production-qlora-recipe.json \
  --image natlang-train-kernels \
  --model LiquidAI/LFM2.5-350M \
  --teacher-model Ternary-Bonsai-2-27B \
  --teacher-server http://127.0.0.1:8081 \
  --train-arg=--load-in-4bit \
  --train-arg=--rank --train-arg=32
```

The default trainer arguments are one epoch per stage, rank 32, accumulation
16, microbatch 1, an 8,192-token padded batch budget and maximum length, source
file order, skipped held-out loss evaluation, and no merged export. Trainer
options such as `--load-in-4bit`, `--unsloth`, learning rate, and maximum length
can be passed through `--train-arg`. The recipe generator rejects `--full` because
its later stages are explicitly chained as LoRA adapters.

Stages run in this order:

```mermaid
flowchart LR
  A[Acquire source shards] --> B[Assemble source bundle]
  B --> C[Freeze built runtime]
  C --> D[Observe eligible source cases]
  D --> E[Generate candidates and native replay]
  E --> F[Select teacher seed programs]
  F --> G[Freeze linked group splits and prepare corpora]
  G --> H[Render general] --> I[Train general]
  I --> J[Render coding] --> K[Train coding]
  K --> L[Collect teacher trajectories]
  L --> M[Materialize teacher turns]
  M --> N[Prepare with frozen splits]
  N --> O[Render teacher] --> P[Train teacher]
```

`freeze-runtime` snapshots the already-built `ts-host/dist`, `ts-host/scripts`,
`prelude.js`, and package metadata into `${run}/runtime-host`. Subsequent source
observation, replay, and teacher commands use that snapshot, so a concurrent
clean or build cannot remove or replace the runtime files they are using. Its
`node_modules` entry is a symlink to the source installation, so keep installed
Node dependencies unchanged for the duration of a run. The freezer records file
hashes and rejects a modified existing snapshot; after a snapshot exists, edits
to the live source tree do not silently replace that frozen revision.

`observe-source` executes only the audited pure subset of eligible source
functions in timeout-limited child processes. Its observations are not upstream
test labels and are not considered verified training cases by this step.
`synthetic` combines those observations with generated candidates and native
replay. `teacher-seeds` selects the program set for collection before the general
and coding phases. Logs are written under the run directory, one file per stage.

Use `--until` to stop after a named stage while preparing to inspect or schedule
the next part. For example:

```sh
.venv/bin/python scripts/run_training_pipeline.py \
  runs/production-recipe.json runs/production-2026-09 --until train-coding
```

Re-run the same command without `--until` to continue. Completed stages are
checked against their recorded input and output hashes and are skipped; failed
or stopped stages are retried. The pipeline config is content-identified. If a
recipe, source, or completed output changes, the runner refuses unsafe reuse;
create a new recipe/run directory for intentional content or configuration
changes.

## Stops and completion

The runner records stage intent before launch, starts each stage in its own
process group, and forwards SIGINT or SIGTERM to that group. The trainer handles
these signals at optimizer-step boundaries and writes a checkpoint; the
synthetic replay stage also checkpoints between replay cases. A stopped stage is
not marked complete, even if its child writes a checkpoint and exits with code
zero. Re-run the same recipe and run directory to resume it. A training stage is
complete only when its checkpoint state reports that `trained_examples` reached
the corpus's `target_examples`.

A forced kill (SIGKILL), host crash, or power loss cannot be checkpointed. In
that case, recovery is limited to the latest fully written trainer checkpoint
or replay cache entry. Do not treat a stopped run or a saved adapter as proof
that all stages completed.

Training stages have a GPU 0 availability gate of 2,048 MiB free by default.
The runner checks `nvidia-smi` before launching each training stage and returns
75 with a `resource_wait` state if the gate is not met. This is a resumable
resource wait: no training child is launched for that stage, and rerunning the
same command checks GPU 0 again. Configure the threshold when creating the recipe
with `--min-free-vram-mib`. The recipe does not stop or reconfigure the teacher
service; keep it managed as-is and resume when the configured GPU gate is
satisfied.

## Corpus evidence and splits

The preparation stage keeps linked source/program groups together and carries
explicit test or validation membership into the frozen split registry. Teacher
turns use that same registry, so linked teacher material cannot move an existing
test group into training or move a frozen training group into test.

Corpus rows carry different evidence levels. Downloaded source code and
`code-proposals.jsonl` are plain source proposal rehearsal: they can be admitted
as training targets while still unverified. They are not labeled as native
verified examples. Synthetic reference outputs and observed source results remain
candidates until native replay accepts them. Only accepted replay turns are
written to `verified-turns.jsonl`. Rejected source inventory is excluded. Teacher
program seeds are collected and materialized in later stages, then prepared
against the frozen group registry.

`prepare_training_stages.py` receives verified replay turns and source proposals
as separate files, while combining them for the coding curriculum. Provenance
fields continue to distinguish their evidence. The pipeline manifest and row
provenance preserve these distinctions; the recipe makes no claim that training
has completed or that the resulting model has passed behavioral evaluation.
