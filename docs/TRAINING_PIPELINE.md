# Staged training pipeline

The repository has one staged production recipe. It builds a source and native
curriculum, then trains one adapter through the general code, coding, and joint
stages in sequence. The joint stage contains admitted teacher decisions from
every discovered curriculum track and bounded general/coding rehearsal. The
recipe does not launch baseline runs, ablations, or
comparison runs, and creating the recipe does not launch any work.

For verified student-failure corrections after that initial curriculum, use the
separate, resumable continuation described in
[Student improvement loop](STUDENT_IMPROVEMENT_LOOP.md). That document also
documents student-state collection, exact replay-and-handoff, and managed
single-GPU model swapping.

The base model defaults to `LiquidAI/LFM2.5-350M`. The teacher collector defaults
to model id `Ternary-Bonsai-2-27B` at `http://127.0.0.1:8081`. Both can be
configured. The recipe preserves the configured split registry across stages and
chains adapters: coding starts from the general checkpoint adapter, and joint
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

For a Pi subscription or API provider, use `--teacher-provider` with a model
ID supported by that provider, instead of `--teacher-server`. For example, after
`natlang auth login openai-codex`, pass `--teacher-provider openai-codex
--teacher-model gpt-6-luna`. The same connection is used for the seed teacher
and every discovered curriculum track. `--inline-shapes` controls the generated
cases per family (default 2); increasing it creates a new recipe and run.

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
  G --> R[CPU training-readiness gate]
  R --> H[Render general] --> AH[Token audit general] --> I[Train general]
  I --> J[Render coding] --> AJ[Token audit coding] --> K[Train coding]
  K --> L[Collect seed teacher and discovered tracks]
  L --> M[Admit and materialize teacher turns]
  M --> N[Prepare all teacher tracks with frozen splits]
  N --> Q[Assemble joint corpus with rehearsal]
  Q --> O[Render joint] --> AO[Token audit every track] --> P[Train joint]
```

The recipe creator reads `ts-host/scripts/inline-curriculum/families.mjs` through
`list-tracks.mjs`. Every track with generated families is added automatically;
today this includes `interpreter` and `authoring`. Adding a family to that
registry causes a newly created recipe to include it in the appropriate track.
The recipe freezes the discovered family names and the runtime build, so a
source edit cannot change a running job. Source-backed families require their
pinned external dataset caches and remain in the separate acquisition workflow
in [Inline curriculum](INLINE_CURRICULUM.md); production discovery currently
includes generated families only. A new track with no generated families fails
recipe creation instead of silently being skipped.

Each discovered track builds cases, collects trajectories, checks admission,
and materializes teacher turns under `${run}/tracks/<track>/`. `prepare-teacher`
passes every track's turns through the same group-safe split registry. The
`assemble-joint` stage includes every prepared teacher decision, then samples
at most that many coding rows and at most that many general rows by stable ID
hash. It removes repeated IDs across those lanes. Joint training shuffles the
rows deterministically so the final phase does not finish on just one track.
Its manifest records source
hashes and row counts per track. `audit-joint` records usable rows and training
tokens by `training_track` and stops if any teacher track has no usable training
row. This prevents a collection or tokenizer rejection from quietly removing a
track from the final training set. It does not prove the trained model retained
every skill; that still needs evaluation.

Generated case pools, trajectories, admission ledgers, rendered corpora, and
token caches are run-local working files needed for resume and audit. They are
never auto-discovered as inputs to later recipes. There is no persistent
"artifact registry" to curate: create a fresh run for a new source revision,
and remove an old run directory when its checkpoint and evidence are no longer
needed. Do not remove files from an active or resumable run, because the runner
checks completed output hashes before continuing.

`freeze-runtime` snapshots the already-built `ts-host/dist`, `ts-host/scripts`,
the TypeScript source used to fingerprint the tool surface, `prelude.js`, and
package metadata into `${run}/runtime-host`. Subsequent source
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

## Data-quality and training-readiness gates

Each render is followed by `audit-general`, `audit-coding`, or `audit-teacher`.
The trainer consumes **`<phase>.ready.jsonl`**, not the unfiltered render. These
audits use the selected student tokenizer, immutable renderer identity, and the
effective `--max-len` (including `--train-arg=--max-len` overrides). They never
truncate a completion. Empty targets, invalid termination, missing split/admission,
prompt/target tokenization boundary mismatches, and over-budget examples are
recorded in a rejection ledger and excluded before training budgets are computed.
An audit without any usable training rows fails the stage. The trainer requires
the matching audit and treats a subsequently encountered overlength row as an
error, rather than skipping it and repeatedly consuming other rows.

The renderer also records invalid training views and unapproved rows in
`<phase>.sft.jsonl.rejected.jsonl`. An embedded target termination token is
rejected instead of silently truncating the assistant response. Rendering and
auditing commit hash-checked chunks; SIGINT/SIGTERM finishes the current chunk
and exits 75. Use a new output/run directory when code, tokenizer, budget, or
inputs change. Exact retries reuse committed chunks and validate final artifacts.

For each phase, inspect:

- `<phase>.ready.jsonl.audit.json`: eligible/rejected counts, prompt and supervised
  tokens, length quantiles, and source/repository/family/template/evidence/split
  distributions. Training-only counts and token shares exclude held-out rows.
- Duplicate implementation/prompt-target clusters and largest repeated groups.
  This reports exact identities, not semantic or renamed-code deduplication.
- `<phase>.ready.jsonl.rejected.jsonl`: individual post-render rejection reasons.
- Upstream rejection ledgers, when provided by the recipe, are reported separately;
  their event counts are not added to the rendered corpus's rejected row count.

Existing static reference results (`runs/inline-curriculum/ref-v1.results.jsonl`
and `ref-composed-v1.results.jsonl`) are discovered when present, re-admitted and
materialized on the frozen runtime, and joined into the final teacher curriculum.
Use repeatable `--teacher-results` to select explicit completed result snapshots,
or `--no-existing-teacher-results` to omit them. Old code turn snapshots containing
IR `/1` are held; eligible self-contained saved unit captures are replayed before
coding preparation. See [the static data audit](../plans/STATIC_DATA_RECOVERY.md)
for recovered counts and source adapters still missing since Python retirement.

The replayable synthetic generator has 16 distinct implementation families:
deduplication, prefix sums, word counts, grouping, numeric record totals, longest
strings, numeric/stable-key sorting, strict decimal parsing, ASCII validation,
flattening, row totals, matrix transpose, second distinct maximum, Unicode code
point counting, and sorted unique values. It checks family-specific properties
alongside separate executable references. Some references share algorithms with
the implementation; they are **not** claimed to be independent correctness
oracles. Native replay must still reproduce the expected results.

`--max-family-repetitions` defaults to 50 for replayable families, globally by
index within a generated range: a 1,000-row request emits at most 800 replayable
tasks. The manifest reports requested/emitted/capped counts. All inputs for the
same implementation family share a split group. Broader object, error-handling,
async, and dependency examples use a separate code-only lane; they are not
promoted to native replay evidence.

Source observation now tries deterministic empty/singleton/duplicate/negative/
Unicode boundary inputs and independently varied parameters, plus compatible existing test/capture inputs.
The recipe discovers existing `data/direct-code-2026-09-23/*/captures.jsonl`
snapshots, or accepts repeatable `--captures` overrides. Calls are joined by exact
source task ID, retaining return snapshots as capture fidelity evidence rather
than independent assertions. Captures do not relax the observer's type/effect restrictions. A generated
input outside a function's domain can fail without discarding other valid cases.
Explicit upstream test assertions are checked without replacing their expected
values with observed results: mismatches, contradictory duplicate assertions,
or execution failure on a supported asserted return quarantine the task. Capture
snapshots are only fidelity evidence, not independent assertions; disagreement
with a capture also quarantines conversion. Mutation, unsupported exception
outcomes, async functions, imports, and unsupported types remain explicitly
deferred by the pure source observer. Timeout workers are not a security sandbox.

`training-readiness` is a CPU-only engineering gate, not a student experiment.
It checks completion-only masking, finite-value guards, and actual production
checkpoint serialization/reload with a tiny model, AdamW, scheduler, and RNG.
The continued update must match the uninterrupted update exactly. Its immutable,
idempotent report records source hashes and package versions. Training checks
finite loss before backward and finite gradient norm before optimizer mutation.
A failure before the optimizer step restores the data cursor/counters and RNG
and writes the safe boundary. A failure inside an optimizer/scheduler step is
not transactionally reversible in memory; recover from the prior disk checkpoint.
This CPU gate does not certify GPU kernels, GPU memory fit, or every student
architecture. No teacher process is stopped and no GPU training is performed by
these checks.

Native teacher IR keeps the teacher's original tool calls, arguments, reasoning,
decision order, and execution outcomes. For capable non-reasoning models, pass
`--execution-plans` to `teacher-collector.mjs`, pass `--teacher-execution-plans`
to `create_training_pipeline.py`, or set `TEACHER_EXECUTION_PLANS=1` for
`run_teacher_generation.sh`. Before each action the collector makes a
separate request offering only a required `execution_plan` tool. It validates the
call, feeds the plan into the action request, and stores the plan as
`assistant.execution_plan` and as the turn's training reasoning instead of any
provider reasoning trace. `--execution-plan-tokens` controls the planning reply
limit (default 512). Pi providers use their native required/any tool-choice mode
where available. Planning remains best-effort: if a provider rejects or ignores
required tool choice, the ordinary action continues and that turn records no
synthetic reasoning.

Failed proposals remain in the IR but
are denied positive SFT admission. The training renderer converts JSON-encoded
tool arguments to objects when the selected tokenizer's chat template requires
that shape; it does not change the stored IR. For a small end-to-end CPU smoke
run, pass `--device cpu` to `train_lora.py` with an audited corpus. Production
training still defaults to CUDA.
