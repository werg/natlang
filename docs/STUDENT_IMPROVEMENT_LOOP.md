# Student improvement loop

The fastest useful path is a **single curriculum followed by repeated,
execution-verified correction rounds**. Do not spend teacher tokens producing
many interchangeable ideal traces for easy tasks. Spend them on states the
current student actually visits and cannot resolve. A failed proposal is
evidence about what to teach, not an SFT target.

## What is implemented

The production recipe (`create_training_pipeline.py`) already builds and
trains, in order:

1. General JS/TS code completion and implementation capability.
2. Native coding cases: verified unit-test replay, synthetic cases, and
   correctly labeled source-proposal rehearsal.
3. Accepted native teacher decisions, including failure-repair decisions,
   after exact-runtime admission and split-preserving preparation.

`create_improvement_round.py` makes a **new, resumable one-phase recipe**
from additional accepted native teacher turns. It starts from the previous
completed teacher adapter, reuses the previous frozen split registry, renders
and audits with the same student model/tokenizer settings, and trains only on
the new approved corrections. Its inputs and outputs are hashed by the normal
pipeline runner. It refuses a partial or unrelated base run, missing adapter,
missing admission evidence, or a correction file with no approved decisions.
As with the initial recipe, SIGINT/SIGTERM asks the trainer for a checkpoint;
re-running the same recipe/run directory resumes.

For a supplied, already-verified turn file:

```sh
.venv/bin/python scripts/create_improvement_round.py \
  --base-recipe runs/production-recipe.json \
  --base-run runs/production-2026-09 \
  --teacher-turns data/teacher/round-1-verified.turns.jsonl \
  --output runs/round-1-recipe.json

.venv/bin/python scripts/run_training_pipeline.py \
  runs/round-1-recipe.json runs/round-1
```

For round 2, pass round 1's recipe and run as `--base-recipe` and `--base-run`.
The helper requires its `train-correction` stage to be complete, initializes
from that adapter, and carries forward the original frozen split registry.
Repeat with a new recipe and run directory for each round.

If the original recipe uses Docker, the base run and teacher-turn file must be
inside the repository because those paths must be visible under its existing
mounts. Use a separate run directory per model and round. Never mutate a
completed input artifact in place.

## Complete collection-and-training round

`create_student_improvement_pipeline.py` assembles the complete round around
two OpenAI-compatible chat-completions endpoints. The student endpoint **must
serve the adapter from the completed base run**. Model ID alone does not prove
that; check your server configuration before starting. The teacher endpoint
serves the stronger teacher. The recipe freezes the current built TypeScript
runtime, and every collector/materializer stage uses that same snapshot.

```sh
npm --workspace @natlang/typescript-host run build:node

.venv/bin/python scripts/create_student_improvement_pipeline.py \
  --base-recipe runs/production-recipe.json \
  --base-run runs/production-2026-09 \
  --programs runs/production-2026-09/teacher-programs.jsonl \
  --run runs/improvement-1 \
  --student-server http://127.0.0.1:8080 --student-model trained-student \
  --teacher-server http://127.0.0.1:8081 --teacher-model Ternary-Bonsai-2-27B \
  --output runs/improvement-1-recipe.json

.venv/bin/python scripts/run_training_pipeline.py \
  runs/improvement-1-recipe.json runs/improvement-1
```

The recipe is bound to its `--run` directory. `--limit N` restricts student
programs; the default zero means all. `--workers N` and `--root-seed N` apply
to both collectors. The supplied program file must contain focused
`natlang.program/2` records with exact native oracles. The base run must be
complete; its adapter and split registry are hashed inputs to the continuation.

For a non-quantized LoRA checkpoint, `train_lora.py --merge-only` can export a
merged Hugging Face model, and `scripts/to_gguf.sh` plus `scripts/serve.sh` can
serve supported models. A QLoRA adapter cannot use that `--merge-only` path;
serve it with an OpenAI-compatible stack that loads the base plus PEFT adapter.
With the ordinary pipeline runner, serving/port allocation is external:
restarting an endpoint does not corrupt completed jobs, but the operator must
ensure the same checkpoint remains served throughout a round. Use the managed
runner below to automate service switching and GPU release.

### Automatic single-GPU model swapping

If the trained student has been exported to a GGUF file under `models/`, the
managed runner can own both Docker servers and swap them on one GPU. Create the
round recipe above, then make its service plan:

```sh
.venv/bin/python scripts/create_model_swap_config.py \
  --recipe runs/improvement-1-recipe.json \
  --run runs/improvement-1 \
  --student-gguf trained-student-Q8_0.gguf \
  --output runs/improvement-1-services.json

.venv/bin/python scripts/run_managed_improvement.py \
  runs/improvement-1-recipe.json runs/improvement-1 \
  --services runs/improvement-1-services.json
```

The service plan gives each container a run-specific name; unlike manual
`serve.sh`/`serve_bonsai.sh`, managed launches do not remove an existing
default-named container. It pins the student GGUF and base adapter file hashes;
the operator must still ensure that the GGUF was exported from that adapter.
The supervisor refuses a live endpoint it does not own. It waits for student readiness, runs through `build-handoffs`, stops
the student container and waits for the endpoint to go down, starts Bonsai,
runs through `combine-verified`, stops Bonsai, then resumes preparation and
GPU training. Student and teacher may use the **same port**. Signals propagate
to the pipeline so a training interruption can checkpoint; owned services
are cleaned up. The service journal allows a restart to clean up an orphaned
owned launch after a supervisor crash. Re-running the command skips completed
collection phases without relaunching their models.

The default uses separate containers sequentially because Bonsai's PTQ1_0
weights need the Prism llama.cpp fork, while the student GGUF script uses
standard llama.cpp. They do not occupy VRAM together. A custom service plan
can reuse one container if it contains both servers and its start/stop commands
switch the active process safely.

The default service-plan generator targets the repository's llama.cpp GGUF
student script and Bonsai Docker script. For other model families or QLoRA,
provide a `natlang.model_swap/1` JSON service plan with per-role `endpoint`,
`ready_url`, `start` argv, optional `stop` argv, `env`, and
`startup_timeout_seconds`. Use a stop command scoped to a service/container
created for this run; the supervisor will not stop an unowned pre-existing
endpoint. The server must implement the OpenAI-compatible chat-completions
surface used by the collector. A service plan automates **switching**, not
checkpoint export: ensure the student launch actually serves the completed
adapter.

Each student job records its model requests, offered tools, responses, action
outcomes, scope failures, and final oracle result. Decoded replies are journaled
before actions execute; a stopped job replays them locally on restart.

`build-handoffs` (`ts-host/scripts/build-handoffs.mjs`, `teacher/handoff.ts`)
turns each failed run into handoff tasks, one per site: a decision that failed
(an action the runtime rejected or refused, one that raised, an attempt the
task's checker refused; the first of a call's failures in a row), or the result
a run returned that the task did not accept. A handoff task is the program
record with the run's earlier turns attached, per call. The collector replays
them call by call, calls identified by the order they start rather than by
their opening text (which changes with runtime versions), and the teacher
answers the failed turn and every turn after it. A site is used only if the
failed turn, replayed under the current runtime, sees the results it saw and
fails again, so runs from older runtimes, other models and evaluation runs all
qualify.

The rules are:

- Oracle-accepted student successes become positive SFT with distinct
  `student-native` provenance. Failed proposals within an accepted run remain
  in the IR but are not positive targets.
- The teacher takes over **at** the failed decision, from the student's actual
  state before it: its decision there is what the student should have done.
- A correction is admitted only after the runtime executes its full
  continuation and the task oracle (and a curriculum case's admission) accepts
  it. Replayed student decisions are never positive targets.

The teacher correction and student successes are combined, validated, prepared
against the original frozen split registry, rendered, token-audited, and
trained in the new round. Every stage has hash-checked inputs/outputs and
resumes through the normal runner. Stop with SIGINT/SIGTERM to request an
optimizer-boundary checkpoint.

## Preference pairs and DPO

`build-preferences` (`ts-host/scripts/build-preference-pairs.mjs`) writes
`preference-pairs.jsonl`: at the handoff point, the teacher's approved decision
is chosen and the student's failed decision rejected, both after the same
context. The pair is checked again by replay: the rejected response, put in
the chosen decision's place, sees the same results and fails. Corrected
variants (`corrections.mjs`: a run whose call failed and then fixed itself,
rewritten to make the fix first) give pairs too: the fix over the failed
attempt, with the same reasoning on both sides. `build_lora_sft.sh` builds the
corrected-variant pairs with every SFT set (and handoff pairs given
`NATLANG_HANDOFFS`), rendered to `preferences.jsonl` by
`export-preference-pairs.mjs`. `scripts/train_dpo.py` trains a short DPO stage
on top of the SFT adapter, its reference being that adapter.

## Operational admission rules

The improvement recipe accepts `natlang.teacher_training_turn.native/1` rows
with accepted final oracle, explicit `exact-native-runtime-oracle` training
admission, and admitted trace evidence. The curriculum preparation drops
individual unapproved decisions (for example a failed proposal inside an
ultimately accepted trajectory), links them to the original program groups,
and applies the frozen split registry. The render and token audit must find
usable train rows before the optimizer starts. Inspect the resulting
`correction.ready.jsonl.audit.json`, rejection ledgers, and checkpoint state;
mere existence of a turn file is not evidence of trainable corrections.

This is one production methodology, not a baseline or ablation program. The
stages above are engineering/admission checks, not comparative experiments.
