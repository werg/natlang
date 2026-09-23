# Program IR and reproducible training builds

The durable teacher data is **program IR** (`natlang.program/2`), not a tool
trace or a model-formatted SFT pair. Each JSONL row is a small natlang project
plus its evaluation contract:

```json
{ "version": "natlang.program/2", "id": "source:dependency_plan:0", "kind": "lambda_graph",
  "family": "cb_dependency_plan", "split": "train", "source_groups": ["cb_dependency_plan"],
  "license": "project-generated",
  "semantics": { "root": "plan.nl", "files": { "plan.nl": "---\nargs: …", "plan/step.nl": "…", "types.ts": "…" },
                 "inputs": { "tasks": [] }, "expected": { },
                 "effects": { }, "folder_files": { }, "expected_files": { }, "failure_seed": { } } }
```

- `root` and `files` are exactly what an author writes: a root `.nl` function, its callable folder, and `types.ts`. The runtime loads them with the same loader as applications (`ts-host/src/teacher/program.ts`).
- `inputs` bind the root's parameters by name; `expected` is the checked result; `operation: "blocked"` expects a reported blocker instead.
- `effects` describe service contracts the harness provides and checks (`record_args`, `deliver_once_ack_loss`); `folder_files` and `expected_files` give a directory reducer's input folder and required result; `failure_seed` injects a failing first eval to collect repair behavior.
- IR contains no tool schema, system prompt, assistant turn, chat template, or tokens. Traces and SFT pairs are rebuilt from it when the runtime, prompts, or template change.

## Producers

```bash
node ts-host/scripts/generate-synthetic-ir.mjs --out data/teacher/native-synthetic.ir.jsonl --seed 909 --n 300
node ts-host/scripts/freeze-source-teacher-cases.mjs          # codebases/ programs × frozen seeds
node ts-host/scripts/build-read-before-code-probe.mjs         # folder-reading probe
node ts-host/scripts/failure-corpus/freeze.mjs OUT.jsonl      # seeded failure-and-repair cases
node ts-host/scripts/import-playground-cases.mjs CASES.jsonl OUT.jsonl
node ts-host/scripts/code-corpus/…                            # captured source-function corpus (plans/CODE_CORPUS.md)
```

Producers that describe functions as data use `definitionProject` to write the
files. `ts-host/scripts/migrate-program-ir.mjs FILE...` upgraded the tracked
`natlang.program/1` files (a `$lambda` root with definition-style children) once;
new data is produced as `natlang.program/2` directly.

**Eval rejects `var`.** Any code that becomes an eval body in training data (captured source functions, teacher repairs, synthetic implementations) must use `let`/`const`: rewrite each `var` declaration to `let` while preparing the data (the scoping differences are edge cases), rather than dropping the example.

## Collection, materialization, export

```bash
node ts-host/scripts/teacher-collector.mjs IR.jsonl JOBS/ OUT.jsonl --model-id ID --root-seed 909 --server URL --limit 50 --workers 4
node ts-host/scripts/materialize-native-teacher.mjs OUT.jsonl TURNS.jsonl
node ts-host/scripts/export-native-sft.mjs TURNS.jsonl SFT.jsonl --server URL --end-token TOKEN
```

The collector runs each program through the interpreter with the teacher
model, journals every model reply so an interrupted job resumes without
re-decoding, checks the value, effects, and files against the contract, and
publishes one result per job atomically. Provenance pins the IR digest, model,
seed policy, system prompt, segmentation, and a hash of the interpreter source
(`tool_surface_sha256`); a job only resumes against identical provenance. The
materializer turns accepted trajectories into template-neutral turns; the
exporter renders them with a specific chat template and is model-specific and
disposable. See [TEACHER_SETUP.md](TEACHER_SETUP.md) and
[ts-host/TEACHER_MATERIALIZATION.md](ts-host/TEACHER_MATERIALIZATION.md).

Changing the interpreter or prompts creates a new trace build; changing a chat
template creates a new SFT build. Never append records made under different
versions to one file.

## Retired sources

The Python pipeline built program IR of other kinds (`typed_decision`,
`state_sequence`, `numeric_program`, `scene_program`, `lambda_scenario`) from
NanoJev and typed decisions, sales labels, SCONE, Schema-Guided Dialogue, FinQA,
CLEVR, and generated failure/support scenarios, and materialized them through the
Python runtime. Those adapters were removed with that runtime. Their source
files and the IR they produced remain under `data/`, and the adapters are in the
repository history. Bringing a dataset back means writing an adapter that emits
`natlang.program/2` projects; until then those corpora are not part of the
training mix. The pre-IR traces and SFT pairs in `data/external_pilot/` are not
canonical training sources.

Source references: [SCONE](https://nlp.stanford.edu/projects/scone/),
[Schema-Guided Dialogue](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue),
[FinQA](https://github.com/czyssrs/FinQA), and
[CLEVR](https://web.eecs.umich.edu/~justincj/clevr/).
