# natlang

A research project: fine-tune a very small language model (LFM2.5-350M) to be
the **interpreter** of a natural-language programming language.

A natlang program is a **pseudocode algorithm**: functions with typed
signatures, subroutine calls, `for each`, `repeat until`, `if`/`else`, local
variables, organised as a **code base** of `.nl` files (frontmatter plus a
pseudocode body; `somefun.nl` with an optional companion folder `somefun/`;
`.ts` files for exact functions; reuse through `uses` links). The author
states the structure. The model carries it out one small step at a time with
eight tools (`read`, `write`, `edit`, `run_code`, `call`, `mark_done`, `report_blocker`, `report_error`);
every function instance is a fresh short episode over a typed object tree.
Prompt-like tasks (judge, classify, extract, rewrite) are the leaves. The
harness provides memory, typing, a sandbox and I/O; it parses no instructions
and owns no control flow.

```
# examples/triage/main.nl (body)
function triage(tickets, rubric) -> Report
  labels   = for each t in tickets: classify(t, rubric)
  not_spam = for each l in labels: l is not "spam"            # exact: use code
  real     = select_by_flags(tickets, not_spam)
  flags    = for each t in real: is_urgent(t)
  ...
```

## Documents

| File | Contents |
|------|----------|
| `PLAN.md` | Thesis, language design, tool surface, runtime, data generation, phases, status and findings |
| `plans/AMBITIOUS_ARCHITECTURE.md`, `plans/EMBEDDING_REFACTOR.md`, `plans/EXECUTION_INTERFACES.md`, `plans/AMBITIOUS_PROJECTS.md`, `plans/AMBITIOUS_ROADMAP.md` | Minimal-core architecture, embedding refactors, stream/evaluator/type/seed/trace interfaces, 20 application families and teacher/student training loop |
| [Infrastructure implementation plan](plans/INFRASTRUCTURE_IMPLEMENTATION.md) | Prioritised cross-project infrastructure, 17 proposed patches, code touchpoints and acceptance gates |
| [Individual project plans](plans/projects/README.md) | Concrete natlang feature dependencies, eval/host boundaries, delivery gates and teacher/trace requirements for all 20 projects |
| `spec/SPEC.md` | The normative language specification (v0.2-draft); `spec/CODEBASES.md` gives the rationale for code bases and `call` |
| `TYPES.md` | Type system, validation, write-time typing, how validation feedback reaches the model |
| `TRAINING.md` | Use cases, skill taxonomy, datasets, the teacher's roles, training recipe, evaluation |
| `SYNTHETIC_DATA.md` | Detailed designs and prior art for the fifteen synthesized datasets |
| `examples/`, `codebases/` | Code bases on disk: `triage`; `nlprolog`, `highlighter`, `webserver`, `moderation`, `shopkeeper`, `legal_move`; the linkable crisp library `std/` |
| `conformance/` | Program references and infrastructure baseline fixtures |
| `natlang/` | The harness (Python): tree, types, code bases, tool surface, constrained decoding, generators |

```
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e '.[js,dev]'
.venv/bin/python -m pytest -q          # harness, code bases, tool surface, generator
python3 tools/check_conformance.py    # static checks on the suite and the spec's type expressions
```

Generating training data (CPU only; every turn is checked by the harness, by
the grammar of its own turn, and against the expected value):

```
.venv/bin/python scripts/generate.py --n 2000 --seed 1 --out data/ref.jsonl
```

The generator mixes leaf tasks, synthesized pseudocode programs (`natlang/gen/synth.py`) and the hand-written
code bases (`natlang/gen/codebases.py`); choose with `--families`.

The whole loop on one 8 GB GPU (the training image reuses any local PyTorch image: see `docker/train.Dockerfile`):

```
scripts/serve.sh &                                                          # the base model, for its chat template
.venv/bin/python scripts/export_sft.py data/ref.jsonl data/sft.jsonl        # pairs rendered exactly as at inference
docker build -t natlang-train -f docker/train.Dockerfile docker
scripts/train_and_publish_browser_docker.sh data/sft.jsonl runs/lora natlang-350M 300
# trains LoRA on the GPU, merges the checkpoint, converts Q4_K_M GGUF, and publishes the browser default
# training checkpoints are resumable; a repeated wrapper invocation continues the same run
# for manual merge-only or fresh starts, call scripts/train_lora.py with those flags
node scripts/publish_browser_model.mjs --run runs/lora --name natlang-350M --quant Q8_0  # optional larger browser variant
.venv/bin/python scripts/eval_turns.py data/ref.jsonl                       # next-turn accuracy per kind of turn
.venv/bin/python scripts/baseline.py                                        # whole programs, graded by their checks
```

The published browser catalog is `models/browser-catalog.json`. It is written only after GGUF conversion and verification succeed. The playground, browser pilot, browser board, and applications using `BrowserNatlangClient.loadDefaultModel()` read its `defaultId`. The catalog and generated weights live in the local, ignored `models/` directory; copy both with the matching template to a deployment's `/models/` directory. Direct calls to `scripts/train_lora.py` still produce a merged checkpoint; use `scripts/publish_browser_model.mjs --run runs/NAME --name NAME` afterward to publish it.

A web server whose every request is interpreted by the model (`codebases/webserver`):

```
scripts/serve_web.py --port 8000 --server http://127.0.0.1:8080
```

Running programs with a model (needs Docker with the NVIDIA runtime):

```
mkdir -p models && curl -L -o models/LFM2.5-350M-Q8_0.gguf \
  https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/resolve/main/LFM2.5-350M-Q8_0.gguf
scripts/serve.sh &                      # llama.cpp CUDA server on 127.0.0.1:8080, official chat template; stop: docker stop natlang-llama
.venv/bin/python scripts/baseline.py 01 02 23      # tool surface + native constrained decoding (defaults)
.venv/bin/python -m natlang run examples/triage/main.nl --in tickets=./tickets/ --in rubric=./rubric.md
```

Validation-policy experiments (current student server):

```
.venv/bin/python scripts/validation_probe.py --out runs/validation.json
.venv/bin/python scripts/validation_probe.py --controlled-only --out runs/validation-controlled.json
.venv/bin/python scripts/application_probe.py --validation-feedback caller --out runs/applications-caller.json
```

The default interpreter leaves a failed lambda quiesced and returns validation
diagnostics to its caller (`validation_feedback="caller"`). Local repair remains
an explicit experiment: `ToolAgent(validation_feedback="local")`, or
`--validation-feedback local` in the application/conformance runners. This does not roll back effects or automatically
retry. The probe distinguishes impossible instructions from repairable execution
mistakes, and counts budget exhaustion separately from deliberate failure.
Results and limitations are in TRAINING.md. `serve.sh` caps its host prompt cache
at 256 MiB; override with `NATLANG_CACHE_RAM` if needed.

Teacher (Ternary Bonsai 2 27B on an 8 GB GPU; the two-worker launch is capped at 5 GB of host RAM):

```
docker build -t natlang-prism-runtime -f docker/prism.Dockerfile docker
# binaries: PrismML-Eng/llama.cpp release, CUDA 12.8 tarball, unpacked into vendor/prism/bin
# model:    prism-ml/Ternary-Bonsai-2-27B-gguf  Ternary-Bonsai-2-27B-PTQ1_0.gguf  -> models/
# template: prism-ml/Ternary-Bonsai-2-27B-mlx-2bit chat_template.jinja -> models/templates/Ternary-Bonsai-2-27B.jinja
scripts/serve_bonsai.sh 8081 32768 99 2 & # stop: docker stop natlang-bonsai; scripts/watch_bonsai.sh keeps it up
scripts/run_teacher_generation.sh        # resumable coverage generation; rescans sources between waves
.venv/bin/python scripts/baseline.py --decode server --server http://127.0.0.1:8081 --thinking 512 --temperature 0.6 \
    --system-file natlang/prompts/tools_delegate.md --alias call=call_function --verbose 23
.venv/bin/python scripts/paraphrase.py --server http://127.0.0.1:8081     # paraphrases, kept only after a round trip
.venv/bin/python scripts/teacher_leaves.py --server http://127.0.0.1:8081  # references for leaves that generate text, kept only if checks pass
```


Training now holds out complete programs and saves the split in `runs/<run>/split.json`.
Checkpoints verify the corpus and split before resuming. Checkpoints made before
this change can still be exported with `--merge-only`; use a new run directory
for training with the program split.

For paired model comparisons, reuse `eval_turns.py --manifest PATH` (the default
is beside the corpus). The manifest fixes sample IDs and content hashes across
models. Results separate work actions from replies and are written as JSON to
`runs/`, or `--out PATH`. `baseline.py` also writes JSON with computed totals and
actual emitted records. Pass `--model-label NAME` to identify a checkpoint.

Crisp code runs in QuickJS. Plain JavaScript needs no Node installation; erasable
TypeScript annotations additionally need Node >=22.13 for
[`stripTypeScriptTypes`](https://nodejs.org/download/release/v22.13.1/docs/api/module.html).
This strips annotations without static type checking; values remain checked at
the tree boundary. Effectful code has a killable worker and a wall-clock limit.
Host capabilities retain their state in Python; a timed-out host operation can
still complete, so external effects need host cancellation or idempotency.


Three application families exercise larger algorithmic and architectural patterns:

| Entry point | Pattern | Generated edge cases |
|---|---|---|
| `codebases/reconciliation/reconcile.nl` | Deduplicate events, join customers, map semantic triage, aggregate exact cents | Empty inputs, unknown customers, conflicting redeliveries, negative amounts, reserved dictionary keys |
| `codebases/dependency_plan/plan.nl` | Bounded topological planning with semantic priority and exact readiness checks | Cycles, missing dependencies, disconnected tasks, empty graphs, deterministic ties |
| `codebases/order_saga/step.nl` | Event fold with an outbox, compensation, duplicate suppression and resumable dispatch | Repeated/out-of-order events, payment then cancellation, lost acknowledgements, idempotent effects |

Generate verified trajectories with `scripts/generate.py --families
cb_reconciliation cb_dependency_plan cb_order_saga`. These families are available
explicitly; the existing v7 mixture remains stable until the application pass is
reviewed. The oracles compare final state and, for the saga, the exact delivered
command sequence. The model remains responsible for interpreting each `.nl` body.

### Explicit execution errors

`report_blocker(missing=...)` signals missing information or an uncovered case.
`report_error(message=...)` signals contradictory instructions, an invalid operation,
or an impossible required result on the executed path. Both end the current episode
without a completed return and pass a diagnostic to the caller. Existing effects
remain; neither tool automatically retries. The distinction is model-facing, not
an additional runtime failure state.

The anti-fudging prompt is experimental:

```bash
.venv/bin/python scripts/validation_probe.py --policies local --system-file scripts/prompts/no_fudging.md --out runs/no-fudging.json
```

Omit `--system-file` for the usual prompt; add `--no-error-tool` to reproduce the
previous tool inventory. Use `--controlled-only` for matched execution mistakes
and contradictory programs after the same injected validation error.

### Throughput controls

The validation and application probes default to four independent workers (matching
`scripts/serve.sh`'s four server slots). Use `--workers 1` for a serial comparison.
Each case owns its decoder, usage counters, deadlines, and runtime; calls within a
program retain their dependencies. Results include wall time and per-case usage.

Training supports `--microbatch N`, `--batch-tokens N`, `--token-cache PATH`, and
`--[no-]gradient-checkpointing`. `--accum` still counts **examples per optimizer
step**, so changing microbatch size preserves the intended loss weighting.
`--checkpoint-above-tokens N` retains checkpointing only above that padded-token
count. `--retain-every-n-layers N` keeps every Nth decoder layer's activations
and checkpoints the intervening layers, allowing a measured memory/speed tradeoff.
Larger batches and reducing checkpointing are opt-in: neither is reliably faster
or memory-safe on the laptop. Per-step throughput is saved in
`throughput.json`; `--benchmark-steps N` runs without saving a trained model.
Use `--unsloth` for optimized dense-model QLoRA; `--unsloth-lfm-experts` adds
the packed expert restoration required by the prequantized LFM MoE checkpoint.

Build the optional CUDA convolution kernel against the existing training image:

```bash
docker build -f docker/kernels.Dockerfile -t natlang-train-kernels .
```

Generate matched error/blocker/success/repair references with grammar, return, and
effect verification:

```bash
.venv/bin/python scripts/generate_failures.py --groups 100 --workers 4 --out data/ref-failures.jsonl
```

Each group contains twelve programs; paired variants share a program identifier
so a training/held-out split cannot separate the pair. Invalid injected actions
appear only in history; supervised targets are successful repairs or explicit
failure reports.

To diagnose whether typed decoding hides incorrect writes:

```bash
.venv/bin/python scripts/validation_probe.py --policies caller --write-constraints runtime --trace-probs --out runs/runtime-write-types.json
```

`--write-constraints runtime` is now the native-decoder default: literal write
types/values reach the runtime validator, while tool syntax and path choices
stay constrained. Use `--write-constraints typed` for comparisons. A single
JSON-text layer can still be parsed, and scalar values can acquire missing quotes
for a Text slot. Existing Text is preserved verbatim. Extra `{value: ...}` object
wrappers are rejected unless the destination actually expects that record.
`--trace-probs` saves selected token IDs/logprobs and top alternatives before
sampling constraints. These are next-token probabilities, not calibrated
probabilities that the action is correct. Missing top-k entries are unknown.


### Experimental careful mode

`--careful-threshold P` enables a check before applying low-confidence writes or
edits in validation/application/conformance probes. The score is the geometric
mean of raw selected-token probabilities over the proposed value. It is **not**
a calibrated probability of correctness; unknown confidence stays unknown.

The review forks the pre-action context with the pending proposal. Its guided
response contains a short reason followed by `approve`, `error`, or `blocker`.
Only approval releases the unchanged proposal to normal runtime validation.
The review conversation never enters the original agent history. Budgets cover
both execution and reviews, and no action in a reviewed batch executes before
all required reviews approve. Careful mode is off by default.

```bash
.venv/bin/python scripts/validation_probe.py --policies caller --careful-threshold 0.5 --out runs/careful.json
.venv/bin/python scripts/confidence_probe.py --groups 10 --out runs/confidence.json
```

See [careful-mode results and limitations](CAREFUL_MODE.md) for the measured
threshold, instance-held-out results, and examples of confident errors and false
rejections. The threshold is experimental and specific to this model/task mix.


### Experimental agent support

[TEACHER_SETUP.md](TEACHER_SETUP.md) describes the simple teacher configuration,
teacher-specific interface fixes, behavioral audits, and audited leaf collection.
[LEARNING_LESSONS.md](LEARNING_LESSONS.md) links observed teacher/student
difficulties to honesty, persistence, and calibration training contrasts.

[AGENT_SUPPORT.md](AGENT_SUPPORT.md) records the execution-state, action-review,
and prompt-reminder experiments. Expanded state is opt-in (`--state-view`);
the current checkpoint regressed with it. `--review-scope actions` also checks
calls, completion marks, edits, and source copies. `--withdrawal-policy retry`
allows one reconsideration of a withdrawn batch; the default returns to the
caller. `--review-prompt repeat_instructions` or `checklist` changes only the
isolated review fork. These options are available in the baseline, application,
validation, and confidence probes. `scripts/review_probe.py` compares decisions
on identical saved proposals without executing them.
