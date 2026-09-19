# Handover: natlang, state of work on 2026-09-19

For an agent (or person) who continues the work of generating data, training and
supervising runs. Read this first, then `PLAN.md` §10 (decisions, status, findings)
and `spec/SPEC.md` (the language). Everything below was true when written; verify
paths and flags before relying on them.

## 1. What this project is, in five sentences

A very small model (LiquidAI **LFM2.5-350M**) is fine-tuned to be the **interpreter**
of natural-language **pseudocode programs**. A program is a code base of `.nl`
files (frontmatter with a typed signature, then a pseudocode body) and `.ts` crisp
functions; the author states the structure (calls, for-each, fold, repeat-until,
if/else, locals), and the model carries it out one small tool call at a time over
a typed object tree. The harness provides memory, typing, a sandbox and I/O; it
**parses no instructions, holds no cursor, does no control flow**; its constraints
are type-level and applied at write time, including through grammar-constrained
decoding. Prompt-like tasks (judge, classify, extract, rewrite) are the *leaves* of
programs. The thesis to demonstrate: small model + stated structure does useful
work at a tiny fraction of a large model's time.

## 2. Decisions by the project owner that you must not re-litigate

All are recorded in `PLAN.md` §10.1. The ones most likely to matter day to day:

- The **model** is the interpreter. No harness cursor, no parsing of instructions.
- **No anonymous lambdas.** A sub-task is always an instance of a code-base
  function, started with `call(function, to, inputs, over | over+init | init+until+max)`,
  which places and runs it in one action; calling again with only `function` and
  `to` resumes what did not finish. There is no `run` tool.
- **No recursion.** The loader refuses any code base with a cycle. Repetition is
  `call` with `over` / `init` / `until`. (The Prolog code base is forward chaining
  by repeat-until for this reason.)
- To vary a function: copy it into a local (`write` with type `Function<f>`),
  `edit` the copy, call `let/<copy>`. Code bases are immutable.
- **Instructions are immutable while a lambda runs. What is done is state**:
  `marks` per line, shown as a numbered listing with `[ ]` `[x]` `[-]`; written
  with `mark_done(start, end?, skipped?)` or the `done` argument of `write`/`call`.
  Crossing finished steps off by editing the text is gone. Which marking style
  models prefer (standalone turn / grouped with the next action / en passant) is
  **to be measured** with teacher and student; see §7.
- The reply never carries the result; the result is what was written to `return`.
  Data reaches the model only through the tool channel.
- Seven fixed tools: `read, write, edit, run_code, call, mark_done, report_blocker`
  (`call`, `mark_done` only when the lambda has functions). No selective filtering.
- Template-agnostic by default: prompts are rendered by the loaded model's own
  official chat template; per-model adaptations (tool aliases, prompts) are opt-in.
- Teacher paraphrases vary wording, dialect and names, **never structure**.
- Generative leaves get **teacher-written references accepted by checks**.
- Training at scale happens on the owner's larger GPU; this laptop must prove the
  whole system, including a memory-constrained training mode.
- Process preferences: **no fresh Fable subagents; forks sparingly; rote work may
  go to a Sonnet agent on request. Work on branch `main`, commit and push often.**
  Ask before large downloads or killing things that are not yours. Attribution
  lines on commits as in `git log`.

## 3. Machine and services

Laptop, 12 cores, 14 GB RAM (it swaps; other desktop apps run), one 8 GB NVIDIA GPU,
Docker with the NVIDIA runtime. Python venv: `.venv` (uv). GPU memory is the
constraint that decides what can run together:

| service | how to start | port | GPU | notes |
|---|---|---|---|---|
| student (base or tuned GGUF) | `scripts/serve.sh [MODEL.gguf]` | 8080 | ~1 GB | container `natlang-llama`; official chat template from `models/templates/<name before -Q>.jinja`; stop: `docker stop natlang-llama` |
| teacher, Ternary Bonsai 2 27B | `scripts/watch_bonsai.sh 8081 &` (keeps it up) or `scripts/serve_bonsai.sh 8081` | 8081 | ~6 GB | container `natlang-bonsai`; Prism ML llama.cpp fork in image `natlang-prism-runtime`; `--no-mmap --cache-ram 1024`, 5 GB memory cap, 12k context. Stop: kill the watchdog (`pgrep -f watch_bonsai`) **then** `docker stop natlang-bonsai` |
| training | see §6 | - | ~3.2 GB | container `natlang-train`, image `natlang-train` (`docker/train.Dockerfile`, built on the local `chesst-zero-train` PyTorch image) |

Teacher + student fit together; **teacher + training do not** (stop the teacher first).
Bonsai quirks already absorbed: its server cannot emit a tool literally named `call`
(use `--alias call=call_function` / `tool_aliases`); it delivers records as JSON text
and sometimes wraps values as `{"value": X}`; judge calls must disable thinking with
`chat_template_kwargs: {enable_thinking: false}` (fine for simple yes/no checks on an answer; for judging whether two
lines are *equivalent* it said "no" to plainly equal lines, and a 160-token thinking budget fixed that: calibrate a
judge on known pairs before trusting it). Speed: ~23 tok/s, 6 to 8 s per turn,
4 to 10 minutes per code-base program.

When this file was written: Bonsai and its watchdog are **up**; the student server on
8080 serves `natlang-350M-v5-Q8_0.gguf` (the 15-minute proof model); no training runs.
A cron heartbeat exists only inside the previous agent's session and dies with it.

Never use `pkill -f <pattern>` with a pattern that also matches your own shell command
line (it kills your shell; this happened twice). Use `kill $(pgrep -f "[g]enerate.py")`.

## 4. Repository map

- `spec/SPEC.md` (v0.2-draft, normative), `spec/CODEBASES.md` (rationale), `PLAN.md`,
  `TRAINING.md`, `SYNTHETIC_DATA.md`, `TYPES.md`, `README.md` (run commands).
- `natlang/`: `types.py, values.py, nodes.py, refs.py, paths.py` (typed tree);
  `codebase.py` (loader for `.nl`/`.ts`, lexical scope, `uses` links, cycle refusal);
  `runtime.py` (Runtime, Session, `_place_call`, `_op_*` tools, combinators, guards,
  `_progress`, `_op_mark_done`); `surface.py` (the model-facing tools with per-turn
  schemas and grammar alternatives; `NATLANG_MARKS=0` and `NATLANG_DONE_ARG=0` switch
  marking off); `render.py` (listings, marks); `native.py` (GBNF over the model's
  native call text; accepts JSON literals); `decoder.py` (llama.cpp server client,
  tool aliases, strips `x-` schema keys for servers); `tool_agent.py` (the agent loop,
  nudges); `checks.py` (grading: value / status / crisp checks / judge); `host.py`
  (`load`, `load_fold` for long-lived folds over an open list, `instantiate`);
  `js.py` + `prelude.js` (QuickJS sandbox; `args`, `locals`, `fx`).
- `natlang/gen/`: `programs.py` (leaf families, `Plan`, `Program`), `synth.py`
  (shapes, the compositional synthesizer, phrase bank, marking styles),
  `domains.py` (declarative domains), `codebases.py` (generators + reference scripts
  for the hand-written code bases, `with_marks`, teacher-reference store),
  `policy.py` (ReferenceAgent: runs plans through the real harness and asserts that
  the turn's grammar accepts every reference turn), `world.py`.
- `codebases/`: `nlprolog, highlighter, webserver, moderation, shopkeeper, legal_move,
  mail_rules`, plus `std/`; `examples/triage` (+ `examples/std`). Each has a scripted
  end-to-end test in `tests/`.
- `conformance/programs/01..23` (code-base style, each with a replayable `reference:`),
  `conformance/harness/`, `tools/check_conformance.py`.
- `scripts/`: `generate.py`, `export_sft.py`, `train_lora.py`, `to_gguf.sh`,
  `eval_turns.py`, `baseline.py`, `paraphrase.py`, `paraphrase_steps.py`,
  `teacher_leaves.py`, `serve.sh`, `serve_bonsai.sh`, `watch_bonsai.sh`, `serve_web.py`.
- Not in git (`.gitignore`): `models/`, `runs/` (all logs), `vendor/`, `data/` except
  the force-added teacher outputs `data/paraphrases.json`, `data/paraphrase_report.json`.
  **Force-add** `data/leaf_references.jsonl` and `data/phrases.json` when they have
  content (`git add -f`): they cost GPU hours.

Tests: `.venv/bin/python -m pytest -q` (97 pass) and `python tools/check_conformance.py`.
Run both before every commit.

## 5. Data generation (the current focus)

Every reference turn is verified three ways at generation time: the real harness
accepts the call, the grammar built for that turn accepts the turn's text, and the
program's outcome equals what a Python twin / the latent world says (for exact steps
the *executed code* is the oracle; Python and JavaScript round differently).

```
.venv/bin/python scripts/generate.py --n 20000 --seed 71 --mix v7 --out data/ref-v7      # sharded, parallel, resumable
.venv/bin/python scripts/generate.py --n 500 --seed 779 --mix v7 --out data/ref-eval-v7.jsonl --keep-alternatives
```

- A directory as `--out` writes `part-XXXXXXX.jsonl.gz` shards; finished shards are
  kept on a rerun. Program `i` of a run is a function of `(seed, i)` only
  (`make_program`), so runs can be resumed and revisited. ~1,000 programs / 35 s on
  10 workers; ~6 MB per 1,000 programs. Evaluation sets need `--keep-alternatives`
  (the per-turn grammar), which is most of a sample's size.
- `MIXES["v7"]` in `scripts/generate.py` holds the family weights.
- `scripts/export_sft.py` reads a `.jsonl`, a `.jsonl.gz`, or a shard directory. It needs
  the student server on 8080 for `/apply-template`. (`eval_turns.py` still expects a
  plain `.jsonl` made with `--keep-alternatives`.)
- Marking: every synthesized and hand-written program marks lines; the style is
  chosen per program (`NATLANG_MARK_STYLE=standalone|grouped|en_passant` forces one).

Families: leaves (`judge, classify, extract, crisp_scalar`, 12% undetermined instances
ending in `report_blocker`), `tiny`, fixed shapes (`ticket_report, review_digest,
expense_audit, nested_assessment, map_leaf, map_then_count`), `per_item_condition`,
`fold_with_steps`, `composed` (sampled move by move over 11 domains: leaf over a list,
flags from labels/claims with ==, !=, numeric predicates, and/or, filter with an
if/else block, count/share/group aggregates, folds `count_if`/`tally`/`add_amount`,
repeat-until `halve`/`raise_limit`, empty-input guard, blockers that travel up), and
the seven `cb_*` code bases.

Teacher-side generation (needs Bonsai on 8081):
- `scripts/paraphrase_steps.py` fills `data/phrases.json`, the phrase bank of step
  templates (keeps a variant only if placeholders are identical, it is one line, and
  the judge says it asks for exactly the same step). **Running in the background when
  this was written** (`runs/paraphrase-steps.log`; first variants kept look right, e.g.
  `{out} = [{fn}(x) for x in {over}]`). Only nine step kinds are keyed so
  far (`each, each_with, select, flags_eq, flags_neq, count_true, group_count,
  return_one, return_record`); the other `say()` sites in `synth.py` should get
  `key=`/`fields=` too. Inspect the kept variants by eye before a large run.
- `scripts/teacher_leaves.py --mix v7 --seed S --n N` writes references for leaves
  that generate text (`say`, `page_content`, `submission_page`) to
  `data/leaf_references.jsonl`; without a reference those turns are left out of the
  corpus (`template=True`). First run: 13 of 23 kept. Use the same `--seed`/`--mix`
  as the corpus run so that the keys match.
- `scripts/paraphrase.py` (older): round-trip-verified paraphrases of the leaf
  families' instruction texts, in `data/paraphrases.json`.

Known gaps in the corpus, in rough priority:
1. Glue-code variety is still narrow (the tuned model wrote `!l === "spam"`).
2. No recovery data: rejected call followed by the corrected call; an unexpected
   result (an empty list) followed by the sensible continuation. `PLAN.md` §5.3.
3. Item texts are templates; the teacher should re-render them in many voices with
   the hidden attributes fixed (Y1 in `SYNTHETIC_DATA.md`).
4. More code bases from `TRAINING.md` §1.4: poker bot, probe monitor with budgets,
   semantic spreadsheet, checklist executor, prose-defined API endpoints.
5. Larger inputs (paged reads), two-collection programs (joins, dedup pairs),
   records from more domains than expenses.
6. Conformance program 18 (narrowed legal actions) was simplified; narrowing a call's
   result type is deferred.

## 6. Training, evaluation, supervision

```
scripts/serve.sh &                                                     # student server, for the chat template
.venv/bin/python scripts/export_sft.py data/ref.jsonl data/sft.jsonl [--every K]
docker run --rm --gpus all -v "$PWD:/work" -e HF_HOME=/work/models/hf -e HF_HUB_OFFLINE=1 \
    --name natlang-train natlang-train python scripts/train_lora.py data/sft.jsonl runs/lora-vN --steps 600
docker stop -t 120 natlang-train          # clean stop: finishes the step, writes a checkpoint
# the same `docker run` line continues from runs/lora-vN/checkpoint; --merge-only exports; --fresh starts over
scripts/to_gguf.sh runs/lora-vN/merged models/natlang-350M-vN-Q8_0.gguf
docker stop natlang-llama; scripts/serve.sh natlang-350M-vN-Q8_0.gguf &
.venv/bin/python scripts/eval_turns.py data/ref-eval-vN.jsonl --per-cell 10      # teacher-forced next-turn accuracy
.venv/bin/python scripts/baseline.py --judge-server none                          # whole programs (conformance), graded by checks
```

- Training: LoRA r=32 on all linear layers, bf16 base, gradient checkpointing, one
  sequence at a time, 16 sequences per step, max 3,072 tokens, lr 2e-4 cosine;
  **3.2 GiB, ~6 s per step**. Checkpoints every 25 steps (adapter, optimizer,
  scheduler, step, data cursor, RNG); stop/resume was tested end to end. The data
  order depends on `--seed` only.
- Pairs are rendered by the server's `/apply-template`, i.e. exactly as at inference.
  The template renders nested arguments as JSON (`true`, double quotes); the grammar
  and `parse_calls` accept both Python and JSON literals. Completion-only loss.
- **An evaluation set must come from the same harness version and marking switch as
  the model's training data.** v5/v6 models were trained without marks: evaluate them
  with `NATLANG_MARKS=0` in the environment (`data/ref-eval-v6-nomarks.jsonl` exists).
- Results so far (unseen programs, seed 777, v5 harness): untuned 29% exact next
  turn / 43% right tool; after 15 minutes of LoRA (2,656 pairs) **75% / 88%**;
  `call` turns exact 100 / 60 / 40% (shapes / composed / code bases). Conformance
  (21 programs): untuned 2 correct and never calls; tuned 5 correct, 13 incorrect, 3 judge-ungraded,
  0.1 to 6 s per program (Bonsai: 25 to 640 s, 19/20 on the older suite).
- Failure modes seen in the tuned model's traces (program 23): buggy glue code for
  unseen phrasings; losing its place after a surprising result; imitating its own
  rejected calls; leaf judgments themselves (program 06 was structurally perfect and
  wrong on labels); folds/repeats (rare in v5). The progress lines, marks, more
  folds/repeats and glue variety answer these; none is confirmed by a trained model yet.
- `data/sft-v6.jsonl` (52,938 pairs, no marks) exists; a v6 model was **not** trained
  to completion (the owner redirected effort to data generation). `runs/lora-resume-test`
  is a 12-step throwaway.

Supervising long runs: start them with `run_in_background`, never poll with sleep;
watch `runs/*.log`; check `free -m` (another user process, `node pnpm pack`, held 4 GB
at times: not ours, do not kill), `nvidia-smi`, `docker ps`. The watchdog log is
`runs/bonsai-watch.log`.

## 7. What to do next, in order

1. (done) The phrase bank's first pass finished: 28 kept, reviewed by hand (dropped: list-marker and
   checkbox variants, which collide with the `[ ]` listing; garbled `count_true` lines; `=>` for return);
   `data/phrases.json` is in git. `select` and `flags_neq` got no variants: rerun for them, and key more sites.
2. (done) `export_sft.py` reads shard directories and `.gz`.
3. Key the remaining `say()` sites in `synth.py`; extend glue variety; add recovery
   trajectories (perturb a reference state, record the fix).
4. Generate **v7** (`--mix v7`, tens of thousands of programs) and a v7 evaluation set
   with `--keep-alternatives`; run `teacher_leaves.py` for the same seed first.
5. **Marking-style experiment** (owner's request): (a) teacher: run Bonsai on a few
   code-base programs with each style available (prompt `natlang/prompts/tools_delegate.md`
   needs a paragraph on marking; none of the prompts mention `mark_done` yet) and
   measure whether it marks, whether marks match the reference, whether programs still
   complete; (b) student: train small LoRA variants on corpora generated with
   `NATLANG_MARK_STYLE` forced to each style and on the mixed corpus; compare next-turn
   accuracy, marking accuracy, whole-program results; look for confusion when both
   `mark_done` and `done=` exist. Report to the owner before fixing a style.
6. Train on v7 (resumable), convert, serve, evaluate; read traces
   (`baseline.py --verbose 23`) for new failure modes and fold fixes into the corpus.
7. Update `PLAN.md` §10.2/§10.3 with every measured result; keep docs free of
   superseded designs (the owner asked for removal, not annotation).

## 8. Small things worth knowing

- `Session.apply` is the tool entry point; `Session.act` is the old text notation,
  kept only for `conformance/harness` scripts and `tests/test_canonical_traces.py`.
- `MAX_ACTIONS = 40` per episode (marks are free), `MAX_LOCALS = 16`,
  `MAX_FUNCTIONS = 12` per code base, `MAX_PATHS = 48` per enum in the tool schema
  (locals and named values first, list elements last).
- Short texts (≤ 400 chars, ≤ 8 lines) are shown whole in the workspace listing;
  longer ones say they are cut off. This fixed a real failure (a rubric's first line
  looked complete).
- A record value in a reference `write` must list fields in declared order (the
  grammar asks for it).
- Effects: a function that uses `fx.<cap>.<fn>` declares `effects:`; every caller up
  to the root must declare it too (attenuation). Capabilities are given to `Runtime`.
- `dump_state` / `load_program` round-trip a running lambda with locals, marks and
  its code base (swap-out).
- `scripts/serve_web.py` is a real listening web server over `codebases/webserver`;
  with Bonsai it served correct pages at 4 to 6 minutes per request.


## Review fixes (2026-09-19)

The owner asked to fix the seven review findings and continue the iterative
training workflow. The model remains the interpreter; no deterministic
orchestration was added and evaluation is not a prerequisite for more training.

- Training reserves whole programs for held-out loss, writes `split.json`, and
  verifies corpus/split hashes on resume. Older checkpoints can still be exported
  with `--merge-only`; start a new output directory to train with the new split.
- `eval_turns.py` scans the whole corpus (including gzip/shards), creates/reuses
  a manifest of exact sample IDs and hashes, and reports actions apart from replies.
  Use the same manifest for paired comparisons and a new one for a changed corpus.
- `baseline.py` writes machine-readable results and computed verdict totals.
  The saved historical v5 log contains five correct, thirteen incorrect, three
  unjudged programs; it is not a result for the modified runtime.
- Conformance grading checks `expect.emitted` and `effect_checks`; program 14 now
  requires its actual emitted records as well as its return value.
- Model requests, tokens, elapsed time and all tool calls are bounded. Work
  actions remain capped at 40; marks spend the separate 128-call budget.
- Attached `done` ranges are fully validated before writes/calls. Failed operations
  are shown to the model, never silently retried or refunded.
- Effectful JS runs in a killable worker with a host capability bridge. Host hooks
  must manage cancellation/idempotency if they can outlive a timeout.
- TypeScript annotations work with Node >=22.13; plain JS needs only QuickJS.
  Static snippet type checking remains unimplemented and is no longer claimed.
  Grouping/counting/indexing helpers now support all string keys.
