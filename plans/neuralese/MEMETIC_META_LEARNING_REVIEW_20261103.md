# Memetic meta-learning: review, writer root cause, design and recipe stage (2026-11-03)

Owner direction (2026-11-03 evening): the memetic path is the meta-learning technique; review it, improve it
significantly, integrate it into training as a declared stage with all ramifications, and reuse the bank machinery.
Owner addendum: the data, approach and natlang-native implementation, and the expressivity of the supervisor's output,
must be right from the start, and real-model episodes must run end to end before the stage is called ready.

Code: `applications/memetic-improver/` (supervisor, [DECOMPOSITION.md](../../applications/memetic-improver/DECOMPOSITION.md)),
`training/neuralese/natlang_neuralese/rl/memetic_workspace.py` (workspace service),
`rl/memetic_stage.py` (pool, splits, curriculum, baselines, episodes, scheduling, promotion, distillation),
recipe kind `native_memetic_rl` (`train/recipe.py`), controller hook (`rl/interleave.py`), scoring
(`rl/task_rollouts.py memetic_outcome`), advantages (`rl/native_replay.py group_episode_advantages`), transport
(`ts-host/src/teacher/world-bridge.ts` kind `memetic`). Tests: `tests/neuralese/test_memetic_stage.py`,
`ts-host/test/training-guidance.test.mjs`.

## 1. What existed and what worked

| Piece | State before today | Verdict |
| --- | --- | --- |
| `rl/memetic_program.py`, `memetic_batch.py` (MEMETIC_NATIVE_PROGRAM.md) | A crisp hill-climb: a Python prompt string asks the served model for guidance from a failure digest, `write-text` makes a state, the production collector measures dev/confirmation, `common.acceptance` gates. Never run on LFM3 with lift. | Sound evaluation and custody pieces (write-text state, program-guidance binding, paired seeds, acceptance ledger, SFT export of producer+consumer graphs) are reused. The author is **not** a natlang program, its calls are not captured, so it cannot be trained: there is no meta-learning in it. Retired as the main loop; its helpers are the evaluation mechanism. |
| Guidance writer (`applications/guidance-writer/writeGuidance.nl`, `native/training-guidance.ts`, `rl/privileged_context.py`) | Shared JOLT+/memetic writer; every native write since checkpoint 56 stalled. | **Root-caused and fixed (section 2).** |
| `applications/program-improver/` + `ts-host/src/improvement/` | Teacher-era source improver (diagnose / hypothesize / editSource, GEPA, SourceEvaluator, skill authoring). Not trained, not on LFM3. | Its decomposition (NL stages bounded by crisp verifiers, journaled plans, structural stop) is the template; its host evaluator runs ImprovementCases, not RL `natlang.program/2` worlds, so the memetic workspace evaluates through the production collector instead. |
| Bank machinery (`tune/learned/meta_bank.py`, `bank_outcomes.py`, `bank_stage.py`) | PCA bank published at step 5 (`nbank1:20c4524f...`), coefficient fit, held-family outcome harness, pools. | Reused: near-duplicate rule (pool dedupe), held-family split design, outcome promotion shape, and the coefficient fit as the `fit` operation. Not the main line. |
| JOLT+ privileged context, whole-graph RL, interleaved RL stage, selector /2 | Live mechanisms. | Memetic episodes ride on them unchanged. |

## 2. The writer: root cause and fix (first gate)

**Symptom.** Every native guidance write on checkpoints 56, 61, 75 and (measured today) full67 stopped with
`neuralese-native-state-open: generation stopped inside a produce span`; repair turns degenerated into reasoning without
a tool call until the budget ran out.

**Root cause, measured.**
1. The writer used `readout: template` with a `Neuralese<string>` result: the engine forces `return_result` and opens a
   producer span, and the model must sample the body and the dedicated close token `<|/neuralese_produce|>` (125034).
   The recorded logit replay of writer61 (`ops/writer61-logit-replay-1101/result.json`) puts the close token at
   log-probability -18 to -35 (rank 80k-110k) at every body position; the model emits a few special tokens
   (125020-125024) and then EOS. The close token was essentially never a supervised target: SFT exposure to producer
   bodies ending in that token is a handful of rows (writer75 audit: "zero direct producer-body CE"). Native producer
   writing is an unlearned capability, not a writer bug.
2. Text alone was not enough. Live on full67 (actor :8270, greedy, 2026-11-03,
   scratch `wprobe/`): native write 0/4 turns; `Neuralese<string>` with a forced text value echoed the type name
   (`"Neuralese<string>"`); `string` with a forced value echoed the consumer's task text; `Neuralese<string>` without a
   template deliberated about the type for 4×768 tokens and never returned; **`string` without a template returned
   correct numbered guidance in 3 turns (1/1; the second probe was an actor-queue timeout)**.
3. The writer's `consumer.text` was the consumer's whole request including the ~5k-token system prompt, doubling the
   writer prompt.

**Fix (b069f6061).** `writeGuidance` returns `string` in an ordinary call; `TrainingGuidance` writes the exact returned
text as the native state (`typed-text-result` producer span, bound to the writer's final model turn, matched by the
consumer's call id so concurrent consumers do not mix). This is the span basis rule: the state starts from the model's
own input-specific text and morphs through consumer-outcome learning (whole-graph RL; the JOLT learner already resolves
text writers, `privileged_scoring.writer_span` + `jolt.text_owner`). The writer reads the consumer's own request text
without the shared system prompt. Native producer-span writing remains a separate learnable target (section 6).

## 3. Weak points that would have made the memetic path fail

1. **No trainable policy.** The old author was a Python prompt; nothing in an "improve P" decision was a captured,
   rewarded model action. Fixed: the supervisor is a natlang program whose episodes are RL tasks.
2. **Free-text output.** The old author emitted `<guidance>…</guidance>` text the harness had to parse. Fixed: typed
   `Change` values validated before they get an id (section 4).
3. **No in-episode measurement or stop.** The model could not try, measure and refine. Fixed: `evaluate`, `ablate`,
   `iterateOn` with a structural bound and a stated stop rule.
4. **Rewarding the trivial thing.** Group-mean centering alone rewards "no change" whenever peers are worse. Fixed:
   advantage centered on max(group mean, best trivial baseline).
5. **No headroom control.** Programs the parent always or never solves give no signal. Fixed: curriculum admission
   on parent success in [0.1, 0.85].
6. **Leakage.** The author must never see query/dev/confirmation worlds or gold. Fixed by construction (the workspace
   only serves support evidence; `score` is host-only and hidden from the program) plus a verbatim-answer guard.
7. **Writer stall** (section 2).
8. **Nothing measured whether consumers use guidance.** Addressed by the ablation protocol (section 5.4).
9. **Fallow risk.** A prepared stage without a launch owner. The stage binds into the interleaved controller and the
   fallow register carries its remaining blockers with an owner (section 8).

## 4. Action space (owner criterion 1)

The supervisor acts only through typed calls (`workspace` service declaration in `rl/memetic_workspace.py`):

- **Guidance**: `{kind:"guidance", scope:{kind:"program"} | {kind:"function", name} | {kind:"inputs", when}, text}`.
  Stored as one program-guidance text composed scope by scope, written as a native state (span basis), delivered
  through the runtime's program-guidance channel; ids make every change diff-able, `revert` undoes one.
- **Skills**: `{kind:"skill", function, name, description, body, action: create|revise|retire}` placed at
  `<folder of that function>/skills/<name>/SKILL.md`, the program's own skills folder the runtime binds.
- **Instruction and signature patches**: `{kind:"instructions", file, find, replace}` (exact, once) and
  `{kind:"signature", file, field: args|returns, name?, type}`; every changed program is loaded by the natlang loader
  before the change gets an id; problems come back as the tool's answer.
- **Adaptation**: `fit({worlds?, components?, steps?, scope?, init?})` calls the bank coefficient fit and returns a
  change id usable in `evaluate`; fits are evaluated before commit. Unavailable runs say so explicitly (section 8).
- **Evaluate and iterate**: `evaluate(ids)` (support worlds only, paired seeds), `ablate(ids)` (per-change
  attribution), `budget()`, and `refine.iterateOn(...).withLimit({maxSteps: budget.evaluations}).until(done)`.
- **Combination**: change sets combine kinds; `ablate` attributes gains; `commit(ids, reason)` ends the episode.
- **Defects**: `reportDefect(text)` records a harness/oracle defect without reward.

**Checked against real failures.** A separate agent hand-analyzed every failing full67-era invocation available
(15 model failures, 12 distinct modes; fewer than the requested 20 exist: the quality-parent round was 20/20 and the
full67 pilot is still collecting; analysis in this session's scratch `failure-edit-analysis.md`, summarized here):

| Failure mode (count) | Hand-written fix | Kind | Expressible |
| --- | --- | --- | --- |
| contract_diagnosis computes the total inline, never `edit_code`s the helper (4) | numbered steps: compare helper to its doc comment, `edit_code` the disagreeing span, then compute through the helper | instructions (+guidance) | yes |
| repair_missing_child: writes the file then never returns / verifies via `delegate`, bash pseudo-calls (4) | program guidance "write, diff, return; no delegate"; numbered procedure skill | guidance + skill | yes (tool removal would be stronger: not expressible yet) |
| repair_missing_child: never commits a write, re-reads files to the budget (2) | procedure skill with a write-first step | skill | yes |
| repair_missing_child: wrong second-parameter type/name (4 type, 4 name) | guidance on reading the caller's positional arguments | guidance | yes (name is an oracle defect H2) |
| edit_stream: retyped the document, skipped the move (1) | skill template (iterateOn over edits with nl per edit) | skill + guidance | yes |
| edit_stream: appended `\nnull` read from eval rendering (1) | harness fix (H1) | defect | via `reportDefect` |
| repair_ts_syntax: misdiagnosed the error (1) | skill with a parse helper | skill (+ crisp parse tool) | partly |

Extensions the analysis asked for: tool-surface edits per program (not in the runtime yet; register entry), a parse
helper usable from skills (register entry), compaction notes as an editable target (deferred), and a defect channel
(added: `reportDefect`). Harness/oracle defects found (H1 eval `console:` rendering shows a bare `null`; H2
repair_missing_child oracle demands an unstated parameter name; H3 edit_stream `:hinted` requirement not enforced; H4
bash `2>/dev/null` and Pyodide file access; H5 completion budget not clamped to context; H6 the "full67" pilot is
bound to bf95 `71c3eb45`; H7 call-store "database is locked") are reported to the owners through coord.

## 5. Data, approach and rewards (owner criteria 2 and 3)

1. **Programs.** P = an RL task family of the admitted pool (each family is one natlang program template; its tasks
   are invocation worlds): the selection pool's families, the authored task batches and composed codebases enter
   through the same pool file. Related worlds (shared source group or curriculum split group) form one component and
   stay on one side; near-duplicate worlds (inputs plus root text, difflib ≥ 0.9, the bank pool's rule) are dropped;
   protected-related worlds never enter.
2. **Splits.** Meta-training programs: support (seen as evidence) and query (reward, never shown). Held programs (a
   seeded 20% of families, never meta-trained): support, development, confirmation. Promotion compares development
   gains; confirmation is spent once through `common.acceptance`.
3. **Curriculum.** Admission while the parent's success on support and query lies in [0.1, 0.85]; re-evaluated per
   parent. Repetition is measured per batch (`repetition`: families, max episodes per family, mean brief
   similarity); the episode generator is one row per (program, parent), so diversity is the pool's family diversity.
4. **Baselines and reward.** Per (program, parent): (a) no change, (b) the default crisp improvement: the bank's
   default coefficient fit when an adapter-capable bank service is configured, else mechanical exemplar guidance from
   a verified successful support attempt (deterministic, input-specific). Quality = committed set's query success minus
   a bounded budget cost (≤ 0.05; another 0.05 when the episode never commits). Advantage = quality − max(group mean,
   best baseline): a candidate that does not beat the trivial improvement never earns a positive advantage. Groups of
   4 samples per program give spread; graded rewards make the selector's spread signal natural.
5. **Leakage.** Supervisor episodes run as students: no JOLT teacher facts exist for memetic rows (no reference).
   Support evidence never includes expected answers; query worlds never leave the workspace; `score` is host-only; a
   verbatim query-answer copy scores zero.
6. **Whole-graph updates.** Episodes are ordinary captured RL rollouts; any Neuralese the supervisor passes between its
   own calls is part of its graph. The guidance states written during evaluation are consumed in separate collector
   runs; those consumer runs distil (section 7) with their producer span resolved
   (`memetic_program.guidance_training_graph`), so consumer use trains the producer in SFT, not by detaching it in RL.
7. **Consumers actually use guidance.** Promotion and periodic audits evaluate a committed guidance set as actual vs
   shuffled (another program's committed guidance) vs none on held development worlds; a guidance arm that ablation
   cannot distinguish from shuffled is a defect (Neuralese span basis). Not yet automated in the stage (section 8).

## 6. Writer and native writing after the fix

The text writer is the immediate path. Native producer-span writing becomes learnable through SFT rows whose targets
include producer bodies and the close token: the text writer's own spans (`[produce_open] text [produce_close]`) are
exactly such rows once distilled. Re-measure the close-token log-probability after those rows earn exposure; switch a
writer to a native template write only when it closes spans live.

## 7. Distillation

Successful improvement episodes (positive anchored gain, committed set) are the supervisor's RFT/SFT rows
(`memetic_stage.distillation_rows`); the improved programs' verified query runs with their guidance import become
producer+consumer SFT graphs through `memetic_program.guidance_training_graph`, admitted by the ordinary source
append with exact actor provenance. Committed change sets for a program are its stored improvement (deployment
binding through the existing program-guidance binding; instruction/skill patches as program source edits).

## 8. Recipe integration and what remains

- Recipe kind `native_memetic_rl` (headless, controller-run) with `rl_stage` (the interleaved stage it rides on),
  `memetic_every`, `programs_per_round`, `group`, `settings`, `promotion_min_gain`; input `program_source`. The
  interleaved controller runs a memetic round every `memetic_every` rounds from `<controller out>/memetic/stage-<parent>.json`
  (built by `python -m natlang_neuralese.rl.memetic_stage build --out <controller out>/memetic --endpoint <actor>`);
  without it the controller records `interleave.memetic.pending` and keeps ordinary rounds.
- Remaining blockers (fallow register M1-M5): automatic per-parent build inside the controller; adapter-capable actor
  for `fit` and the bank baseline; automated actual/shuffled/none ablation; per-program tool-surface edits; live
  48-sample measurement of the writer and supervisor prompts.

## 9. First production episodes measure

Per memetic round: supervisor completion rate (committed), changes proposed per kind, evaluations used, support gains
claimed vs query gains realized (overfitting to support), query gain over both baselines, outcome spread per group,
defects reported, and episode wall time. Promotion: held development gain of the candidate checkpoint vs the published
parent; confirmation once. Nothing here claims capability lift until those held outcomes exist.

## 10. End-to-end check on the real model

See section 11 (filled from the live run).
