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

The text writer is the immediate path (host writes the returned text as the writer's native state). Native writing
must not stay off (coordinator correction 2026-11-03). Why the close marker (125034) gets no effective learning:

- The marker rows are learnable: learned sparse `input_rows` (tied output), FP32 masters, AdamW at the learned-row
  rate (model/shared_state.py; native optimizer routing). Routing is not the defect.
- The objective almost never targets them. Producer nodes built by `data/provider_static_graph.py:97` (teacher text
  encoders) and `data/provider_native_graph.py:163,352` (writer nodes) carry `target_weights` all 0.0, open, body and
  close included: the state is exported but its authoring is never supervised. Inline producing spans inside a
  supervised action target and `data/operator_graph.py` producers (seed + close at weight 1) are the only CE on the
  close marker; the operator rows are new and small, and `learned_operators.seed_weighted` anneals the close marker
  together with the seed toward 0.1. Measured outcome: close log-prob -18 to -35 at the writer site (writer61
  replay), full67 native writes 0/4 closed live.
- Census of the actual corrected v44b mix (3,000-row stride sample under its CE policy, `close_census.py`, unit
  `memetic-census-1`): counts of close/open/body tokens with positive weight; result recorded in the fallow register
  C6 when it lands.

Plan (data and objective, anneal text to native; fallow register C6/C9):

1. **Writer as a learned operator.** Register `guidance` in `model/learned_operators.py` (site: the rendered
   `writeGuidance` call with its consumer and facts, the writer's prompt text-initialized in the real call) and
   convert verified guidance into operator graphs with `data/operator_graph.py`: producer = site + `produce_open` +
   seed + `produce_close`, consumers = the program's invocations importing the span, text-cut sibling as baseline.
   Seeds: the text writer's own guidance from JOLT/memetic episodes whose consumer then succeeded (checked), else the
   mechanical exemplar (section 5) with seed provenance. No new name the model sees.
2. **Close marker never anneals.** The seed schedule scales seed body tokens only; the closing marker keeps weight 1
   (a schedule-version change recorded as an objective transition, not a silent reinterpretation of /1).
3. **Supervise producer close in existing graphs.** Give the close marker of writer-authored producing spans in the
   provider graph converters CE weight 1 (body stays 0 where it is host text) in the next source revision, preserving
   old row bindings' exposure through the ordinary append/correction mechanism.
4. **Gate and anneal.** Re-measure close log-prob at the writer site on the writer61 replay set after the operator
   rows earn exposure; when the native writer closes spans live (target >= 90% on ~48 samples), switch `writeGuidance`
   to `Neuralese<string>` with template write while the seed weight anneals 1 -> 0.1 and consumer-outcome learning
   carries the span (span basis). The text path remains the fallback per call when a span does not close.

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
- Production successor recipe (rebased on the live lineage, coordinator correction 2026-11-03):
  `/home/werg/data/lfm3-rl-dgx-20261029/ops/memetic-successor-v44b-1103/recipe-runtime.json` = control
  `ops/corrected-mix-full67-w1-v44b-1103` recipe + `interleaved_jolt_rl_v1` (task source: selection pool
  `selection-pool-20261103T205416Z`) + `memetic_improve_v1` (`memetic_every` 4, `programs_per_round` 2, `group` 4).
  It loads with `load_recipe`. Adoption requires a runtime frozen from main at or after 95104262d and the nine
  launch root causes of `LFM3_DILOCO_CORRECTED_MIX_20261103.md` applied to cloud seeding; it is an RL-stage
  transition for the controller at the next window, not a change to the running SFT window.
- Remaining blockers (fallow register M1-M8): automatic per-parent build inside the controller (until then the stage
  is inert: the controller records `interleave.memetic.pending`); adapter-capable actor for `fit` and the bank
  baseline; automated actual/shuffled/none ablation and promotion/distillation wiring; per-program tool-surface edits;
  scoped guidance routing (scopes are headings in one guidance state every call imports); evaluation noise (4 worlds x
  2 samples); live 48-sample measurement of the writer and supervisor prompts.

## 9. First production episodes measure

Per memetic round: supervisor completion rate (committed), changes proposed per kind, evaluations used, support gains
claimed vs query gains realized (overfitting to support), query gain over both baselines, outcome spread per group,
defects reported, and episode wall time. Promotion: held development gain of the candidate checkpoint vs the published
parent; confirmation once. Nothing here claims capability lift until those held outcomes exist.

## 10. End-to-end check on the real model

See section 11.

## 11. Independent review and end-to-end status (2026-11-03 late)

An independent reviewer agent checked the integrated work against the four owner criteria. Fixed (95104262d):

- B1: memetic rows went through `privileged_context.case_facts`, which would read the episode's `expected: None` as a
  fact; memetic rows now get no privileged facts (JOLT facts for supervisors are not designed yet).
- B2: programs were keyed by family, mixing different program sources; a program is now `family@sha12` of its
  exact source files, and splits only group source-identical worlds.
- B3: concurrent samples of one episode shared one workspace directory; each run now gets its own
  `runs/<run>` directory (`open(spec, runId)` through the world bridge).
- M1: admission used dev successes; it now uses support only (dev/confirmation stay unseen).
- M2: an episode that ended without a scored workspace dropped out of its group; it now scores 0 against the
  group's anchor (`unscored_workspace`).
- M3: `fit` was offered although no adapter actor exists; it is withheld from the declared service (constants kept
  for the adapter path, M2 in the register).
- M5, M8, M9: skills go to the function's companion folder; repeated identical evaluations are free (cache) and
  count once; baselines and episodes are namespaced by parent checkpoint; `bind_round` no longer overrides the task
  curriculum floor and writes no round records; protocol lines from the world process are guarded.

Open (register): B4 (no build in the controller, stage inert without a manual build), M4 scoped routing, M6 noise,
M7 promotion/distillation/ablation unwired; minor: the leak guard covers string/dict expectations only, the
repetition metric is coarse.

End-to-end (criterion 4) is **not yet demonstrated**. Units `memetic-e2e-build-4` (family `ext_decisions_desk`,
support 2, query 2, samples 2; built before the B2 fix, so mechanics only) and `memetic-e2e-episodes-1` (collect two
samples, then `compile_replay` and `prepare_objective`) run against the shared DGX actor on :8270 (bf95 `71c3eb45`),
which serves the full67 rollout collection at about 3.5 minutes per request. Outputs:
`/mnt/external/natlang-memetic-20261103/e2e-v1/`. No episode has completed, so there is no evidence yet that the
supervisor acts, edits apply, the reward is computed and the update replays on the real model, and no capability
claim of any kind.

## 12. Closing the open items (2026-11-04)

Coordinator follow-up: the open M items meant the stage would not yet train meta-improvement. What changed:

| Item | Now | Commits |
| --- | --- | --- |
| M1 controller build | A due memetic round builds the parent's task source itself (crisp, model-free, 2 s on the 6,665-row production pool: 66 programs from 21 families, 5 held). Evidence and baselines are prepared once per program and parent inside the round's actor lease **before** collection (`prepare_round`), so the episodes' time is spent improving. | 9991751d5, 55a10bfb8, 99a438662 |
| M3 ablation | Actual vs shuffled (another program's committed or default guidance) vs none on the paired query worlds for every committed guidance; recorded in the outcome. | 9991751d5 |
| M5 distillation | `after_collection` (in the lease, after collection) writes `distill/<parent>/manifest.json`: episodes beating their anchor, and the improved programs' verified query runs lowered to guidance producer/consumer SFT graphs. Appending them to the SFT source is the remaining wiring. | 9991751d5 |
| M7 promotion | Every `promotion_every` memetic rounds the held programs are improved by the current parent and scored on development worlds; the paired gain is compared with the previous evaluated parent (`promotion/history.json`). The verdict does not yet gate the lineage's parent. | 9991751d5 |
| M8 noise | Query reserves up to 12 worlds; cost-sized (cheap: every world, >= 24 rollouts; expensive: 4 worlds x up to 4 samples); candidate, baselines and ablation share worlds and seeds; identical change sets are measured once per group; binomial noise recorded with the anchor. | 9991751d5 |
| M6 scoped guidance | One native state per scope: program-wide for every call, function binding (program + function part) only for that function's calls (ts-host per-call selection); input-class guidance becomes a root skill whose description is the class (progressive disclosure). | 7644f3dd8, 9991751d5 |
| M4 tool surface | `withhold:` frontmatter (runtime refusal with a teaching error) and the workspace `withhold` change kind. | 7644f3dd8, c026f8ac6 |
| M2 adapters | The actor already serves per-request bank adapters; `memetic_fit` fits the bank on the parent's verified support successes; `fit` is offered and baseline (b) includes the bank default fit when `NATLANG_BANK_SERVICE` is set. No real fit has run yet. | b03b17f77, 9991751d5 |
| Pool dedupe | The old rule compared program text plus a 4 KB input prefix, collapsing all 22 typo-repair codebases into one world; it now compares world data with the corpus MinHash rule. | 9991751d5, 55a10bfb8 |

Runtime: production cut over to v45 (bundle at b03b17f77) before these commits; the memetic stage adopts at the next
runtime cut from main >= the commits above (successor recipe `ops/memetic-successor-v44b-1103`, README updated).

Native writing (C6, steps 1-3 implemented by a helper, reviewed here): `guidance` learned operator (prompt rows on the
real `writeGuidance` call site, seed cap 512) and its case converter (`data/guidance_cases.py`,
`scripts/neuralese_data/build_guidance_cases.py`); operator seed schedule /2 keeps the close marker at CE weight 1
while the seed body anneals (/1 unchanged, recorded transition); provider converters supervise the producer close
marker under a new conversion identity (old rows keep their bindings and exposure) — 9c51e2f72, f148b34c6, 933b1d188.
Supply: **zero** checked text-writer guidance records exist (JOLT ran `writer: off`). The interleaved stage now
declares `guidance_stage: backbone | writer` (419b222f2) and the successor recipe sets `writer`, which is what produces
them. The close-marker census of the v44b mix is still unmeasured (uncached index of one compressed source stalls it).

Findings from the live runs (anomalies treated as bugs):
- The first post-fix e2e (typo-repair codebases, 900 s per case) produced 4/4 **unscored** support rollouts (wall-clock
  timeouts on the shared, loaded actor; evidence persistence on the external disk took 5-11 s per record). The stage
  had measured that as 0% success. Fixed (2f1cf523d): unscored runs are no measurement; they neither admit nor
  exclude a program. E2E moved to NVMe output, a reading program, 1800 s per case, on a dedicated low-priority actor
  (:8271, bf95 `71c3eb45`) after the shared :8270 actor went away mid-run.
- Writer probe with a 768-token turn cap: 0/3 (truncated prose). With production settings (no turn cap), samples split
  into fast completions (about 1-1.5 min, input-specific, fact-grounded guidance that states the verified result, as
  writeGuidance step 2 asks) and runaway runs that think aloud in 4,096-token visible-text turns without calling
  `return_result`. Per-call wall clock now 600 s; 48-sample result below.
- A memetic episode's world process runs a nested production collection (4 workers): the e2e unit was OOM-killed at a
  4 GiB budget and needs about 12 GiB. A controller running memetic rounds must budget this per concurrent episode.
- The GB10 actor generates about 45 tokens/s per stream; one reading-task rollout takes 10-20 minutes, so one memetic
  episode (preparation + supervisor + evaluations + query) takes hours on the DGX. Production memetic rounds belong on
  the cloud actors; the DGX e2e is a mechanics check only.
