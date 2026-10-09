# Program improver: decomposition, part by part

Status: implemented on main 2026-10-09 (plans/OWNER_REVIEW.md, review after the fact). What differs from the draft below:

- Owner decision 2026-10-10: the tool-economy sentences of the old `rewriteProgram.nl` (use the given source files directly, inspect only clipped parts, finish once the edit is supported, write each replacement once) were restored as positive per-stage steps in `editSource`, `editSourceStructural`, `hypothesize` and `diagnose` (the shared system prompt is unchanged). Measure in the teacher window whether they cut tool calls per experiment, repeated reads and reprinted evidence.
- One experiment is the crisp `improveStep/lifecycle.ts` (question 1 answered): `folder.iterateOn`, `folder.propose` and
  `folder.apply` take a host or TypeScript function `(folder, ...args)` as a directory reducer (a general language feature, not an
  improver special case; spec/ext/directory-reducers.md, skills/natlang-authoring/references/patterns.md). It runs on a private
  transaction copy of the folder, which is installed when it returns and discarded when it throws, exactly as for a `.nl`
  reducer; a `.nl` that is not a `kind: directory-reducer` is still refused. `improveProgram` applies
  `improveStep.lifecycle.step(folder, evaluator, plans, state, policy)` with the evaluator and the `plans` journal held by the host,
  so no model call sequences an experiment and no `.nl` stage can reach the evaluator. `improveStep.nl` remains only as the typed
  root of the callable folder `improveStep/` (the project compiler recognizes a callable folder by a `.nl` beside it; a TypeScript
  module with a folder of the same name is a plain project folder there, and its relative imports would need `.js` extensions and its
  `.nl` imports generated declarations); it is never sent to a model. A step that fails because a natural-language policy
  chose outside its crisp bound is `incomplete-search`, not an infrastructure failure (the authored `PolicyBoundError`).
- The stages are `diagnose`, `hypothesize`, `editSource` (instruction) and `editSourceStructural` (structural). The doc's
  `finish` is the host: the edit returns `{summary, preserves}` and the host derives the changed paths from the diff.
- Refinement types are crisp verifiers in `improveStep/crisp.ts` (and `ts-host/src/optimization/policies.ts`) rather than
  `Is<...>` types, which are judged by a model; the closed `Disposition` union and typed `lastExperiment` are in `types.ts`.
- The shared GEPA module is `ts-host/src/gepa`, reached from the application as the platform module `natlang:gepa`
  (callable folders may import only siblings and packages). It is a module rather than a `gepa` service, so the crisp
  defaults run in the application and in the engine from the same code.
- `chooseComponents` reads, per eligible key, the parent's covering training cases and how many failed (the feedback the
  doc counted is produced after the keys are chosen). `findOpportunity = none` ends the search (`no-opportunity`).
- `transformations.ts` is the data table (a TypeScript module, since callable folders import siblings); the names are the
  keys, so `policy.transformation: 'repairProgram'` replaces the former string substitution of reducer files.
- The counterexample loop's stop is the shared `shouldStop` policy (same slot, same `policy.policies.shouldStop` setting, same
  `improveStep/shouldStop.nl` with a second list of steps selected by `facts.search`): the crisp default is the prose rule as
  written (no admitted example, the repaired training suite passing, the oracle allowance exhausted, a repair that did not
  complete), `ts-host/src/improvement/counterexample-stop.ts`. The host measures the round's facts and the step calls
  `counterexamples.stop()`; the model no longer reports them. The one `suggestCounterexamples` returns `{inputs, reason}`.
- `rewriteComponents` is a TypeScript directory reducer around the natural-language `editComponents.nl`
  (`componentSearchStep/rewriteComponents.ts`): the candidate is the components.json the edit leaves, the constraints (only
  the selected keys differ; kinds, slot ids and segment counts are unchanged; segments are an array of strings) are checked
  exactly by `componentProblem` in the shared module `natlang:gepa`, which the engine's `search.check` applies again, and one
  retry carries the problem text in `request.problem`. The `Is<Candidate, ...>` type of the first draft is this check. The
  "return that exact object" instruction and the constraint paragraph are gone from the `.nl`. (`componentSearchStep.nl`, the
  step of the component engine, is still model-run; it is not part of this move.)
- Training export: `ts-host/scripts/self-improvement/export-followup-training.mjs` and `scripts/skills/export-training.mjs` key on
  the definition source a trace records (table and shared helpers in `scripts/self-improvement/improver-stages.mjs`): the
  current stages (`diagnose`, `hypothesize`, `editSource`, `editSourceStructural`, the natural-language policies,
  `suggestCounterexamples`) and, for runs recorded earlier, the single `rewriteProgram` editor with its model-run step. Each row
  carries `improver_stage {stage, generation, definition_source, shape, mode}` and a `supervision` block (whole-trajectory: prompts,
  instructions and inputs trained, tool output and mechanical feedback after the first reply at the lower feedback weight,
  nothing masked; the classes mirror `serve/grad.py _context_weights`). With no step invocation to record the outcome, a stage is
  admitted from the run's own state history: an edit by the entry whose candidate is its resulting source, a diagnosis or
  hypothesis by the edit of its experiment (linked by the values passed on, in order), a policy by its replay and the verifier
  that bounded it (a shadow disagreement keeps it as context). Definitions recorded but not exported are listed in the
  manifest's `omitted` with the reason. `captureExactRewriteIO` now captures every stage's exact arguments and result.

The text below is the design as drafted.

This is the natlang program that improves natlang programs (plans/NATLANG_PROGRAM_IMPROVEMENT.md). The host entry is
`improveProgram` (`ts-host/src/improvement/program.ts:24`, re-exported by `main.ts:3`); the authored program in this
folder is bundled into `ts-host/src/improvement/authored-source.ts` by `bundle-improver.mjs` and enters the run
identity (`program.ts:32`). So every change here changes the identity of resumable runs, which is intended
("newest code always").

The review (plans/NATLANG_NATIVE_REVIEW.md, P4 and the model-facing cleanups) found:

- The search policy (which parent, which component, compose or edit, when to stop, which opportunity) is TypeScript
  in this folder (`parents.ts`, `selection.ts`, `population.ts`, `context.ts`) and in the host
  (`authored-engine.ts:39-59`), implemented twice.
- `improveStep.nl`, `componentSearchStep.nl` and `counterexampleStep.nl` are model-written control flow over fixed
  sequences of service calls that the host re-verifies anyway (`authored-engine.ts:75-79`, `program.ts:70-80`).
- `rewriteProgram.nl` is one call of about 600 words that diagnoses, chooses a hypothesis, edits and runs the finish
  protocol.

Decisions: **fn**, **inline**, **implicit**, **crisp**, **service**, **host**, **data**, **pluggable**.

## Policy

- **Natural language: the semantic decisions of an experiment.** What the evidence shows (`diagnose`), what to try
  (`hypothesize`), the edit (`editSource`, `rewriteComponents`), counterexample suggestions, and, as pluggables,
  `findOpportunity`, `chooseParent`, `chooseMove`, `chooseComponents`, `selectIncumbent` and `shouldStop`.
- **Crisp: exact acceptance and exact math.** Evaluation and scoring (`SourceEvaluator`), source policy and
  compilation checks, the paired-improvement re-check, state assembly, the journal, the budget gateway. The
  selection math (frontier, seeded draws, pruning, "better") exists once.
- **Natural-language choices are bounded by crisp verifiers.** A parent is a population member; a component key is
  eligible and closed under dependencies; the selected incumbent is among the best-quality members; a stop below the
  declared `maxExperiments` is allowed, a continue past it is not. The crisp default of every pluggable is today's
  behavior.
- **A plan is journaled.** The plan of an experiment (parent, move, keys, minibatch, who chose) is written to the
  journal before the edit, so a resumed run reuses it. Crisp plans were reproducible from the seeded RNG
  (`authored-engine.ts:32`); natural-language plans are not.
- **Termination is structural.** The measure is `maxExperiments - iteration` (`program.ts:69`); the stop decision
  can only end earlier.
- **State model.** An experiment decides on a snapshot (population, evidence) and applies a pure state update
  (`finishStep`, `population.ts:6-10`). Derived values form a DAG: evidence, opportunity, diagnosis, hypothesis,
  edit, check, measurement, acceptance, selection, state.

## Parts

### Source search (`improveStep/`)

| Part | Decision | Unit | Why |
| --- | --- | --- | --- |
| One experiment: evaluate baseline, pick parent, test, advance | crisp pipeline | `lifecycle.step`, `lifecycle.ts:9-17` | Fixed sequence. Today the model is asked to run it in `eval` (`improveStep.nl:8`); see question 1. |
| Baseline state from the first measurement | crisp | `start`, `baseline.ts:4-6` | State assembly. |
| Capability types (`Snapshot`, `ExperimentEvaluator`, ...) | crisp | `capabilities.ts:1-8` | Types for the services. |
| Training evidence for the parent, never validation | crisp | `inspect`, `feedback.ts:5-9` | Exact data access. |
| Source files of a snapshot | crisp | `experiment.ts:12-13` | Reads. |
| Opportunity: fixture, quality, efficiency, source-size, none | pluggable | `findOpportunity`, today `context.ts:38-44` | A judgment about what to try. Today only `fixture` is used (`experiment.ts:11`); the other kinds never reach a model, and `none` does not stop the run. |
| Parent choice | pluggable | `chooseParent`, today `parents.ts:3-13` | GEPA frontier draw is the crisp default. |
| Diagnose: what the evidence shows | fn | `diagnose` (exists as `diagnose.nl`, not called today) | The first stage of `rewriteProgram.nl`. |
| Hypothesize: what edit to try and where | fn | `hypothesize` | The second stage of `rewriteProgram.nl`. |
| Edit the source folder | fn (directory reducer) | `editSource` | The third stage. The transformation reducers specialize it. |
| Finish: changed paths from the diff, summary, preserves | crisp | `bookkeeping.finish`, `rewriteProgram/bookkeeping.ts:4-7` | The host derives `changed`; the model supplies `summary` and `preserves`. |
| Empty diff: no supported hypothesis | crisp | `experiment.ts:15` | Typed disposition `no-hypothesis` (today a magic string compared at `population.ts:20`). |
| Duplicate of an earlier candidate | crisp | `experiment.ts` (new) | Candidate digest equal to a history entry is rejected without evaluation, like `plan.duplicateIds` for components (`authored-engine.ts:58`). Replaces "without repeating a rejected hypothesis". |
| Compile and source-policy check | service | `evaluator.check`, `experiment.ts:18-19` | Exact. |
| Train and validation measurement | service | `evaluator.evaluate`, `experiment.ts:8, 20-21` | Exact scoring. |
| Improvement predicate: quality, then source size or model calls | crisp | `experiment.ts:22-25`, one `meanBetter` | Exact acceptance; see the shared implementation. |
| Gates passed, accepted or rejected with reason text | crisp | `experiment.ts:29-31` | Exact. |
| Brief: the compact source and evidence card | crisp | `brief`, `context.ts:9-35` | Deterministic rendering. Clip sizes (700, 400, 900, 1200, 6000 characters) become named settings; instruction sentences inside it move to the `.nl` (`context.ts:10`, `16`). |
| Add accepted member, select, prune, install, re-measure | crisp | `advance`, `population.ts:13-24` | Exact state transition. |
| Select incumbent: best quality, objective tie-break, baseline-preserving | pluggable | `selectIncumbent`, today `best`, `selection.ts:3-10` | Tie-breaking among best-quality members is a policy; crisp bound: the choice is among the best-quality members. |
| Prune population to `maxPopulation` keeping baseline, selected, frontier | crisp | `prune`, `selection.ts:11-20` | Exact; shared with the component engine. |
| Stop: objective satisfied, no hypothesis, fixture error, experiment limit | pluggable | `shouldStop`, today `population.ts:19-23` | A policy; `maxExperiments` stays a crisp bound. |
| Assemble the next `SearchState` | crisp | `finishStep`, `population.ts:6-10` | Pure. |

### Component search (`componentSearchStep*`)

| Part | Decision | Unit | Why |
| --- | --- | --- | --- |
| One experiment sequence: plan, merge or propose, check, mini-batch compare, full evaluation, finish | crisp pipeline | the host services `search.*`, `authored-engine.ts:39-97` | The model "owns experiment sequencing" (`componentSearchStep.nl:8`) and every step is re-verified by the host. |
| Coverage check: components covered by training evidence | crisp | `authored-engine.ts:42-44` | Exact. |
| Parent draw from the frontier | pluggable | `chooseParent` (shared with source search) | See above. |
| Component choice: bandit-like `ComponentSelector.pick` and update group | pluggable | `chooseComponents`; `ComponentSelector`, `getUpdateGroup` (`vendor/ax-gepa`) | Crisp default is the vendored selector; natural language picks from the feedback; crisp closes the set under dependencies. |
| Compose or edit: `iteration % 4 === 3` and more than one member | pluggable | `chooseMove`, today `authored-engine.ts:53` | A magic schedule; natural language can compose when two members changed disjoint components. |
| Three-way merge of two candidates | crisp | `mergeCandidates`, `strategies/gepa.ts:16-24` | Exact; conflicts give null. |
| Minibatch draw | crisp | `authored-engine.ts:51, 55` | Seeded shuffle. |
| Edit the selected components | fn (directory reducer) | `rewriteComponents` | The semantic edit. |
| Unselected components unchanged, kinds and slot ids kept | crisp | `check`, `authored-engine.ts:63`; `validateCandidate` | Exact verifier of the edit. |
| Paired minibatch improvement re-check, duplicate check | crisp | `finish`, `authored-engine.ts:77-80` | Exact acceptance. |
| Guidance coverage after a guidance change | crisp | `authored-engine.ts:83-85` | Exact. |
| Prune, incumbent, selector snapshot, events | crisp | `authored-engine.ts:86-96` | Exact. |

### Counterexample search (`counterexampleStep*`)

| Part | Decision | Unit | Why |
| --- | --- | --- | --- |
| One round sequence: evidence, suggest, admit, repair, select, record | crisp pipeline | services `counterexamples.*`, `counterexample-program.ts` | Same pattern as above; the oracle and repair are services. |
| Suggest counterexample inputs | fn | `suggestCounterexamples` | The semantic step. Two copies exist: `counterexampleStep/suggestCounterexamples.nl` (inputs only) and `reducers/suggestCounterexamples.nl` (inputs and reason, `maxSuggestions` typed). One remains. |
| Admit against the independent oracle | service | `counterexamples.admit` | Exact; the oracle is outside the improver. |
| Repair on the new suite | service | `counterexamples.repair` | Runs the source search. |
| Stop: repaired suite passes, checks exhausted, no admitted example | pluggable | `shouldStop` (counterexample form) | Same policy slot. |

### Transformation reducers (`reducers/`)

| Part | Decision | Unit | Why |
| --- | --- | --- | --- |
| Seven reducers with the same two closing paragraphs and different first sentences | data plus one fn | `editSource` with a `transformation` instruction from `transformations.json` | `workflows.ts:6-10` already substitutes a reducer as `improveStep/rewriteProgram.nl` by string replacement of type names. A table of instructions removes the replacement and the repeated boilerplate (`reducers/*.nl:8-10`). |

## Splitting `rewriteProgram.nl`

Where each sentence of the 13-line file goes (paragraphs at lines 7, 9, 11, 13):

| Sentences (line) | Destination |
| --- | --- |
| Choose one coherent, evidenced hypothesis (7) | `hypothesize` |
| `request.hypothesis` is a caller hint (7) | removed; `lifecycle.step` always passes the empty string (`lifecycle.ts:15`), so the field is dead (`context.ts:4`, `9`, `11`, `types.ts:6`) |
| You are editing source; `folder`, `allowedFiles`, the goal, runtime-folder outputs (7) | `editSource` framing (and `context.ts:10`, which repeats it) |
| `history` and `lastExperiment`; use failed actions to revise; do not repeat a rejected hypothesis (7, 11) | `hypothesize` for the content; the crisp duplicate-candidate check for the enforcement |
| Use the brief; inspect only clipped parts; never reprint the evidence (7) | runtime: paged evidence view (`EvidenceView`, as `rewriteComponents` already gets) and tool prompt text; removed here |
| `sourceFiles` already holds the source; `allowedFiles` may name new paths (7) | `editSource` steps 1-2 |
| `eval` intermediate and `finish:true` protocol (7, 13) | runtime tool prompt; removed here |
| Do not call `nl`, `folder.apply`, `folder.propose` or an evaluator (7) | capability scope: the evaluator service is scoped to the step (`program.ts:60`); removed here |
| Structural mode and instruction mode (9) | two variants of `editSource`, selected by `policy.mode`; the mode rule is enforced by `sourcePolicy` (`program.ts:39`) |
| Keep judgments semantic; never substitute keywords, known answers or holdout data (9) | `editSource` step 4 (positive); holdout data is not visible to the improver (`feedback.ts:4`) |
| "including negation and sarcasm" (9) | removed pending measurement; it is a task-specific hint in a general instruction |
| Ground service fields in `evidence.serviceDeclarations`; explain the failure the change addresses (9) | `editSource` step 3; the `summary` type |
| Repair execution mistakes: unavailable filesystem API (9) | `diagnose` category `execution-method`, `hypothesize` rule |
| Generalize; paths and counts are runtime data (9) | `editSource` step 4 |
| Reorganize; inspect helper behavior; root reducer judges a batch; keep helpers for arithmetic (11) | `hypothesize` structural options; `editSource` structural variant |
| After a repair passes, examine efficiency (11, 13) | `findOpportunity` kind `efficiency`, then `diagnose` |
| `write_file`, `edit_file`; write each replacement once (13) | runtime tool prompt |
| Remove contradictory and duplicated guidance (13) | `editSource` instruction variant, step 3 |
| Finish with `bookkeeping.finish(...)` (13) | `finish` (crisp); the host computes `changed` |
| If no useful change is supported, leave the folder unchanged (13) | `hypothesize` returns `none`; disposition `no-hypothesis` without an edit call |

## Natural-language functions, step by step

All contracts below are new or revised; names follow the folder.

### `diagnose`

```
args: evidence: EvidenceView (paged training rows), lastExperiment: ExperimentFeedback | null,
      history: ExperimentOutcome[], goal: string, opportunity: Opportunity
returns: { observations: { case_id: string, what_happened: string, expected: string,
                           category: "wrong-result" | "execution-method" | "wasted-action" | "fixture" }[],
           pattern: Is<string, "one sentence naming what the observations share"> }
```

1. Read the rows of `evidence` for the cases the opportunity names (failed cases for `quality`; passing cases with
   more than one model request for `efficiency`; all passing cases for `source-size`).
2. For each row write one observation: the case id, what the program did (result, files, actions), what the goal
   expects, and a category. Use `execution-method` when the trace calls something that is unavailable;
   `wasted-action` when a model request or tool call did not change the result.
3. When `lastExperiment` is present, add what it changed and what its measurements show.
4. Write `pattern`: one sentence on what several observations have in common.

### `hypothesize`

```
args: diagnosis: Diagnosis, sourceFiles: SourceFile[], mode: "instruction" | "structural",
      allowedFiles: string[], history: ExperimentOutcome[]
returns: { kind: "instruction" | "helper" | "structure" | "efficiency" | "none",
           statement: Is<string, "names the observed failure or wasted action it addresses">,
           files: Is<string[], "a subset of allowedFiles">,
           predicted_change: string }
```

1. Read `diagnosis.pattern`, then the source of the files named by the observations.
2. Choose the smallest edit that explains the pattern: a rewritten instruction (`instruction`); a corrected helper
   such as a join, a file pattern or a trimmed edit string (`helper`); a reorganization, for example a root reducer
   that judges a small batch and does the exact bookkeeping in one final eval (`structure`, structural mode only); or
   removal of a model request or tool call that the results show to be unnecessary (`efficiency`).
3. For an `execution-method` observation, choose the edit that uses the ordinary file tools or scoped folder
   handles.
4. Compare with `history`: choose an edit that differs from every rejected outcome in kind or files.
5. Return `files` within `allowedFiles` and write `predicted_change` as the measurement you expect to move.
6. When the evidence supports no edit, return kind `none`.

### `editSource` (directory reducer; `instruction` and `structural` variants)

```
args: request: { hypothesis: Hypothesis, goal, mode, allowedFiles, sourceFiles, serviceDeclarations,
                 transformation: string }   // one sentence from transformations.json
returns: { summary: Is<string, "names the observed failure or wasted action the change addresses">,
           preserves: string[] }
```

1. Read `request.sourceFiles`; they hold the current text of every existing path.
2. Make the edit `hypothesis` describes in the files it names. Create a new path with `write_file` when
   `allowedFiles` lists one.
3. Ground each service field and enum meaning you use in `serviceDeclarations`.
4. Express each judgment as an instruction that reads the input's meaning, and write exact code only for counting,
   parsing and arithmetic. Write paths and counts as runtime data. In instruction mode, change prose only and remove
   contradictory or duplicated guidance.
5. Return `summary` and `preserves` (the external behaviors the edit keeps).

The host computes `changed` from the diff (`bookkeeping.ts:5-6`); an unchanged folder is the `no-hypothesis`
disposition.

### `rewriteComponents` (component search; existing)

The existing function (`rewriteComponents.nl`) stays one edit task. Its contract changes in one place: the
returned `Candidate` is read by the host from `components.json`, so the instruction "then return that exact object
... The returned value and edited file must agree" (line 11) is removed. The constraints "Edit only request.keys.
Keep every kind, slotIds, unselected component, and the number of segments unchanged" (line 9) become the return type
`Is<Candidate, "only the selected keys differ from the parent; kinds, slot ids and segment counts are unchanged">`,
already checked by `authored-engine.ts:63` and `validateCandidate`.

### `suggestCounterexamples` (existing, one copy)

```
args: request: { goal, contract, evidence, maxSuggestions }
returns: { inputs: unknown[][], reason: string }
```

1. Read the contract and the training failures.
2. Name the boundary or interaction the failures suggest is untested.
3. Write up to `maxSuggestions` argument lists that exercise it, and one sentence of `reason`.

### Pluggable policies

Each has a crisp default equal to today's behavior and a natural-language side; the verifier is crisp.

`findOpportunity(policy, evidence) -> { kind, reason }`

1. When any row has `failureKind` `fixture`, return `fixture`.
2. When any row failed, return `quality`.
3. When `objective` is `model-calls` and a passing row used more than one model request, return `efficiency`.
4. When `objective` is `source-size`, return `source-size`.
5. Otherwise return `none`. (The crisp default is exactly `opportunity`, `context.ts:38-44`. The new behavior is that
   `none` ends the search with reason "no observed failure or cost opportunity".)

`chooseParent(population, seed) -> id`

1. For each case id, find the highest score in the population and collect the members that hold it (the frontier).
2. Prefer the frontier member that has been a parent least often; break ties by id.
3. Return its id. The crisp default is the seeded draw over the frontier list (`parents.ts:3-13`); the verifier
   checks membership in the population.

`chooseMove(population, iteration) -> "edit" | "compose"`

1. When the population has fewer than two members, return "edit".
2. When two members differ from the baseline in disjoint component sets, return "compose".
3. Otherwise return "edit". (Crisp default: compose when `iteration % 4 === 3` and the population has more than
   one member, `authored-engine.ts:53`.)

`chooseComponents(feedback, eligible, history) -> keys`

1. For each eligible key count the feedback rows that cite it as the cause.
2. Return the key with the most citations; break ties by fewest past proposals. Crisp then closes the set under
   dependencies (`getUpdateGroup`).

`selectIncumbent(population, incumbent, objective) -> id`

1. Keep the members with the highest validation quality.
2. Under `source-size`, keep those with the smallest source; under `model-calls`, those with the fewest requests.
3. Return `incumbent` when it remains; otherwise the remaining member with the smallest id. (This is `best`,
   `selection.ts:3-10`; the natural-language side may weigh other costs among the remaining members only.)

`shouldStop(last, state, policy) -> { stop, reason }`

1. When `last.disposition` is `fixture-error`, stop with its text.
2. When `objective` is `quality` and the selected quality is 1, stop: "Declared objective satisfied."
3. When `last.disposition` is `no-hypothesis`, stop: "No further evidenced change."
4. When `state.iteration + 1` reaches `policy.maxExperiments`, stop: "Declared experiments completed."
5. Otherwise continue. (Crisp default is `population.ts:19-23`. The natural-language side may also stop on a stall:
   the last three experiments were rejected for the same failing cases.)

## The shared GEPA implementation

Today selection logic exists in several places that diverge in small ways:

| Concept | Locations |
| --- | --- |
| Per-case frontier (winners per case) | `optimization/strategies/gepa.ts:6-14` (candidates with `validation.results`), `applications/program-improver/improveStep/parents.ts:4-9` (members with `scores`), `selection.ts:16-19` (again, for pruning), `authored-engine.ts:87` (calls the first) |
| Seeded xorshift draw | `authored-engine.ts:32` (stateful `state.rng`), `parents.ts:11` (stateless from a seed) |
| "Is this candidate better": quality, then a tie-break | `strategies/gepa.ts:31-45` (`meanBetter` with `tieBreak: baseline, modelCalls, latency, cost` and selection limits), `experiment.ts:22-25` (`objective` quality, source-size, model-calls), `selection.ts:3-10` |
| Prune a population | `selection.ts:11-20` (sort and slice), `authored-engine.ts:87-89` (remove one) |
| Parent for an experiment | `parents.ts:3-13`, `authored-engine.ts:45-46` |

Proposal: one module, `ts-host/src/optimization/strategies/gepa.ts`, generic over
`Member = { id: string, scores: { caseId: string, quality: number }[], cost?: number, modelCalls?: number }`:

- `frontier(members)`: winners per case, sorted.
- `draw(seed)`: one xorshift step, returning `{ value, next }`.
- `better(a, b, objective)`: eligibility limits, then quality, then the objective's tie-break; `objective: "source-size"`
  adds `sourceBytes` to the existing tie-break set.
- `prune(members, protectedIds, limit)`: frontier-aware, removes lowest-ranked first; both engines call it.
- `mergeCandidates` stays (components only).

It is exposed to the authored program as a service (`gepa`), declared in `EVALUATOR_DECLARATION`'s neighbor, so the
crisp defaults of the pluggables run host-side and the bundle in `authored-source.ts` shrinks by `parents.ts`,
`selection.ts` and the duplicated parts of `population.ts`. The component engine's adapters map
`SearchCandidate.validation.results` to `scores`. The vendored selector (`ComponentSelector`, `getUpdateGroup`) stays
vendored; it is the crisp default of `chooseComponents`.

## Refinement candidates

| Slot | Proposed type | Check |
| --- | --- | --- |
| `Hypothesis.files` | `Is<string[], "a subset of allowedFiles">` | crisp |
| `Hypothesis.statement` | `Is<string, "names the observed failure or wasted action it addresses">` | judged |
| `RewriteResult.summary` | the same predicate | judged |
| `RewriteResult.changed` | host-derived from the diff | crisp |
| `Diagnosis.observations[].case_id` | `Is<string, "the id of a row in the evidence">` | crisp |
| `ExperimentOutcome.reason` | a closed `Disposition` union: `accepted, rejected, no-hypothesis, fixture-error, invalid-candidate, duplicate` plus a text field | crisp; replaces `population.ts:20-21` string comparisons |
| `SearchState.lastExperiment` | `ExperimentFeedback \| null` (today `unknown`, `types.ts:3`) | crisp |
| `Candidate` (component edit) | `Is<Candidate, "only the selected keys differ from the parent; kinds, slot ids and segment counts are unchanged">` | crisp |
| `chooseParent` result | `Is<string, "the id of a population member">` | crisp |
| `chooseComponents` result | `Is<string[], "eligible keys, closed under their dependencies">` | crisp |
| `selectIncumbent` result | `Is<string, "the id of a member with the highest validation quality">` | crisp |
| `shouldStop` | `Is<StopDecision, "stop is true when iteration + 1 reaches maxExperiments">` | crisp |
| `suggestCounterexamples.inputs` | `Is<unknown[][], "at most maxSuggestions argument lists">` | crisp |
| `SearchState.quality` | `Is<number, "equals the independently measured validation quality of the incumbent">` | crisp (`program.ts:77`) |
| `PopulationMember.source` | `Is<string, "a digest present in the journal">` | crisp |
| `RewriteRequest.hypothesis` | removed (dead field) | n/a |
| Training evidence text (`error`, `value`, `modelTrace`) | `Untrusted<string>` | crisp marking |

## Model-facing changes needing live measurement

The program improver's own quality is measurable by running the fixtures in
`ts-host/test/improvement-steps.test.mjs` and `folder-improvement.test.mjs` on the live improver: accepted
improvements per experiment, rejections by reason, and tokens per experiment.

1. **`rewriteProgram.nl` split** into `diagnose`, `hypothesize`, `editSource`, `finish`. Compare accepted
   experiments and cost per accepted experiment. Hypothesis: smaller tasks help the small improver; cost per
   experiment rises (three calls) and rejections for invalid edits fall.
2. **Removal of the tool and finish protocol sentences** from the `.nl` (they live in the runtime tool prompt).
   Measure finish-protocol errors before and after.
3. **Removal of "negation and sarcasm"** (9). Measure rejection rate on the sentiment-style fixtures.
4. **Positive rewrites of guards:** "Do not call nl, folder.apply ..." (removed; scoped), "never substitute keywords,
   known answers or holdout data" (step 4), "do not repeat that candidate unchanged" (crisp duplicate check), "do not
   create the example output in this draft" (framing in `editSource` step 2).
5. **`brief` with instruction sentences removed** (`context.ts:10`, `16`).
6. **One shared `suggestCounterexamples`** with a `reason` field.
7. **Natural-language pluggables** are measured by shadow comparison with their crisp defaults on recorded searches,
   not by live sampling alone. `findOpportunity = none` ending the search is a behavior change: confirm on the
   efficiency and source-size fixtures.

## Questions for the owner

1. (Answered 2026-10-10.) The experiment is a crisp function that calls the natural-language stages: `iterateOn` accepts a host or TypeScript function as a directory reducer (see the status block). No `.nl` step remains.
2. The seven reducers become rows of a data table. Training data and tests refer to their names
   (`workflows.ts:14-17`, `reducers/*.nl`). Keep the names as table keys and aliases?
3. Natural-language `shouldStop` with a stall rule is new behavior, not a relocation. Include it in the first move
   or only after shadow data?
4. The 200-record page size and the clip lengths in `brief` are fixed constants. Settings with their reasons, or
   leave until a problem shows up?
