# Audit & First-Principles Analysis: The Language Side of natlang (2026-11-22)

**Parent:** [Campaign Index](README.md)  
**v2 (2026-11-23):** the census and evidence below are the original audit's and are kept. Facts re-verified against main 0f87d11a4 are corrected in place, marked *(v2)*. §5-§6 conclusions are rewritten for the campaign's v2 direction.  
**Purpose:** Complete quantitative and qualitative evidence from the 2026-11-22 repository audit across examples, applications, codebases, browser playground, documentation, skills, website, compiler/runtime source, and training/RL task pipelines — plus the first-principles reasoning behind the [Semantic Functional Programming Campaign](README.md).

---

## 1. Executive Summary

The repository's language runtime is **far ahead** of its examples, docs, and training data. Under the hood in `ts-host/src/`, natlang already implements:
- Hindley–Milner-style local typed-hole inference for inline `nl` (`compiler/holes.ts`);
- Lexical closures in prose with live mutable capture cells (`compiler/inline.ts`);
- Contravariant/covariant function subtyping (`native/types.ts:519-520`);
- Content-addressed curried contexts with structural interface rebinding `fn.in(context)` (`runtime/contexts.ts:219-300`);
- Transactional, branching `Folder` algebras (`propose`, `accept`, `snapshot`, `branch`, `select`, `at`, `folder.iterateOn`);
- Refinement and trust types `Is<T, P>` and `Untrusted<T>` (`native/refinement.ts`, `native/untrusted.ts`);
- Calibrated probability-scoring `readout: decision` (`native/decision.ts`, `model/decision-scoring.ts`);
- `pluggable({ crisp, nl, shadow })` (`runtime/pluggable.ts`);
- Trace-guided JIT specialization (`calls/specializer.ts`);
- Algebraic `natlang:neuralese` combinators (`map`, `zip`, `ap`, `combine`, `empty`, `split`, `read`, `ask`) + law-based rewrites (`compiler/rewrites.ts`) + `natlang:learning` (`grad`, `valueAndGrad`, `selfDistill`).

Yet our audit of **2,062 `.nl` files**, **712 TS/JS files**, **38 playground examples**, **~37,323 `grpo-pool` RL tasks**, and the live SFT mix established that **almost none of this paradigm is actually practiced, taught, or trained**. Instead, the repo split into two opposite failure modes:

1. **The Timid Front Door:** `natlang init`, `website/`, `skills/`, and the playground present natlang as *"ordinary TypeScript with occasional narrow ticket classifiers."*
2. **Prose-as-Slow-Interpreter & Cue-Word Operations Desks:** Bulk agent-authored apps (83% of all `.nl` files) write 200–779 word imperative TypeScript algorithms in English prose with zero `.ts` helpers; meanwhile, RL authoring rounds 10–23 converged on cue-word/substring checklists to survive exact-oracle + literal blind-solve admission.

---

## 2. Quantitative Census of Examples, Applications, Codebases & Playground

Comment-stripped scan excluding `dist/`, `.natlang/`, `node_modules/`, and generated `.d.nl.ts` (2026-11-22):

| Metric | `examples/` | `applications/` (39 apps; *(v2)* 42 directories incl. `build/`, `dist/`, `design-briefs-host/`) | `codebases/` excl. `design_briefs` (17) | `codebases/design_briefs/` (100) | `ts-host/playground/examples/` (38) | `ts-host/examples/` | **Total** |
|---|---:|---:|---:|---:|---:|---:|---:|
| `.nl` files | 12 | 592 | 68 | 1,390 | 0 *(15 embedded)* | 0 *(3 inline)* | **2,062** *(+18)* |
| TS/JS code files | 20 | 446 | 100 | 140 | 4 *(23 crisp)* | 2 | **712** |
| `.nz` files / `natlang.d/` dirs | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | **0 / 0** |
| `.nl` median / mean / max words | 23 / 29 / 101 | **195 / 210 / 779** | 53 / 62 / 190 | 55 / 81 / 548 | ~25 | — | — |
| `.nl` ≥ 150 words | 0 (0%) | **436 (73.6%)** | 4 (5.9%) | 145 (10.4%) | 0 | — | **585 (28.4%)** |
| `.nl` leaf (no companion dir) | 8 | 391 | 39 | 977 | 10 | — | **1,415 (68.6%)** |
| Companion dir: `.nl` + `.ts` mixed | 2 | **19 / 201 (9.5%)** | **29 / 29 (100%)** | **25 / 413 (6.1%)** | 2 | — | **75 / 647 (11.6%)** |
| Companion dir: `.nl`-only | 1 | **176 / 201 (87.6%)** | 0 | **378 / 413 (91.5%)** | 3 | — | **555 / 647 (85.8%)** |
| Inline `` nl`...` `` in TS/JS | **3** | **0** *(48 comment hits)* | **0** | **0** | **0** | **1** | **4 / 712 (0.56%)** |
| `nl.with(...)` in TS/JS | **0** | **0** *(27 prose in `.nl`)* | **0** | **0** | **0** | **0** | **0** |
| Higher-order `.nl` sigs (`=>`) | **0** | **0** | **0** | **0** | **0** | **0** | **0 / 2,062 (0%)** |
| Context rebinding `.in(...)` | **0** | **0** | **0** | **0** | **0** | **0** | **0 (0%)** |
| `iterateOn` in TS/JS | 1 | 1 (`withMeasure`: 1) | 0 | 0 | 0 | 0 | **2** |
| `iterateOn` in `.nl` prose | 0 | 45 | 2 | 0 | 0 | 0 | **47** |
| `readout: decision` | 2 | 15 | 0 | 81 | 0 | 0 | **98 (4.8%)** |
| `decide(...)` calls (`.nl` / TS) | 0 / 0 | 6 / 3 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | **6 / 3** |
| `kind: directory-reducer` | 1 | 37 *(21 Astra, 9 nldb, 6 prog-imp, 2 spec)* | 0 | 0 | 0 | 0 | **38 (1.8%)** |
| `Is<T, P>` refinements (`.nl` / TS `types.ts`) | 0 / 0 | 0 / 9 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | **0 / 9** |
| `Untrusted<T>` (`.nl` / TS) | 3 / 0 | 11 / 24 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | **14 / 24** |
| `Neuralese<T>` type (`.nl` / TS) | 0 / 0 | 0 / 3 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | **0 / 3** |
| `pluggable(...)` in TS | 0 | 31 | 0 | 0 | 0 | 0 | **31** |
| `uses:` / `generic:` / `withhold:` in `.nl` | 0 / 0 / 0 | 39 / 0 / 0 | 0 / 0 / 0 | 21 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | **60 / 0 / 0** |
| `edit_code` / `diff_code` usage | 0 | 0 | 0 | 0 | 0 | 0 | **0** |

### Structural Skews Hidden in the Totals
- **87 / 100** `codebases/design_briefs/` projects have **zero executable `.ts` helpers** (only `types.ts`).
- **11 / 39** `applications/` (the "Astra" suite: `cave-mission`, `dialogue-perspective-ledger`, `field-gloss-workbench`, `polyphonic-edition-workshop`, `protocol-meaning-debugger`, `saga-forensics`, `seed-lineage-laboratory`, `semantic-procedure-compiler`, `treaty-obligation-ledger`, `turn-window-adjudicator`, `witness-rewrite-studio`) hold **321 of 592 `.nl` files (54.2%)**; 9 of those 11 have only `types.ts`, zero `readout: decision`, and zero crisp `.ts` helpers.
- Combined, `design_briefs/` + Astra = **1,711 / 2,062 `.nl` files (83.0%)** — overwhelming any agent or human grepping the repo for examples.
- *(v2)* These corpora come with exact behavioural checks outside their directories: `training/semantic_applications/design_briefs/<NNN>/cases.json` (800 exact-oracle cases, 8 per brief, one `split_group` per brief), and for the 11 Astra/Sol apps `training/semantic_applications/{astra,sol}/<app>/project.json` with a fixture world, oracle and `verify-result.mjs`. So they can be migrated and checked behaviourally (WS7) rather than quarantined. None is in a `grpo-pool` entry. All 11 Astra/Sol apps and ~365 brief paths are pinned by the `grpo-sd-teacher` manifest `sft-teacher-qwen-october-20261120`.

---

## 3. Taxonomy of Observed Anti-Patterns

1. **Imperative TypeScript & Deterministic Plumbing Written in Prose ("English-as-a-Slow-Interpreter"):**
   - `applications/pi/generation/answer.nl:11-26`: line-by-line English transcription of deterministic `pi-durable generation.ts` op construction and retry branching — zero semantic judgment.
   - `applications/cave-mission/planMission/survey/readFrames/advanceFrameUnit.nl:1-67` (779 words): 11 numbered paragraphs of exact set unions, ID membership checks, projection copying, and literal `eval({code:"return next;",finish:true})` tool syntax (L64).
   - `applications/logs/investigate.nl:19-47`: arithmetic (`item.occurred_at + settings.quiet_ms`), array slicing, and threshold comparisons written in `.nl` prose rather than a 30-line `.ts` orchestrator.
   - `codebases/design_briefs/003-handover-continuity-manager/planDepartureHandover.nl` & `100-immersive-theatre-stage-manager/controlPerformance.nl` (548 words): rigid sequential JSON-threading pipelines with zero `.ts` helpers.
   - Classic `codebases/{spell_arena,dependency_plan,semantic_merge}`: top-level orchestrators written as Python-like pseudocode in `.nl` burning a full model call for 2–4 lines of deterministic glue.

2. **Rubric-Gamed / Cargo-Cult Feature Usage (Astra Suite):**
   - Synthetic rubric (*"Visible TypeScript contains declarations only"*, `applications/treaty-obligation-ledger/DECOMPOSITION.md:5`; *"Merely constructing a literal earns no contribution credit"*, L91) caused 27 `.nl` files to spawn redundant inline `nl.with<...>` calls performing the *exact same judgment* the `.nl` function was just invoked to do (`judgeObservationScope.nl:14-16`, `challengeAuthority.nl:11-12`).
   - `applications/field-gloss-workbench/analyzeCorpus/translateHeldOut.nl:12`: uses `iterateOn` with a decreasing `pending.length` cursor to perform a plain independent `Promise.all(targets.map(...))` over a fixed array.

3. **Pure Deterministic Computing Wrapped in an LLM:**
   - `applications/compilers/compiler.nl:1-55` + 29 leaf `.nl` files (`c/parse.nl`, `opt/mem2reg.nl`, `aarch64/allocate.nl`, etc.): attempts an entire optimizing compiler in prose with zero deterministic AST/SSA/register-allocator code.
   - `ts-host/playground/examples/natural.mjs:150-174, 195-219` (`release-review`, `migration-plan`) and `examples/npm_app/main.nl:1-7`: trivial pass-through wrappers around 1-line deterministic helpers (`return tests_passed;`, `is-number`).

4. **Browser Playground Disconnected from Natlang's Value Proposition:**
   - `ts-host/playground/examples/crisp.mjs`: **21 of 38** example entries are pure-TypeScript LeetCode utilities *(v2: they are test fixtures, not shown in the public gallery: `examples.mjs:4`; the gallery is the natural and interface examples)* (`balanced-brackets`, `shortest-route`, `merge-intervals`, `form-validation`) with **zero** `.nl` functions; line 114 (`while (pending.size)`) even violates natlang's own finite-iteration rule.
   - `ts-host/playground/examples/natural.mjs`: all 13 examples use archaic pre-v0.5 phrasing (`"Write the number 7 to return"`), 0 `readout: decision`, 0 inline `nl`, 0 `iterateOn`, 0 `Untrusted<T>`, 0 `Is<T, P>`.
   - `ts-host/playground/examples/interfaces.mjs:37-53, 79-97`: regenerates an entire hardcoded JSON DOM tree through an LLM on every button click.

5. *(v2)* **Our own guidance taught the prose-algorithm style.** The authoring skill says "the author (you) puts the algorithm ... in the source, and the interpreter only has to take the next step" (`SKILL.md:8-11`) and "Port an algorithm as its steps over named data" (`SKILL.md:104-107`). The generated AGENTS.md and the `natlang new` stub say "Number the steps" (`package/authoring.ts:123,138`). AGENTS.md's rule ("spell algorithms out as numbered steps over named data for small models") is about *policy*. Its next rule is "policy is natural language; mechanism is crisp". The front door dropped that distinction, so authors numbered mechanism too. The fix is in the guidance (WS4 §4 item 3), not only in the corpora.

6. **Best Existing Exemplars (The Positive Proof):**
   - `examples/triage/triage.ts:1-38`: TS exact bookkeeping + parallel `.nl` leaves + inline `nl` lambda passed into `iterateOn(...).withLimit({ maxSteps: 3 }).until(exactWordCheck)`.
   - `ts-host/examples/browser-board/program/board.ts:1-78`: inline `nl` returning discriminated unions (`Decision`, `ViewPlan`) checked by exact TS structural invariants.
   - `examples/review/`: 3-level callable folder mixing `Untrusted<string>`, leaf `readout: decision` (`verdict.nl`), concurrent fan-out, and exact TS aggregation (`summarize.ts`).
   - `applications/notebook/index.ts:1-295`: real SQLite/VM workspace, `iterateOn` with genuine structural measure (`remainingCells`), `pluggable({ crisp, nl })`, `untrusted()` tainting, exact `id@revision` citation verification with bounded feedback retry.
   - `applications/helpdesk/index.ts:1-320`: event-sourced `KeyedEventLoop` with exact state transitions + async semantic `.nl` projections + `pluggable({ crisp, nl, shadow })`.
   - `applications/nldb/index.ts:1-187`: atomic `Folder` overlay transactions (`folder.apply`) + SQLite compiling NL queries to exact SQL while caching semantic `readout: decision` predicates (`sqlite/judge.nl`).
   - `applications/program-improver/improveStep/{lifecycle.ts,experiment.ts}`: crisp experiment mechanics (`Folder` branching/proposing/accepting, Pareto selection) separated cleanly from semantic `diagnose`/`hypothesize`/`editor`.
   - `applications/wiki/merge/block.nl`, `applications/workflow/{types.ts,refinements.ts}`, `applications/failure-explainer/types.ts`, `applications/memetic-improver/`, `codebases/{nlprolog,type_studio}`.

---

## 4. Training Data, RL Pool & Authoring Pipeline Audit

### 4.1 Pool Composition (`grpo-pool`, ~37,323 rows across 18 corpora)
- **89.7%** (~33,467 rows) comes from two early corpora (`memetic-program-source-20261105` + `natlang-rl-bgkit-kr-20261103`; `TRAINING_DATA_DISCOVERABILITY_20261120.md:22-24`).
- Base 33,164-row breakdown (`TASK_SUPPLY_20261117.md:17-41`): **43.0%** Reading/QA, **20.5%** static codebase Q&A/syntax typo repair (`natcode.py`/`codebase.py`, partially excluded by `grpo.py:5651`), **11.7%** math/puzzles/logic, **8.2%** decision desks & 180 design-brief gates, **6.8%** Haiku-authored R1–R7 operations desks, **4.5%** tables/SQL/finance QA, **4.4%** procedural inline curriculum, **1.0%** single-call tool selection.
- Recent additions (~3,856 rows, ~10.3%): 908 tau-bench/APIGen-MT retail/airline rows (1 repeated program `serve_customer.nl`), 447 R8–R9 desks, **1,765 R10–R23 Haiku-authored tasks** targeting `held-v3` construct shares (`TASK_SUPPLY_20261118.md:36-375`), and 736 `git_history` Q&A rows.

### 4.2 Feature-by-Feature Coverage Gaps in Training Data

| Semantic-FP Capability | Training Data Coverage | Root Cause |
|---|---|---|
| Inline `` nl`...` `` lambdas inside `eval` | **~0% in RL** (0 occurrences in `scripts/rl_tasks/` & `RL_TASK_AUTHORING_KIT.md`; 0.9% required in Nov 3 RL); **2.8%** legacy SFT | Kit omits inline `nl`; `authored.py:157` sets `inline: "optional"`. |
| `iterateOn` with NL stopping predicate | **0% in RL and SFT** | Every `iterateOn` uses a deterministic cursor/flag predicate; `conformance/neuralese/cases/termination-learning-graph.yaml` (`iterate-nl-predicate-unbounded`) is `status: pending`. |
| Context rebinding `.in(ctx)` | **0% everywhere** | Not in authoring kit, not in `intrinsics.ts`, not in skills. |
| `Is<T, P>` refinement types | **0% in RL, 0% in SFT, 0% in `design_briefs/`** | Broken for inline `nl` (`targets.ts:166-172`), strictly unary (`refinement.ts:94-97`), omitted from authoring kit. |
| `edit_code` / runtime malleability | **0% in `scripts/rl_tasks/`** | Repair tasks (`natcode.py`) use `write_file` on folders instead. |
| `readout: decision` in SFT | **0% in SFT warmup** (`SFT_WARMUP_MIX_20261118.md:192`) | "Teachers cannot score options" — no logprob/soft-target distillation built yet; `decisions.py:20` drops probability-floor abstention for lack of exact gold. |
| Real `read_code` discovery of semantic libraries | **~0% for codebases** (~1.5% `skills/`) | RL prompts spell out exact helper call syntax in root instructions (`RL_TASK_AUTHORING_KIT.md:92-93`); `natcode.py` and `codebase_ops.py` never call `.nl` functions; `sol_programs.json` and `training/semantic_applications/` have 0 references in `grpo-pool`. |
| Higher-order semantic functions | **0% everywhere** | No polymorphism, untyped lambda runtime check, aggressive `nl-not-called`. |

### 4.3 The "Exact-Oracle + Literal Blind-Solve" Cue-Word Trap
In `TASK_SUPPLY_20261118.md` (rounds 10–23), tasks are written by Haiku workers and filtered by an independent Sonnet blind-solver (`scripts/rl_tasks/blind_solve.py:54-62`) plus exact-match deep equality (`authored.py:42`). Any task with two defensible readings is rejected.
- **Observed consequence:** Free-text span seeds failed blind-solve at massive rates (Qasper 3/23, MuSiQue 11/20 → R15 declared *"closed-set answers only"*, `TASK_SUPPLY_20261118.md:161-163`). Rejections clustered around brittle cue-word disputes (`"freeze"` vs `"freezes"`, `"Outdoor seat"` vs outdoor cue, `"not a complaint"` containing cue word, `"injur"` fragment, exact trailing newlines; lines 17, 122, 143, 194, 362–364).
- **Selection pressure:** Authors learned that the fastest way to pass literal blind-solve + exact oracle is to write **deterministic keyword/cue-matching rules in prose** inside `.nl` helpers and wrap them in folder-sorting "operations desks." This actively trains the model *away* from real semantic reasoning and *toward* keyword-checklist compliance.

---

## 5. Mental Model, Half-Wired Superpowers & Structural Ceilings

### 5.1 Front-Door Mental Model Bias
- `ts-host/src/package/authoring.ts:5-24, 79-125` (`natlang init` + generated `AGENTS.md`): 1-file 3-label billing/technical/other classifier; *"Exact work belongs in TypeScript; judgment belongs in instructions."*
- `website/index.html:132-176, 673-699`: hero & Ch. IX frame natlang around narrow single-shot ticket classifiers (`classify.nl`, `is_urgent.nl`).
- `skills/natlang-authoring/SKILL.md:90-93` & `references/patterns.md:3-12`: *"Most applications are ordinary code with a few `nl` decisions where meaning matters."*
- `ts-host/src/native/prompt.ts:9, 13, 110, 121-124`: defensive system prompt. *(v2)* `SYSTEM_PROMPT` = `TOOLS_PROMPT` 17,162 + `GENERATION_GUIDANCE` 885 = 18,047 characters. It discourages re-delegation in prose (`SEMANTIC_CALL_COMPLETION_PROMPT`), although the runtime already refuses the same question crisply (`runtime/kernel.ts:534-537`). It contains task-specific patches (`more: true` pagination, "save JSON and return it", paired-file and conversation-state rules) and a copyable refund example, guarded by `PROMPT_EXAMPLE_TEXTS` after LFM3-1.7B copied it into 9/42 triage calls. It does not describe `decide` (only `read_code`'s builtin docs do, `native/runtime.ts:270`). The persistent-helper contradiction was in `APPROACH_PROMPT`, which *(v2)* is not sent at runtime (teacher `--approach-guide` only). It is also pro-semantic in places ("If you get a question about meaning, don't try to solve it via code, keyword or regular-expression matching"). Since d17a253fc (2026-11-12) the system prompt and tool list are meant to be one constant cached prefix, with per-call notes at the end of the opening. `native/agent.ts:1276` now adds `edit_code`/`diff_code` conditionally, which may break that prefix (WS2 R0).

### 5.2 Built Capabilities That Were Invisible or Half-Wired
1. **Curried `Context` & `fn.in(context)`**: implemented in `runtime/contexts.ts:219-300` and `callable.ts:148-152`, **0 mentions** in `skills/` or `website/`, omitted from `NatlangCallableMethods` (`intrinsics.ts:129-133`, `surface.ts:31-35`). *(Type bug fixed directly in Wave 0.)*
2. **Speculative/branching `Folder` algebra** (`propose`, `accept`, `snapshot`, `branch`, `select`, `at`): taught to the model in `prompt.ts:16`, **0 mentions** in `skills/`/`website/`; `intrinsics.ts:180-184` mistyped `apply`/`propose`/`iterateOn` as excluding crisp TS reducers and mistyped `moveTo`. *(Type bug fixed directly in Wave 0.)*
3. **`Is<T, P>` in inline `nl`**: worked in `.nl` frontmatter and `eval`, crashed with `TargetError` in inline `nl` (`targets.ts:166-172`). *(Fixed directly in Wave 0.)*
4. **`pluggable({ crisp, nl, shadow })`**: on `website/index.html`, **0 mentions** in `skills/natlang-authoring/`.
5. **Law-based combinator rewrites** (`compiler/rewrites.ts`): built and called from `project.ts:634`, gated off by `DEFAULT_REWRITE_GATE` (`project.ts:205`) with no `natlang.json` config path.
6. **`natlang:learning` (`grad`, `valueAndGrad`, `selfDistill`, `law`)**: **0 mentions** in `skills/` or `website/`. *(v2)* Implemented end to end against the Python native engine's `/v1/neuralese/grad` (the only engine serving it; the browser throws `ModuleUnavailableError`). Its five conformance cases are `status: pending`.
7. *(v2)* **Natlang-native product tools already exist** and are the pattern for the campaign's own tools: `applications/specializer` (promotion policy via `pluggable`, `runtime/promotion.ts:28`), `applications/migration` (repository migration: snapshot, patches as data, measured `iterateOn` repair), `applications/program-improver` behind `natlang improve` (`ts-host/src/improvement/`, adopt/rollback; needs a hand-written `CASE.json`), `applications/memetic-improver` (GRPO share 0.02; forced to 0 in expert-iteration collection), `fusion/planner.ts`, `ops-placement`, `ops-retention`.
8. *(v2)* **Training mechanisms half-wired for semantic FP:** an executable-verifier oracle exists (`trusted_exact_verifier`, `ts-host/src/teacher/trusted-exact-verifier.ts`), but authored tasks accept only `exact`/`normalized`. An option-normalized decision objective exists (`train/decision_supervision.py`), but only the native-graph path uses it, not `exp0_sft`. `grpo.default_held()` admits exactly one `held-eval` entry, so a second held set needs code.

### 5.3 Structural Language Ceilings (Addressed in WS1–WS3)
1. **No parametric polymorphism / generics** (`targets.ts:127`, `loader.ts:272-282`) → blocks reusable `@natlang/std` semantic combinators.
2. **Second-class higher-order semantic functions** (`values.ts:200-203` skips signature checks; `inline.ts:369-390` rejects factory arrows with `nl-not-called`; no `.partial()`).
3. **Filesystem-as-call-graph + brittle `uses:`** (`loader.ts:290-292, 341-373`; no aliases, no npm/workspace `.nl` imports, callable-folder `.ts` barred from `uses:` at `modules.ts:256-258`, 10 KB single-line `.d.nl.ts` blowup at `project.ts:160-193`).
4. **Blanket ban on `.nl` recursion** (`kernel.ts:543-545`) despite existing structural-descent checking for TS (`context.ts:223-258`), plus hard `MAX_AD_HOC_NL_DEPTH = 5` (`kernel.ts:524-529`).
5. **Unary-only `Is<T, P>`** (`refinement.ts:94-97`) → cannot check relational contracts like `"copied verbatim from source"`.
6. **`edit_code` in calls with their own code** (`agent.ts:1276`). *(v2)* It is not always on: it is offered when the call has a codebase and edits are not denied. Evaluation, self-play, improvement and teacher runs deny it, and `withhold:` exists. Edits are run-scoped `ProgramView` revisions, never written to disk. The real gaps: a stuck model can edit what judges it, and a good edit is lost when the run ends. WS1 L3 keeps malleability and makes it a transaction.
7. **Flat primitive-only `readout: decision`** (`decision.ts:19-37`) → cannot score small structured decision records (`{ route, urgent, needsHuman }`).

---

## 6. Conclusions (v2)

1. **Small local models win when the structure carries the exactness.** A 1.7B model fails at simulating 500 words of
   imperative plumbing or at holding a 20-step open loop in its head. It does well when TypeScript owns the data,
   the exact steps and the invariants, and the model runs focused typed judgments: scored decisions, bounded
   `iterateOn` / repair loops with exact checks, and `Neuralese<T>` views of large values. Repeated paths compile to
   0-token crisp cases. That is why the target is not parity with a frontier agent loop but beating it on repeat work
   (README §4).
2. **Language, runtime, front door, corpora and training move together, now.** Fixing only the compiler leaves
   authors following guidance that teaches prose algorithms (§3.5). Fixing only the guidance leaves 83% of the repo's
   `.nl` showing the opposite. Fixing only the corpora leaves LFM3 untrained on the constructs. The waves (README §3)
   run these in parallel. They are not sequenced by layer.
3. **Migrate, don't quarantine.** The legacy corpora have exact behavioural checks (§2, v2 note). A migration
   checked against them yields both a better repo and training pairs (WS7).
4. **The campaign's tools are natlang programs.** Deciding what is a prose algorithm, how to decompose a program,
   whether a task is genuinely semantic and when to escalate are policies. The repo already has the pattern (§5.2.7).
   Building them in natlang on LFM3 is the strongest test the campaign can run on itself (WS3, WS6 Tier 0).
5. **Train through expert iteration, and measure first.** New constructs need oracles that keyword rules fail and
   teacher demonstrations of idiomatic programs. held-v4 is built before the families it measures are admitted (WS5).
6. **Neuralese, memetic learning and self-improvement are built but unreachable from what consumers write.** The
   standard library is where they become reachable (WS8).
7. **Teach in errors.** The prompt and the front door lose their task-specific patches and wrong forms. Checks that
   fire when a wrong form is written teach the fix (WS2 R1, WS4 F7).
