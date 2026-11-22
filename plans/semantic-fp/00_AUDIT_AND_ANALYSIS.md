# Audit & First-Principles Analysis: The Language Side of natlang (2026-11-22)

**Parent:** [Campaign Index](README.md)  
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

| Metric | `examples/` | `applications/` (39 apps) | `codebases/` excl. `design_briefs` (17) | `codebases/design_briefs/` (100) | `ts-host/playground/examples/` (38) | `ts-host/examples/` | **Total** |
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
   - `ts-host/playground/examples/crisp.mjs`: **21 of 38** gallery entries are pure-TypeScript LeetCode utilities (`balanced-brackets`, `shortest-route`, `merge-intervals`, `form-validation`) with **zero** `.nl` functions; line 114 (`while (pending.size)`) even violates natlang's own finite-iteration rule.
   - `ts-host/playground/examples/natural.mjs`: all 13 examples use archaic pre-v0.5 phrasing (`"Write the number 7 to return"`), 0 `readout: decision`, 0 inline `nl`, 0 `iterateOn`, 0 `Untrusted<T>`, 0 `Is<T, P>`.
   - `ts-host/playground/examples/interfaces.mjs:37-53, 79-97`: regenerates an entire hardcoded JSON DOM tree through an LLM on every button click.

5. **Best Existing Exemplars (The Positive Proof):**
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
- `ts-host/src/native/prompt.ts:9, 13, 110, 121-124`: ~15.3 KB defensive system prompt (`TOOLS_PROMPT` + `GENERATION_GUIDANCE`) that discourages delegation (`SEMANTIC_CALL_COMPLETION_PROMPT`), contains hardcoded benchmark patches (requiring `PROMPT_EXAMPLE_TEXTS` because LFM3-1.7B copied the refund example into 9/42 triage calls), and contradicts `scope-compiler.ts:749-802` on persistent eval helpers.

### 5.2 Built Capabilities That Were Invisible or Half-Wired
1. **Curried `Context` & `fn.in(context)`**: implemented in `runtime/contexts.ts:219-300` and `callable.ts:148-152`, **0 mentions** in `skills/` or `website/`, omitted from `NatlangCallableMethods` (`intrinsics.ts:129-133`, `surface.ts:31-35`). *(Type bug fixed directly in Wave 0.)*
2. **Speculative/branching `Folder` algebra** (`propose`, `accept`, `snapshot`, `branch`, `select`, `at`): taught to the model in `prompt.ts:16`, **0 mentions** in `skills/`/`website/`; `intrinsics.ts:180-184` mistyped `apply`/`propose`/`iterateOn` as excluding crisp TS reducers and mistyped `moveTo`. *(Type bug fixed directly in Wave 0.)*
3. **`Is<T, P>` in inline `nl`**: worked in `.nl` frontmatter and `eval`, crashed with `TargetError` in inline `nl` (`targets.ts:166-172`). *(Fixed directly in Wave 0.)*
4. **`pluggable({ crisp, nl, shadow })`**: on `website/index.html`, **0 mentions** in `skills/natlang-authoring/`.
5. **Law-based combinator rewrites** (`compiler/rewrites.ts`): built and called from `project.ts:634`, gated off by `DEFAULT_REWRITE_GATE` (`project.ts:205`) with no `natlang.json` config path.
6. **`natlang:learning` (`grad`, `valueAndGrad`, `selfDistill`, `law`)**: **0 mentions** in `skills/` or `website/`.

### 5.3 Structural Language Ceilings (Addressed in WS1–WS3)
1. **No parametric polymorphism / generics** (`targets.ts:127`, `loader.ts:272-282`) → blocks reusable `@natlang/std` semantic combinators.
2. **Second-class higher-order semantic functions** (`values.ts:200-203` skips signature checks; `inline.ts:369-390` rejects factory arrows with `nl-not-called`; no `.partial()`).
3. **Filesystem-as-call-graph + brittle `uses:`** (`loader.ts:290-292, 341-373`; no aliases, no npm/workspace `.nl` imports, callable-folder `.ts` barred from `uses:` at `modules.ts:256-258`, 10 KB single-line `.d.nl.ts` blowup at `project.ts:160-193`).
4. **Blanket ban on `.nl` recursion** (`kernel.ts:543-545`) despite existing structural-descent checking for TS (`context.ts:223-258`), plus hard `MAX_AD_HOC_NL_DEPTH = 5` (`kernel.ts:524-529`).
5. **Unary-only `Is<T, P>`** (`refinement.ts:94-97`) → cannot check relational contracts like `"copied verbatim from source"`.
6. **Always-on `edit_code` footgun** (`agent.ts:1275-1279`) → pushes pure domain code out of callable folders into host services.
7. **Flat primitive-only `readout: decision`** (`decision.ts:19-37`) → cannot score small structured decision records (`{ route, urgent, needsHuman }`).

---

## 6. First-Principles Reasoning Behind the Campaign

1. **Why small local models (1.7B) win under Semantic FP, not under monolithic prompts or prose checklists:** A 1.7B model fails when asked to simulate 500 words of imperative TypeScript or hold a 20-step open agent loop in its head. It *excels* when TypeScript owns the exact algebra, data structures, concurrency, and invariants, while the model executes focused typed semantic lambdas, calibrated `readout: decision` passes, `Neuralese<T>` compressed views, and bounded `iterateOn` / `semanticRepair` loops — and when trace specialization (`natlang specialize`) compiles hot paths down to 0-token crisp JS.
2. **Why fixing the language, docs, and training pool must move together:** Fixing only the compiler leaves agents writing timid classifiers; fixing only the skills leaves authors hitting `TargetError` on inline `Is<T, P>` and `no-such-function` on shared libraries; fixing only examples leaves LFM3-1.7B untrained on inline `nl`, `.in()`, relational refinements, and library discovery.
3. **Why Wave-0 fixes ship immediately:** Every Wave-0 defect is strictly wrong on first principles (`AGENTS.md`), zero-risk to running training, and immediately unlocks cleaner authoring and testing for Waves 1–4.
