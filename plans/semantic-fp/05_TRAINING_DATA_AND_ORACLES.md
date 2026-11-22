# WS5: Training Data, Oracle Revolution & `RL_TASK_AUTHORING_KIT` v2

**Parent:** [Campaign Index](README.md)  
**Analysis:** [00_AUDIT_AND_ANALYSIS.md](00_AUDIT_AND_ANALYSIS.md)  
**Goal:** Break the Exact-Oracle + Literal Blind-Solve cue-word trap, close every 0%-coverage Semantic-FP gap in `grpo-pool` and SFT (`expert_iteration.py`), wire the repo's real multi-module applications into training, and replace synthetic "operations desks" with execution-grounded and metamorphically verified Semantic-FP task families.

---

## 1. Mechanism Inventory

| Mechanism | Components | Scope in WS5 |
|---|---|---|
| RL task schema, conversion & verification | `scripts/rl_tasks/sources/authored.py`, `authored_checks.py`, `authored_novelty.py`, `qualify_authored.py`, `verify.mjs`, `structure_smoke.mjs`, `blind_solve.py` | Add 4 new oracle modes, anti-cue-word lint, inline-`nl` / `.in()` / `Is<T,P>` / structural-recursion construct tags |
| Collector & rollout scoring | `training/neuralese/natlang_neuralese/rl/task_rollouts.py`, `ts-host/src/teacher/collector.ts` | Support execution-verifier, metamorphic-pair, proper-scoring (Brier), and relational-refinement oracles |
| SFT / Expert-Iteration data builder | `scripts/neuralese_data/sft_sources.py`, `training/neuralese/natlang_neuralese/rl/expert_iteration.py` | Add `readout: decision` soft-target / scoring supervision; ensure new constructs flow cleanly through `sft_sources.build` |
| Data registry & custody | `scripts/data_registry.py`, `ops/held-v3/exposure/collect_exposure.py`, `training/protected-worlds.json` | Register every new corpus with clean `held-v3`/`held-v2` separation (`held_eval verify`) |
| Authoring kit & rotation | `plans/neuralese/RL_TASK_AUTHORING_KIT.md` → `RL_TASK_AUTHORING_KIT_V2.md`, `scripts/rl_tasks/authoring_rotation.py` | New 7-archetype rotation matrix replacing cue-word operations desks |
| Held evaluation | `ops/held-v3/` (keep intact for continuity) + new `ops/held-v4/` | Track `held-v3` continuously; build `held-v4` to measure full Semantic-FP breadth |

---

## 2. Dismantling the Cue-Word Trap: Four New Oracle Classes

Exact string/JSON deep equality + literal Sonnet blind-solving (`blind_solve.py:54-62`) made genuine semantic tasks un-authorable, forcing Haiku workers to build keyword/cue-word checklists (`TASK_SUPPLY_20261118.md:17, 122, 143, 163, 362-364`).

We keep exact oracles where answers are genuinely canonical (e.g., git history, SQL execution, exact state transitions), and add **four oracle classes** that make real semantic understanding strictly easier to verify than cue-word shortcuts:

### Oracle Class 1: Execution / Simulator / Invariant Oracles (`oracle: "verifier"`)
- **How it works:** The semantic program outputs a typed artifact — a SQL query, a schedule, a migration patch, a `Folder` edit, a compiled predicate lambda, a conflict-resolved document, or a sequence of service calls. A deterministic TypeScript verifier script bundled in the task **executes the returned artifact** against hidden unit tests, simulators, or domain invariants and returns a graded score `[0, 1]` + binary pass (`all_invariants_hold`).
- **Why it defeats cue-word gaming:** Multiple valid semantic solutions pass; any broken shortcut fails execution. Zero blind-solve ambiguity disputes over phrasing!
- **Immediate sources:** `applications/{nldb,notebook,helpdesk,games,wiki,workflow,program-improver}`, `examples/triage`, `ts-host/examples/browser-board`, and all WS6 flagship apps.

### Oracle Class 2: Metamorphic Paraphrase-Invariance Oracles (`oracle: "metamorphic"`)
- **How it works:** Every task carries a **paired twin world** (or 3-way paraphrase group) expressing the *exact same semantic facts/rules* with **zero overlapping domain cue words** (verified statically by low lexical Jaccard < 0.25 on content lemmas while preserving semantic equivalence).
- **Scoring:** In GRPO/expert iteration, a task's program/trajectory earns full credit only when it solves the semantic judgment consistently across paraphrases (or a batch of paired worlds). In authoring qualification, `authored_checks.py` runs an automated **anti-cue-word check**: any `.nl` helper whose instructions cite literal cue strings (`"freeze"`, `"injur"`) is rejected by `idiom-cue-word-checklist` (WS3).
- **Why it kills cue-word desks:** Literal keyword rules fail the twin world 100% of the time; genuine semantic judgment passes both.

### Oracle Class 3: Calibrated Proper-Scoring Oracles for `readout: decision` (`oracle: "proper_score"`)
- **How it works:** For `decide()` tasks (including confidence-floor escalation and risk-sensitive abstention), grade the returned probability distribution $p$ against ground truth $y$ using Brier score / bounded log-loss / expected asymmetric decision utility (e.g., false-negative security miss costs $10\times$, unnecessary escalation costs $1\times$, correct local route costs $0$).
- **Closes two huge gaps at once:**
  1. Re-enables probability-floor abstention/escalation tasks that `scripts/rl_tasks/sources/decisions.py:20` explicitly excluded.
  2. Enables **SFT supervision for `readout: decision`**! Today `SFT_WARMUP_MIX_20261118.md:192` has **0% `readout: decision` targets** because *"Teachers cannot score options."* Fix: compute teacher option probabilities (via the teacher's logprobs/scoring endpoint or calibrated multi-sample soft labels on our own Python/SGLang server) and train `readout: decision` rows in `exp0_sft.py` with cross-entropy / KL against the verified option distribution (opt-out `--no-decision-sft`).

### Oracle Class 4: Relational Refinement & Provenance Oracles (`oracle: "refinement"`)
- **How it works:** Combines exact structural/provenance checks (`quotes.every(q => source.includes(q))`, exact entity IDs, zero hallucinated fields, schema fit) with crisp/judge `Is<T, P>` and `ensures:` checks (WS1). Used for grounded synthesis, review comments, citation extraction, and semantic merging where free-text exact match is absurd but verifiability is high.

---

## 3. Closing Every 0%-Coverage Semantic-FP Gap in `grpo-pool` & SFT

Target **rebalanced pool mix** for newly collected expert-iteration / GRPO cycles (gradually diluting the 89.7% legacy QA/puzzle dominance via selector weights + new high-value sources):

| Target Capability (Currently 0–2%) | Concrete New Training Sources | Target Pool Share |
|---|---|---:|
| **1. Inline `` nl`...` `` lambdas in `eval`** | Procedural + authored **Inline Semantic-FP Suite**: root `.nl` or TS orchestrator receives structured data + evolving criteria, writes `Promise.all(items.map(x => nl<T>\`...\`(x)))`, `nl.with({ ... })`, or dynamic semantic predicates inside `eval`, checked by exact aggregation or execution verifier. | **12%** |
| **2. Real codebase `read_code` discovery & composition** | **Library Discovery Suite**: mount 8–20 mixed `.ts` + `.nl` domain functions (using `uses:` v2 + progressive opening summaries from WS2) and **forbid naming the helper calls in the root prompt**. The model must use `read_code` / `search_code` to discover the right helpers, compose them in `eval`, and verify invariants. Seeded directly from `applications/{notebook,helpdesk,nldb,wiki,logs,workflow,games}`, refactored `sol_programs.json`, and WS6 apps! | **15%** |
| **3. `iterateOn` with NL stopping predicates & `semanticRepair`** | **Convergence & Repair Suite**: open-ended refinement loops (`until(nl\`...\`)`), `nlprolog`-style semantic fixed points, and `semanticRepair` loops fed real compiler/test/verifier errors (`notebook` citations, `board` layout invariants, `nldb` SQL plans, `pr-shepherd` doc updates). | **10%** |
| **4. Higher-order semantic lambdas & `.in(context)` rebinding** | **Policy-Compilation & Lens-Swap Suite**: tasks that (a) compile prose rules into a typed `(x: Event) => Promise<Decision>` lambda and execute it on a hidden test suite (`compileRule` oracle!), or (b) run the same `.nl` function under two swapped `Context` bindings via `.in(ctx)` and compare/reconcile results. | **8%** |
| **5. Relational `Is<T, P>` & `ensures:` repair** | **Semantic Contract Suite**: functions with `Untrusted<T>` inputs (including adversarial prompt-injection payloads!) and crisp + semantic `ensures:` postconditions, testing both clean first-pass satisfaction and recovery from `RefinementCallError`. | **8%** |
| **6. Structural `.nl` recursion** | **Recursive Semantic Tree Suite**: recursive `.nl` functions with `recursive: structural` / `measure:` operating over nested ASTs, repo trees, nested legal/spec clauses, and hierarchical org/dependency graphs. | **7%** |
| **7. Calibrated `readout: decision` (flat + factorized) & `escalate()`** | **Decision & Escalation Suite**: flat + product-type `readout: decision` tasks with proper-scoring / asymmetric-cost oracles, plus SFT soft-target distillation rows. | **10%** |
| **8. Transactional `Folder` branching (`propose`/`accept`/`branch`)** | **Speculative Workspace Suite**: tasks where a candidate edit may fail a verifier (`tsc`, `pytest`, schema check) and the model must `propose` → check → either `accept` or `branch` and try an alternative without corrupting the base folder. | **10%** |
| **9. Neuralese producer→consumer & `@natlang/std` fusion** | Retained & upgraded from existing composed/authored Neuralese tasks, shifted toward typed `Neuralese<{...}>` records, `semanticReduce(..., { neuralese: true })`, and long-document `view()` bottlenecks (replacing formulaic 1-helper QA wrappers). | **10%** |
| **10. Real-world live workflows & general reasoning anchor** | Observed live VibeStudio workflows (`LIVE_WORKFLOW_SFT_20261118.md`), `git_history`, writing/chat diversity, and curated subset of hardest legacy reasoning tasks (down-sampling easy 88%-pass GSM8K/puzzles). | **10%** |

---

## 4. Activating Fallow High-Value Repo Assets Immediately

Before authoring net-new tasks, wire three already-built, zero-cost assets sitting unused in the repo:
1. **Real `applications/` Execution Harvest:** Build `scripts/rl_tasks/sources/flagship_apps.py` turning deterministic fixture suites from `applications/{notebook,helpdesk,nldb,wiki,logs,workflow,games,program-improver,specializer}` into executable `natlang.program/2` RL tasks with real host services and exact invariant/execution verifiers (checked against `held-v3`/`held-v2` custody).
2. **`training/semantic_stack/sol_programs.json` (100 typed multi-operation programs over 18 shared worlds):** Audit for `.nl`-only anti-patterns, extract pure TS plumbing helpers where appropriate, verify non-overlap with `ProtectedWorlds` / `held-v3`, and admit executable multi-operation tasks into `grpo-pool`.
3. **Curated subset of `codebases/design_briefs/`:** Instead of static AST Q&A (`natcode.py`) or pure-TS-only execution (`codebase_ops.py`), select the 25 highest-quality design-brief worlds, add crisp `.ts` verification helpers + real execution fixtures (expanding `design_brief_gates.py` from 180 rows to ~1,500 genuine executable codebase-composition tasks).

---

## 5. `RL_TASK_AUTHORING_KIT` v2 Spec

Supersede `plans/neuralese/RL_TASK_AUTHORING_KIT.md` (which drove R1–R23 toward closed-set cue-word operations desks) with `plans/neuralese/RL_TASK_AUTHORING_KIT_V2.md`:

1. **Seven Semantic-FP Archetypes (replacing the 10 operations-desk problem keywords in `authoring_rotation.py:59-70`):**
   - `inline-lambda-pipeline` (dynamic inline `nl` inside `eval` over structured collections)
   - `library-discovery-composition` (10+ mixed `.ts`/`.nl` codebase, zero helper names pasted in prompt; requires `read_code`/`search_code`)
   - `semantic-convergence-repair` (`iterateOn` with NL stopping predicate or `semanticRepair` against an executable verifier)
   - `higher-order-policy-compiler` (compiling prose rules into typed semantic closures or `.in()` context swaps)
   - `speculative-folder-transaction` (`folder.propose` → verifier check → `accept` or alternative `branch`)
   - `adversarial-untrusted-contract` (`Untrusted<T>` injection payloads + relational `ensures:` / `Is<T, P>` contracts)
   - `structural-recursive-decomposition` (recursive `.nl` over nested trees/ASTs/clauses)
2. **Mandatory Idiom & Anti-Cue-Word Qualification:** `qualify_authored.py` runs `natlang check --idiom` (including `idiom-cue-word-checklist`, `idiom-prose-algorithm`, `idiom-nl-only-tree`, `idiom-fake-iterate`) as a hard gate before admission.
3. **Replace literal blind-solve on execution/metamorphic tasks:**
   - For `oracle: "verifier"`, qualification runs a real reference solution + ≥ 2 failing `wrong_code` scripts against the actual verifier (proving both solvability and discriminator power without needing Sonnet to hand-simulate byte-exact file formatting!).
   - For `oracle: "metamorphic"`, qualification verifies lexical disjointness (`content_lemma_jaccard < 0.25`) between twin worlds and confirms a stronger model agrees on the semantic outcome across both twins.

---

## 6. `held-v4`: Measuring What Actually Matters

Keep `held-v3` (256 tasks, sha `45471060`) frozen as the historical continuity anchor so we never lose comparability with prior experiments. In parallel, assemble **`held-v4`** (256–320 tasks, completely disjoint worlds/source-groups from all training pools and `held-v3`) stratified across the real Semantic-FP product target:
- 20% Mixed codebase discovery & composition (`read_code` over 8–20 function `.ts`/`.nl` libraries)
- 15% Inline `nl` lambdas inside `eval` (closures, map/filter/reduce, typed holes)
- 15% Execution-verified `Folder` transactions (`propose`/`accept`/`branch` with real verifiers)
- 15% Calibrated `readout: decision` (flat + factorized) & confidence-floor escalation (scored by accuracy + Brier/utility)
- 10% Semantic `iterateOn` (NL predicates + `semanticRepair`) & structural `.nl` recursion
- 10% Higher-order lambdas (`compileRule`) & `.in(context)` lens swaps
- 10% `Untrusted<T>` prompt-injection resistance + relational `Is<T, P>` / `ensures:` contracts
- 5% Typed `Neuralese<T>` multi-stage pipelines & `@natlang/std` fusion

---

## 7. Acceptance Gates for WS5

- Every new corpus passes `held_eval verify` against `held-v3` (and `held-v2` + `ProtectedWorlds`) with 0 ID/group overlaps.
- Zero cue-word checklist tasks survive `qualify_authored.py` (tested via negative unit fixtures in `tests/test_authored_checks.py`).
- `sft_sources.py build` produces verified non-zero token/row mass across **every** construct listed in §3, including `readout: decision` soft-target rows and inline `nl` rows.
- `held-v4` passes A/A reproducibility and base measurement on Pop/DGX before replacing `held-v3` as the primary promotion gate.
