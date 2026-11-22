# WS6: 10 Flagship Semantic-FP Applications (Dogfooding, Showcase & 1.7B Benchmarks)

**Parent:** [Campaign Index](README.md)  
**Depends on:** WS1–WS3 (each tier deliberately pressures and validates the corresponding language/stdlib wave).  
**Goal:** Build 10 genuinely useful, flexible, malleable, well-typed semantic programs that automate **repeat agentic work** people currently pay frontier models for (or fail to automate with brittle scripts) — proving on real workloads that compiled + tuned **LFM3-1.7B** matches frontier quality at **≥ 10× lower cost and latency**.

---

## 1. Shared Engineering Contract for Every Flagship App

Every application in this portfolio must satisfy **all six rules**:
1. **Passes `natlang check --idiom` with 0 errors and 0 warnings** (no prose algorithms, no `.nl`-only trees, no fake `iterateOn`, no untyped external text).
2. **Real typed algebra & exact verifiers:** Discriminated unions, `Untrusted<T>` at every external boundary, `Is<T, P>` / `ensures:` contracts, exact TypeScript invariants, and `ONCE_EFFECTS` on mutating services.
3. **Exercises real Semantic FP primitives:** Every app uses at least 3 of: inline `nl` closures, `@natlang/std` combinators, factorized `readout: decision` + `escalate()`, `.in(context)` lens injection, `Folder` `propose`/`accept`/`branch`, structural `.nl` recursion, `Neuralese<T>` doc views, and `pluggable({ crisp, nl, shadow })`.
4. **Runs end-to-end on local LFM3-1.7B** (plus optional frontier escalation profile), with `examples.json` + realistic benchmark workload + `natlang profile` receipt.
5. **Dual-purpose corpus value:** Ships with an executable verifier & fixture pack (separated into train/held worlds) so WS5 can harvest live traces and execution-verified RL tasks directly from it.
6. **Compact, readable source:** Root orchestration in crisp TypeScript (< 150 lines) + focused `.nl` functions (< 40 words each!) + exact `.ts` helpers.

---

## 2. Tier 1 — Immediate Flagships (Wave 2: Prove the Whole Paradigm)

### App 1: `applications/pr-shepherd` — Semantic PR & Invariant Reviewer + Doc-Drift Healer
- **Repeat agentic work automated:** Every pull request needs hunk classification, architecture/invariant checking against repo conventions, breaking-change detection, missing-test identification, changelog synthesis, and doc-drift repair.
- **Typed Semantic-FP Architecture:**
  - Exact TS (`git` service + TS/AST diff parser) splits the PR into typed `DiffHunk[]` with symbol definitions and callers.
  - **Factorized `readout: decision` + `escalate()`:** `classifyHunk.nl` scores `{ kind: "mechanical" | "behavioral" | "api-break" | "security", risk: "low" | "med" | "high" }` on 1.7B; escalates to frontier only when `p(security) >= 0.20` or `confidence < 0.85`.
  - **`.in(context)` lens injection:** `reviewInvariant.nl` is rebound via `.in(teamStandardsContext)` (security rules, API style, perf budgets) without editing program code.
  - **Relational `ensures:` contract:** Every `ReviewComment` carries `quote: string` checked by `ensures: check: "hunk.patch.includes(result.quote)"` — zero hallucinated line references.
  - **Transactional `Folder` repair:** `proposeAndVerify(repoFolder, updateDocs, runTypecheckAndTests)` updates `README.md`/changelogs/API docs only when `tsc` and tests pass.
  - **JIT payoff:** `natlang specialize` compiles lockfile, snapshot, formatting, and import-order hunks into 0-token crisp cases.
- **1.7B Benchmark Metric:** Recall/precision on 50 seeded real-repo PRs (planted invariant violations, API breaks, doc drift) + tokens/PR and wall-time vs frontier single-agent reviewer.

### App 2: `applications/semantic-etl` — Self-Specializing Schema-Drift Healer (The Killer JIT Showcase)
- **Repeat agentic work automated:** Ingesting messy CSV/JSON/webhook feeds from changing external vendors (renamed columns, mixed units/currencies, free-text status notes, duplicate entities) into a strict target schema.
- **Typed Semantic-FP Architecture:**
  - Exact TS profiles incoming sample rows (column types, value distributions, null rates) and enforces strict output schema + domain conservation invariants.
  - **Higher-order synthesis:** `synthesizeMapper.nl` inspects the schema diff and **returns a typed row transformer** `(row: Untrusted<RawRow>) => Promise<CleanRow>` (using exact TS for structural renames/unit math and inline `nl` lambdas only for genuinely messy free-text columns like notes/categories/addresses!).
  - **`@natlang/std` `semanticMatch`:** Deduplicates / links vendor entities against existing catalog records using exact blocking + `readout: decision` pair scoring.
  - **JIT payoff (headline demo):** Run over 10,000 rows → `natlang specialize` mines the recurring row shapes into guarded crisp JS cases → **95–98% of rows execute at native JS speed (~0.1 ms/row)** while malformed/novel rows seamlessly side-exit to local LFM3-1.7B!
- **1.7B Benchmark Metric:** End-to-end exact field accuracy across 12 schema-drift scenarios + throughput (rows/sec) before vs after `natlang specialize`.

### App 3: `applications/sem-sheet` — Reactive Semantic Spreadsheet (Browser `wllama` + Node Viral Demo)
- **Repeat agentic work automated:** Recurring analytical grids (vendor comparisons, hiring rubrics, competitor matrices, paper screening, quarterly reviews) mixing exact formulas with semantic cells.
- **Typed Semantic-FP Architecture:**
  - Exact TS owns the reactive DAG, topological invalidation, undo/redo, CSV/JSON import, and exact numeric/aggregate formulas (`SUM`, `AVG`, `FILTER`).
  - **Semantic cells are typed inline `nl` closures** referencing upstream cells by name (`=nl<'pass' | 'review' | 'reject'>\`Evaluate candidate against rubric\``), automatically using `readout: decision` when finite, `Neuralese<string>` `view()` caching for large attached documents, and `pluggable({ crisp, nl, shadow })` so users can promote stable semantic columns to exact formulas.
  - **Runs 100% locally in the browser** via `@natlang/browser` + `wllama` (and in Node/CLI), showcasing live incremental cell execution, confidence badges, and one-click `natlang specialize` on a column!
- **1.7B Benchmark Metric:** Interactive cell latency on 4-core CPU / browser `wllama` + accuracy on 3 realistic 50-row workbooks (grant screening, vendor security matrix, bug triage grid).

---

## 3. Tier 2 — Everyday Repeat Agentic Automation (Wave 3)

### App 4: `applications/inbox-pilot` — Higher-Order Rule-Compiled Inbox & Document Actionizer
- **Repeat agentic work automated:** Processing daily emails, Slack threads, invoices, and GitHub notifications against personal/team standing rules written in plain English.
- **Semantic FP Architecture:**
  - **Higher-order `compileRule<Untrusted<Message>, ActionPlan>`:** Users edit a `rules.md` file in plain English; natlang compiles each rule into a typed, example-verified semantic closure.
  - `Untrusted<T>` taints every incoming message/attachment (zero prompt-injection surface); `ONCE_EFFECTS` services guard calendar holds, draft replies, and ledger posts; `Is<Reply, "polite, concise, makes no unapproved commitments">` guards every draft; `escalate()` sends ambiguous or high-value items to a human/frontier queue.

### App 5: `applications/ci-doctor` — Test-Failure Clustering, Git Bisect & Speculative-Fix Autopilot
- **Repeat agentic work automated:** Triaging 20–50 CI failures after a merge wave: clustering duplicate root causes, bisecting the culprit commit, separating flakes from regressions, and proposing verified fixes.
- **Semantic FP Architecture:**
  - Exact TS parses JUnit/logs and wraps `git` + test-runner services.
  - `@natlang/std` `semanticPartition` / `semanticMatch` clusters stack traces into distinct root causes; `readout: decision` classifies `flake | env | regression`; `iterateOn` runs a hypothesis-guided semantic bisect over recent commits; `branchAndSelect` runs 2–3 candidate fixes in parallel `Folder` branches and accepts only a patch that turns red tests green without deleting assertions.

### App 6: `applications/sem-merge` — Intent-Preserving Semantic Diff & Three-Way Conflict Resolver
- **Repeat agentic work automated:** Resolving git/Markdown/JSON/config merge conflicts where two branches edited the same section with compatible intent (elevating `applications/wiki/merge` into a real CLI git mergetool!).
- **Semantic FP Architecture:**
  - Pure TS computes structural AST/section diffs and merges all non-overlapping hunks untouched.
  - For conflicted hunks: `.nl` extracts `leftIntent` and `rightIntent`; `readout: decision` checks `compatible(leftIntent, rightIntent)`; `@natlang/std` `semanticMerge` + `semanticRepair` synthesizes the merged hunk and iterates until syntax/linter checks pass AND relational `ensures:` verifies both intents are preserved (explicitly emitting unresolved conflict markers when intents truly clash).

### App 7: `applications/runbook-engine` — Executable Living SRE / Ops Runbooks
- **Repeat agentic work automated:** Running operational procedures (incident mitigation, customer onboarding, release checklists, DB failover) directly from Markdown runbooks with human-in-the-loop safety gates.
- **Semantic FP Architecture:**
  - Parses Markdown sections into a typed step graph; `readout: decision` gates every external service call against risk floors (`p(destructive) >= 0.30` → require human confirmation); `ONCE_EFFECTS` services ensure crash-resumable idempotency; `iterateOn` drives post-action health verification; editing a runbook paragraph updates program behavior cleanly under `natlang check`.

---

## 4. Tier 3 — Deep Frontier & Meta-Programming Showcases (Wave 4)

### App 8: `applications/sem-grep-sql` — Relational + Semantic Workspace Query Engine
- **Repeat agentic work automated:** Cross-cutting analytical queries over mixed local repos, Markdown docs, CSVs, and SQLite databases (*"Which RFC guarantees lack matching unit tests, grouped by owner?"*).
- **Semantic FP Architecture:** Generalizes `nldb`'s SQLite + `readout: decision` UDF into a typed relational + semantic query planner: exact SQLite joins/indexes/aggregations composed with cached semantic `WHERE` predicates (`nl<boolean>`), semantic table extractors (`nl<Row[]>`), and `Neuralese<T>` document views persisted per file SHA so repeated queries over 1,000+ files run in seconds on 1.7B.

### App 9: `applications/codemod-studio` — Hybrid AST + Semantic Migration Workbench
- **Repeat agentic work automated:** Large-scale repo migrations (API v1→v2, library swaps, error-handling refactors) where deterministic AST codemods handle 70% of files and break on the semantic 30%.
- **Semantic FP Architecture:** Uses `Folder` branching (`snapshot`/`propose`/`accept`), runs a crisp TS AST codemod first, routes remaining/failing files to typed `.nl` transformers checked by `tsc` + unit tests inside `semanticRepair`, and runs `natlang specialize` on the solved edge-case traces to **automatically mine new crisp codemod rules** for the remaining repo!

### App 10: `applications/diligence` — Self-Specializing Research & Due-Diligence Pipeline
- **Repeat agentic work automated:** Repeatedly evaluating papers, OSS libraries, vendors, or grant proposals against a structured rubric and a folder of raw evidence.
- **Semantic FP Architecture:** Showcases the entire stack in ~150 lines: `Untrusted<T>` on scraped evidence, `Neuralese<T>` producer→consumer views over long PDFs/docs, structural `.nl` recursion over nested document sections, factorized `readout: decision` rubric gates + `escalate()`, relational `ensures:` citation grounding, `.in(lensContext)` to swap evaluation frameworks, and `natlang tune` to distill a user's historical evaluations into a specialized LFM3-1.7B LoRA adapter.

---

## 5. LFM3-1.7B vs. Frontier Benchmark Protocol

For every Tier-1 and Tier-2 app, ship a reproducible benchmark (`applications/<app>/bench/`) comparing four execution modes on the **exact same held workload**:

| Arm | Configuration | What It Measures |
|---|---|---|
| **A. Frontier Unstructured Agent** | Frontier model in a standard tool loop with a prose prompt | Baseline cost, latency, and quality of today's status quo |
| **B. Natlang Semantic-FP (Frontier)** | Same typed natlang program on frontier model | Upper-bound quality of the typed architecture |
| **C. Natlang Semantic-FP (Base LFM3-1.7B)** | Same program on local LFM3-1.7B (with `readout: decision` + `escalate` off) | Out-of-the-box local small-model performance |
| **D. Natlang Compiled (LFM3-1.7B + JIT + Tuned + Escalate)** | After `natlang compile` (`specialize` + `tune` + `escalate` floor) | **The thesis proof:** quality parity with Arm A/B at ≥ 10× lower cost & latency |

**Reported metrics per app:**
1. Independent verifier pass rate / task quality (paired bootstrap CI).
2. Median & p95 latency per unit of work (ms).
3. Total model tokens & dollar cost per 100 units of work.
4. JIT crisp hit rate (%), local 1.7B resolution rate (%), frontier escalation rate (%).
5. Failure safety (0 unverified commits, 0 prompt-injection escapes, 0 schema violations).
