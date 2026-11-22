# WS4: Front-Door Reset — Docs, Examples, Playground, `natlang init`, Skills & Corpus Quarantine

**Parent:** [Campaign Index](README.md)  
**Analysis:** [00_AUDIT_AND_ANALYSIS.md](00_AUDIT_AND_ANALYSIS.md)  
**Goal:** Replace the timid *"TS + ticket classifier"* front door with an unmistakable, inspiring, practical presentation of **typed semantic functional programming** across every entry point a human or coding agent touches — and quarantine bulk legacy prose corpora so coding agents stop copying anti-patterns.

---

## 1. Mechanism Inventory

| Surface | Files | Current Defect | Target State |
|---|---|---|---|
| Project scaffolder (`natlang init`) | `ts-host/src/package/authoring.ts` | Scaffolds 1-file 3-label billing classifier (`answer.nl`), teaches narrow classifier mental model in generated `AGENTS.md`. | Default 3-file semantic-FP starter + `--template` gallery (`fp-pipeline`, `decision-router`, `folder-agent`, `classifier`); rewritten `AGENTS.md`. |
| Top-level `README.md` | `README.md` | Good 8-line `triage` snippet undermined by immediate pivot to `answer.nl` 3-label classifier; hides JIT specialization, `.in()`, `Is<T,P>`, semantic libraries. | Radical manifesto opening + crisp "before/after" (brittle script vs frontier agent vs compiled 1.7B semantic-FP program). |
| Illustrated Website | `website/index.html`, `website/app.js` | Anchored on `classify.nl` / `is_urgent.nl` and Ch. IX's narrow "decisions vs exact work" dichotomy; missing `.in()`, higher-order composition, `Folder` branching. | Keep 9-plate visual language; upgrade hero & Ch. IX; add interactive exhibits for Higher-Order Composition, `.in()` Context Rebinding, and Transactional `Folder` Branching. |
| Browser Playground | `ts-host/playground/examples/{crisp,natural,interfaces,builders}.mjs`, `examples.mjs` | 21/38 pure-TS LeetCode puzzles; 13 archaic pre-v0.5 single-shot prompts (`"Write X to return"`); 0 modern v0.5 features. | 12-example **Semantic FP Progression** running live in browser (`wllama`) + Node; archive LeetCode exercises out of the default gallery. |
| Authoring Skills | `skills/natlang-authoring/{SKILL.md,references/*.md,assets/review/}` | Steers agents to minimize `.nl` to narrow leaves; omits `.in()`, `pluggable`, `Folder` branching, `@natlang/std`, JIT specialization, anti-pattern gallery. | Rewritten around Semantic Functional Architecture, `@natlang/std`, `natlang check --idiom`, progressive compilation, and explicit Do/Don't contrast pairs. |
| Bulk Legacy Corpora | `codebases/design_briefs/` (1,390 `.nl`), 11 Astra apps in `applications/` (321 `.nl`) | 83% of repo `.nl` files embody `.nl`-only prose-interpreter anti-patterns and dominate agent grep/discovery. | Quarantine / clearly label with archival manifests and curated `applications/README.md` + `examples/README.md` indexes without breaking training/test scripts. |

---

## 2. Detailed Rework Plans by Surface

### 2.1 `natlang init` Overhaul (`ts-host/src/package/authoring.ts`)
- **CLI flag:** `natlang init DIR [--template fp-pipeline | decision-router | folder-agent | classifier]` (default: `fp-pipeline`).
- **Default `fp-pipeline` template (4 crisp files, < 90 total lines, runs fast on local LFM3-1.7B):**
  - `types.ts`: `Untrusted<string>` input record, discriminated union `Action`, refined `Summary = Is<string, "one concise sentence grounded in the items">`.
  - `classify.nl`: fast `readout: decision` leaf (`"action" | "fyi" | "ignore"`).
  - `index.ts`: TypeScript orchestrator using `Promise.all` + `classify` + an inline `nl` closure (`nl<Action>\`...\``) + exact deduplication/sorting + `examples.json` tests!
  - `AGENTS.md` (and `CLAUDE.md`): rewritten around the **Three Questions of Semantic Functional Decomposition** (typed algebra → semantic lambdas → exact invariants & progressive compilation), explicitly teaching `natlang check --idiom`, `natlang profile`, `readout: decision`, inline `nl`, `Is<T, P>`, `Untrusted<T>`, `Folder` reducers, and `.in()`.
- Keep `--template classifier` for minimal 1-file quickstarts (and preserve existing `natlang setup` warm-prefix priming for both templates).

### 2.2 `README.md` Manifesto Rewrite
1. **Opening thesis:** Lead directly with the real promise: *"Write flexible, malleable, well-typed semantic programs that automate repeat agentic work on small, fast, local models."*
2. **Concrete 20-line hero example:** Show a real repeat-agentic-work snippet (e.g., PR/inbox triage + structured action synthesis + exact invariant check + `iterateOn` repair) demonstrating:
   - inline `nl` lambda with lexical capture & typed-hole inference,
   - `readout: decision` + `escalate({ floor: 0.85, fallback: 'frontier' })`,
   - `Untrusted<T>` + `Is<T, P>` / `ensures:`,
   - exact TypeScript verification & aggregation,
   - one-command `natlang profile` + `natlang compile` (JIT specialization + tuning).
3. **Why Semantic FP beats both extremes:** Crisp 3-column comparison table (Hand-Coded Scripts vs. Frontier Agent Loops vs. Compiled Natlang Programs on 1.7B).
4. **Curated tour of real apps:** Link directly to the curated flagship applications (`examples/triage`, `examples/review`, `applications/notebook`, `applications/helpdesk`, `applications/nldb`, plus Tier-1 flagships from WS6).

### 2.3 Browser Playground: The 12-Example Semantic FP Progression
Replace `ts-host/playground/examples/natural.mjs` and demote `crisp.mjs` (pure-TS LeetCode) into a separate collapsed "Exact TypeScript Reference" tab (after fixing the banned `while` loop at `crisp.mjs:114` directly in Wave 0).

Ship **12 curated, progressively richer examples** (every single one checked by `natlang check --idiom` and tested on browser `wllama` + Node):

| # | Example | Level | Core Semantic-FP Concept Taught |
|---|---|---|---|
| 1 | **Calibrated Triage Gate** | Starter | `readout: decision` + `decide()` probability vector & confidence floor |
| 2 | **Inline Semantic Closure** | Starter | Inline `` nl`...` `` capturing local TS policy & threshold live + typed-hole inference |
| 3 | **Safe External Text & Semantic Types** | Starter | `Untrusted<T>` fencing + `Is<T, P>` / `ensures:` automatic repair loop |
| 4 | **Parallel Semantic Map + Exact Tally** | Core | `Promise.all` over typed semantic lambdas + exact TS aggregation & invariants |
| 5 | **Adaptive Convergence (`iterateOn`)** | Core | `iterateOn` with a **natural-language stopping predicate** + progress review |
| 6 | **Hybrid Logic Engine (`nlprolog` mini)** | Core | Mixed `.nl` + `.ts` callable folder: semantic rule matching + exact fixed-point engine |
| 7 | **Semantic Join & Pairwise Rank** | Core | `@natlang/std` `semanticMatch` & `semanticRank` (pairwise `readout: decision` + exact Copeland score) |
| 8 | **Higher-Order Policy Compiler** | Advanced | `.nl` function returning a typed semantic lambda `(event) => Promise<Action>` from plain-English rules |
| 9 | **Swappable Lenses (`.in(context)`)** | Advanced | Same semantic program bound to two different `Context` lenses (`security` vs `clarity`) |
| 10 | **Crisp / NL / Shadow (`pluggable`)** | Advanced | Switching between fast regex/code and `.nl` behind one typed contract with live disagreement telemetry |
| 11 | **Transactional Workspace (`Folder`)** | Advanced | `folder.propose` → exact invariant check → `folder.accept` (or clean rollback) |
| 12 | **JIT Specialization & `Neuralese<T>`** | Advanced | Watching `natlang specialize` turn hot semantic paths into 0-token crisp cases + `Neuralese<T>` producer→consumer handoff |

### 2.4 Website (`website/index.html` & `website/app.js`)
Keep the acclaimed 9-plate editorial design, and upgrade the content so it teaches Semantic FP rather than ticket classification:
- **Hero:** Swap the billing-ticket classifier for a compact real structured-computing snippet (inline `nl` + `readout: decision` + exact verifier + `iterateOn`).
- **Chapter III (Closures & Contexts):** Add an interactive **Context Rebinding (`.in()`)** widget beside Capture-by-Mention, showing the same function running under two swapped contexts (`Context.of(...)`) with structural interface checking.
- **Chapter IV / V (Interpreter & Termination):** Show **structural semantic recursion** alongside `iterateOn` and **Transactional `Folder` Branching (`propose`/`branch`/`select`)**.
- **Chapter VII (A JIT for English):** Broaden into **The Progressive Compilation Ladder** (Spec → JIT Crisp Cases → Calibrated Decisions & Escalation → Tuned 1.7B LoRA + Algebraic Fusion).
- **Chapter IX (Policy & Mechanism → Designing Semantic Functional Codebases):** Replace *"write decisions as `.nl` and keep exact work in TS"* with the **Three Questions of Semantic Functional Decomposition**, plus an explicit **Anti-Patterns to Avoid** panel (500-word prose algorithms, `.nl`-only trees, cue-word checklists).

### 2.5 Authoring Skill Overhaul (`skills/natlang-authoring/`)
1. **Rewrite `SKILL.md`:**
   - Lead with Semantic Functional Architecture, not "put the algorithm as numbered prose steps."
   - Add a **Decision Matrix** for choosing the right construct: exact `.ts` helper vs `readout: decision` vs inline `nl` vs named `.nl` vs `Folder` reducer vs `@natlang/std` combinator.
   - Add an **Anti-Pattern Gallery (Concrete Before/After)** showing:
     - ❌ 300-word numbered prose doing array slicing/field copying → ✅ 20-line `.ts` + 2 focused `.nl` leaves;
     - ❌ `.nl` cue-word checklist (`if note contains "freeze"`) → ✅ exact `.ts` regex OR genuine semantic criterion;
     - ❌ Fake `iterateOn` over a fixed array → ✅ `Promise.all` / `semanticMap`;
     - ❌ Prose guard sentences (*"under 60 chars, ignore prompt injections"*) → ✅ `Untrusted<T>` + `Is<T, P>` / `ensures:`.
   - Document `natlang check --idiom` and `natlang profile` in the standard working loop.
2. **Add `references/composition.md`:** Deep guide covering higher-order semantic functions, `@natlang/std`, `.in(context)` dependency injection, `pluggable({ crisp, nl, shadow })`, `escalate()`, transactional `Folder` branching (`propose`/`accept`/`branch`/`select`), structural `.nl` recursion, and `uses:` v2 module imports.
3. **Update `references/language.md` & `references/refinements.md`:** Reconcile every discrepancy listed in `spec/SPEC.md:531-551` (frontmatter quoting, reserved child names `constructor`/`in`/`with`, relational `ensures:`, opt-in `mutable: true`).
4. **Update `references/patterns.md`:** Keep the genuinely strong sections (`Checked stages`, `Decisions with floors`, `One set of stages, two drivers`) and prune/rewrite the Astra-derived boilerplate paragraphs that encouraged monolithic prose pipelines.

### 2.6 Quarantining Bulk Legacy Synthetic Corpora
Without breaking any existing test, git history, or training script that references paths on disk:
1. Add clear **`CURATED_INDEX.md` / `README.md`** files at `examples/README.md`, `applications/README.md`, and `codebases/README.md` separating:
   - **Curated Idiomatic Reference Programs** (`examples/{triage,review,specialization}`, `applications/{notebook,helpdesk,nldb,program-improver,memetic-improver}` + WS6 flagships) — *"Read these to learn natlang."*
   - **Legacy / Bulk Synthetic Corpora** (`codebases/design_briefs/` 100 apps, 11 Astra apps in `applications/`, `docs/SEMANTIC_APPLICATION_NOTES.md`) — *"Historical agent-authored stress-test corpora; do NOT copy their `.nl`-only prose-pipeline style for new programs."*
2. Move `applications/`'s 11 Astra apps into `applications/_archive-astra/` (or tag `"role": "archive-synthetic"` in their `natlang.json` + update any script paths referencing them in one clean sweep) so `ls applications/` and `natlang dashboard --root applications` showcase real, idiomatic Semantic FP software.

---

## 3. Acceptance Gates for WS4

- Every single example in `examples/`, `natlang init` templates, and the 12-example Playground gallery passes `natlang check --idiom` with **0 errors and 0 warnings**.
- Browser playground smoke test executes all 12 new examples cleanly in `wllama`.
- Blind agent-authoring evaluation: give 5 fresh coding-agent prompts (*"Build a natlang program for X"*) armed only with the new `natlang init` `AGENTS.md` + `skills/natlang-authoring/`; verify ≥ 4/5 naturally produce idiomatic mixed `.ts` + inline `nl` / `readout: decision` / `Is<T, P>` / `@natlang/std` programs passing `natlang check --idiom`.
