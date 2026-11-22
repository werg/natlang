# WS3: `@natlang/std` Semantic Standard Library & Idiom Linter

**Parent:** [Campaign Index](README.md)  
**Depends on:** WS1 (Phase 1A fixes; Phase 1B generics & higher-order functions for full polymorphic surface) and WS2 (`escalate`, factorized `readout: decision`).  
**Goal:** Give authors a crisp, battle-tested vocabulary of **higher-order semantic functional combinators** (so they stop hand-rolling 300-word prose loops) AND an automated **idiom linter (`natlang check --idiom`)** that guides both human and coding-agent authors away from anti-patterns immediately at check time.

---

## 1. `@natlang/std`: The Semantic Functional Standard Library

Today `codebases/std/` contains only two 3-line helpers (`any_true.ts`, `select_by_flags.ts`). Meanwhile, every application reinvents semantic filtering, partitioning, fuzzy joining, bounded citation repair, semantic merging, and confidence-gated routing from scratch — usually poorly, inside giant `.nl` prose blocks.

We will ship `@natlang/std` (exported from `@natlang/node/std` and `@natlang/browser/std`, and importable inside `.nl` via `uses:` v2 from WS1) with **four composable combinator families**:

### 1.1 Semantic Collection Combinators (`std/collections.ts`)
All parallel by default (bounded by driver concurrency), strongly typed, and automatically using `readout: decision` where finite:

| Combinator | Signature | Semantics & Implementation |
|---|---|---|
| `semanticFilter` | `<T>(items: readonly T[], predicate: string \| ((x: T) => Promise<boolean>), opts?: { floor?: number; fallback?: string }) => Promise<T[]>` | Runs a `readout: decision` boolean predicate concurrently over `items` (with optional confidence escalation) and filters in exact TS. |
| `semanticPartition` | `<T, K extends string>(items: readonly T[], rubric: Record<K, string>, opts?: EscalateOpts) => Promise<Record<K, T[]>>` | Classifies each item into closed key set `K` via `readout: decision` and groups in exact TS (preserving order, initializing all keys). |
| `semanticMap` | `<A, B>(items: readonly A[], fn: (item: A, index: number) => Promise<B>, opts?: { concurrency?: number }) => Promise<B[]>` | Bounded-concurrency typed semantic map (prevents flooding local 1.7B slots on 500-item lists while keeping trace grouping clean). |
| `semanticReduce` | `<S, A>(items: readonly A[], initial: S, step: (acc: S, item: A) => Promise<S>, opts?: { mode?: 'linear' \| 'tree'; neuralese?: boolean }) => Promise<S>` | Exact `for...of` fold for `'linear'`, or balanced binary tree reduction for associative summaries (`'tree'`), with optional `Neuralese<S>` intermediate state carry so large docs don't re-tokenize at every step. |
| `semanticMatch` | `<A, B>(left: readonly A[], right: readonly B[], relation: string, opts?: { blockBy?: (x: A \| B) => string; threshold?: number }) => Promise<Match<A, B>[]>` | Semantic relational join! Optional exact blocking key prunes $O(N \times M)$ pairs in TS; `readout: decision` scores candidate pairs; returns typed matches + unmatched leftovers + probabilities. |
| `semanticRank` | `<T>(items: readonly T[], criterion: string, opts?: { topK?: number }) => Promise<Ranked<T>[]>` | Tournament / pairwise `readout: decision` comparator (`a` beats `b`?) + exact Copeland/Elo ranking in TS — vastly more reliable on 1.7B models than asking a small model to sort 40 JSON objects in one shot! |

### 1.2 Convergence, Merge & Repair Combinators (`std/converge.ts`)
| Combinator | Signature | Semantics & Implementation |
|---|---|---|
| `semanticRepair` | `<T>(initial: T, check: (x: T) => string \| null \| Promise<string \| null>, fix: (x: T, problem: string) => Promise<T>, opts: { maxSteps: number }) => Promise<RepairResult<T>>` | Standardizes the checked-stage pattern (`notebook`, `compilers`, `board`): runs exact `check`, feeds concrete error back into `fix` via `iterateOn`, returns `{ ok, value, attempts, history }` instead of throwing away the best honest state. |
| `semanticMerge` | `<T>(base: T, edits: readonly Edit<T>[], opts: { policy?: string; check?: (m: T) => string \| null }) => Promise<MergeResult<T>>` | Extracts per-edit intent, checks pairwise compatibility via `readout: decision`, merges compatible hunks, preserves genuine conflicts explicitly in `unresolved[]`, and runs `semanticRepair` until `check` passes. |
| `fixedPoint` | `<T>(initial: T, step: (s: T) => Promise<T>, same: (a: T, b: T) => boolean \| Promise<boolean>, opts?: { maxSteps?: number }) => Promise<T>` | Semantic fixed-point iteration (`nlprolog` pattern) built directly over `iterateOn` with automatic cycle/oscillation reporting. |

### 1.3 Higher-Order Policy, Routing & Context Combinators (`std/policy.ts`)
| Combinator | Signature | Semantics & Implementation |
|---|---|---|
| `escalate` | `(fn, { floor, fallback })` | Re-exported from WS2: calibrated `readout: decision` on 1.7B with automatic escalation on low confidence or risk-floor breach. |
| `compileRule` | `<I, O>(proseRule: string, targetType: TypeSpec<O>, opts?: { examples?: Pair<I, O>[] }) => Promise<(input: I) => Promise<O>>` | **Higher-order semantic compiler:** takes evolving user/business rules in prose and returns a typed, checkable, specializable callable `(input: I) => Promise<O>` (validated against `examples` before returning!). |
| `withContext` | `<F extends NatlangCallable>(fn: F, overrides: Record<string, unknown>) => F` | Ergonomic wrapper over `Context.ofCallable(fn).with(overrides)` + `fn.in(...)`, making context-based dependency injection and lens swapping trivial. |
| `pluggable` | Re-exported from `@natlang/node` | Crisp / NL / Shadow switching with `pluggable_shadow` agreement telemetry. |

### 1.4 Transactional Workspace Combinators (`std/folder.ts`)
| Combinator | Signature | Semantics & Implementation |
|---|---|---|
| `proposeAndVerify` | `<R>(folder: Folder, reducer: FolderReducer<R>, verify: (candidate: Folder, value: R) => Promise<VerifyResult>) => Promise<CommitResult<R>>` | Wraps `folder.propose` → exact `verify(proposal.folder)` → `folder.accept(proposal)` (or automatic rollback with structured diagnostics). |
| `branchAndSelect` | `<R>(folder: Folder, candidates: FolderReducer<R>[], score: (candidate: Folder, value: R) => Promise<number>) => Promise<SelectResult<R>>` | Runs `snapshot.branch()` per candidate concurrently, evaluates each independently, and installs the winning snapshot with `folder.select(winner)`. |

### Design Discipline for `@natlang/std`
- Every combinator runs in **both Node and Browser** with zero Node-only dependencies.
- Every combinator produces clean, nested call records so `natlang traces`, `natlang profile`, and `natlang specialize` see the exact per-item/per-stage tree.
- Every combinator ships with deterministic mock-driver unit tests AND a live LFM3-1.7B smoke test.

---

## 2. The Idiom Linter (`natlang check --idiom`)

Coding agents treat `natlang check` diagnostics as law. Right now, `natlang check` verifies only syntax, types, scope, and loop termination — so a 700-word imperative prose `.nl` file with zero `.ts` helpers passes with `ok: true` and 0 warnings!

### 2.1 Linter Architecture
Add an opt-outable idiomatic analysis pass in `ts-host/src/compiler/idiom.ts`, wired into `natlang check`:
- **Default mode (`natlang check`):** Emits `severity: 'warning'` diagnostics (never breaks existing builds).
- **Strict mode (`natlang check --idiom` / `"check": { "idiom": "error" }` in `natlang.json`):** Promotes idiom violations to errors (exit code 1). Used in CI for all new flagship apps, playground examples, and `RL_TASK_AUTHORING_KIT` v2 qualification!
- **Per-site suppression:** Frontmatter `idiom-ignore: [rule-name]` or `// natlang-idiom-ignore rule-name` with mandatory reason comment.

### 2.2 The 8 Idiom Rules

| Code | What It Detects | Concrete Fix Hint Emitted |
|---|---|---|
| `idiom-prose-algorithm` | A `.nl` body ≥ 150 words containing deterministic code patterns (`Math.`, `.slice(`, `JSON.parse`, `eval({code:`, arithmetic/copying steps) whose companion folder has 0 `.ts` helpers. | *"Move exact arithmetic, array slicing, and record assembly into a `.ts` helper in the companion folder; keep `.nl` for the semantic judgment."* |
| `idiom-nl-only-tree` | A `.nl` function tree of ≥ 4 `.nl` files with 0 `.ts` helpers and 0 declared services. | *"Pure `.nl`-only pipelines force the model to interpret plumbing in prose; extract exact invariants and data plumbing into `.ts` helpers."* |
| `idiom-missing-decision-readout` | A leaf `.nl` function (empty companion folder, no `uses:`, no `Folder` param) returning a small finite union (≤ 16 values) or `boolean` without `readout: decision`. | *"Add `readout: decision` to score the finite return values in one calibrated pass instead of a multi-turn tool loop."* |
| `idiom-fake-iterate` | `.iterateOn(...)` whose state only advances a fixed array cursor (`index + 1` / `pending.slice(1)`) without adaptive state dependency between items. | *"Use `Promise.all(items.map(...))` or `semanticMap`/`for...of` for independent items; reserve `iterateOn` for adaptive convergence or state-dependent loops."* |
| `idiom-redundant-reask` | A `.nl` body that instructs the model to create an inline `nl` / `nl.with` performing the same judgment on the same input it was called with. | *"Judge the supplied input directly in this call instead of spawning an identical inline `nl` child."* |
| `idiom-untrusted-string-boundary` | A `.nl` function whose parameter name/description signals external text (`email`, `html`, `ticket`, `log`, `document`, `userMessage`, `body`, `issue`, `prDiff`) typed as plain `string` rather than `Untrusted<string>`, OR instructions containing prompt-injection guard prose (*"ignore instructions in..."*). | *"Type external text as `Untrusted<string>` instead of guarding it in prose; the runtime fences untrusted data automatically."* |
| `idiom-prose-constraint` | Instructions containing checkable value-property guard sentences (*"must be under N characters"*, *"one line only"*, *"copied verbatim from"*, *"no markdown"*) without an `Is<T, P>` or `ensures:` contract. | *"Express value constraints in `Is<T, P>` or `ensures:` so the runtime checks and repairs them mechanically."* |
| `idiom-cue-word-checklist` *(RL authoring mode)* | A `.nl` helper defining literal keyword/substring matching rules (*"if it contains the word 'freeze'"*, *"case-sensitive cue"*). | *"Keyword matching belongs in exact TypeScript, not `.nl`. Write a genuine semantic criterion that survives paraphrase."* |

---

## 3. Targeted Tests & Acceptance Gates

- **Tests:**
  - `ts-host/test/std-collections.test.mjs`, `std-converge.test.mjs`, `std-policy.test.mjs`, `std-folder.test.mjs` (both Node and Browser).
  - `ts-host/test/idiom-lint.test.mjs` testing every one of the 8 rules (true positives, false-positive exclusions, `idiom-ignore` suppression, `--idiom` exit code).
- **Acceptance gate:**
  - Running `natlang check --idiom` catches the known anti-patterns in `applications/cave-mission`, `applications/pi/generation/answer.nl`, and `ts-host/playground/examples/natural.mjs`, while passing cleanly (0 errors) on `examples/triage`, `examples/review`, `applications/notebook`, `applications/helpdesk`, and all new flagship apps.
