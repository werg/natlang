# WS1: Compiler & Type-System Fixes and Leaps

**Parent:** [Campaign Index](README.md)  
**Goal:** Close every type-system parity bug between `.nl`, inline `nl`, and TypeScript, then lift the five structural ceilings that currently prevent true typed semantic functional programming: lack of generics, second-class higher-order lambdas, directory-tree-bound modules, banned `.nl` structural recursion, and unary-only refinements.

---

## 1. Mechanism Inventory

| Mechanism | Components implementing / consuming it | Scope in WS1 |
|---|---|---|
| Structural type grammar & validation | `ts-host/src/native/types.ts`, `native/values.ts`, `native/agent.ts` (`schemaOf`), `compiler/targets.ts`, `compiler/intrinsics.ts`, `compiler/eval-check.ts` | Full parity across all 6 files |
| Inline `nl` planning & lowering | `compiler/inline.ts`, `compiler/holes.ts`, `compiler/lower.ts`, `runtime/lowered.ts`, `runtime/callable.ts` | Full parity (Node + Browser compiled projects + `eval`) |
| Named `.nl` & callable-folder loading | `runtime/loader.ts`, `runtime/modules.ts`, `runtime/contexts.ts`, `compiler/project.ts` (`.d.nl.ts` emit) | Full parity (`NodeSourceFiles`, browser virtual files, `loadVirtualNatlang`) |
| Recursion & loop termination checks | `runtime/kernel.ts`, `runtime/context.ts`, `compiler/policy.ts` | Full parity across `.nl`, callable-folder `.ts`, and `eval` |
| Refinement & trust checking | `native/refinement.ts`, `native/untrusted.ts`, `compiler/refined-slots.ts`, `compiler/inline.ts`, `runtime/kernel.ts` | Full parity (frontmatter, inline `nl`, `eval`, services) |
| Model serving engines (llama-server, wllama, Python native, SGLang) | Not applicable for type-checker/loader changes — engines receive rendered prompt/tool JSON schemas from `agent.ts` (`schemaOf` and `formatType`). Verified via `ts-host/src/native/agent.ts:1264` and `docs/ENGINE_PARITY.md`. | No wire-protocol change required in WS1 |
| Training collectors & verifiers | `scripts/rl_tasks/verify.mjs`, `structure_smoke.mjs`, `sources/authored.py` | Update `authored.py` & `authored_checks.py` for new frontmatter keys (`uses:` map, `measure:`, `ensures:`, `mutable:`) |

---

## 2. Phase 1A — First-Principles Fixes

Items 1A.1 and 1A.2 were shipped directly on 2026-11-22 (`ts-host/test/semantic-fp-fixes.test.mjs`); items 1A.3 and 1A.4 remain queued for Wave 1.

### 1A.1 ✅ DONE: Fix `Is<T, P>`, `Untrusted<Is<T, P>>`, and Object Intersections in Inline `nl`
- **Shipped (2026-11-22):** `ts-host/src/compiler/targets.ts`, `ts-host/src/compiler/project.ts`, `ts-host/src/compiler/refined-slots.ts`, tested in `ts-host/test/semantic-fp-fixes.test.mjs`. Unwraps `__natlangRefinement` (including nested `Is`) and `__natlangUntrusted`, merges record intersections `A & B`, rejects non-record intersections, and reports inline `refinedSlots` using `natlang` target + merged aliases.

### 1A.2 ✅ DONE: Expose `.in(context)` and Fix `Folder` Types in `intrinsics.ts` & `surface.ts`
- **Shipped (2026-11-22):** `ts-host/src/compiler/intrinsics.ts`, `ts-host/src/runtime/surface.ts`, `ts-host/src/native/scoped-fs.ts`, tested in `ts-host/test/semantic-fp-fixes.test.mjs`. Adds `Context` + `.in(context)` on `NatlangCallableMethods` / `InlineNatlangFunction` / `NatlangUntypedFunction`, introduces `FolderReducer` accepting crisp TS `(folder, ...args) => R | Promise<R>` reducers, and aligns `moveTo` with `string | Folder | FileHandle`.

### 1A.3 Add Tuple Types, `Record<LiteralUnion, V>`, and Core Utility Types
- **Defect:** `compiler/targets.ts:173-179` flattens `[A, B]` tuples into `(A | B)[]`; `native/types.ts` lacks a tuple AST node (breaking `natlang:neuralese`'s `zip` which declares `Promise<N<[A, B]>>` at `compiler/project.ts:212`). `Record<K, V>` rejects literal union keys (`native/types.ts:219-224`), and `.nl` frontmatter lacks `Partial<T>`, `Pick<T, K>`, `Omit<T, K>`, `Required<T>`.
- **Fix:**
  1. Add `{ kind: 'tuple'; elements: Type[] }` to `native/types.ts`, parser (`[T1, T2]`), `formatType`, `fitsType`, `values.ts` (`coerce` checking fixed length + per-position types), `agent.ts` (`schemaOf` emitting JSON Schema `prefixItems` + `minItems`/`maxItems`), and `compiler/targets.ts`.
  2. Allow `Record<LiteralUnion, V>` (desugared during `TypeEnv.resolve` into a record with one required field per literal member).
  3. Support `Partial<T>`, `Required<T>`, `Pick<T, K>`, `Omit<T, K>` in `native/types.ts` `TypeEnv.resolve` over resolved record types.
- **Targeted tests:** `ts-host/test/types-extended.test.mjs` (parsing, `fitsType`, `coerce`, JSON schema emission, `zip` round-trip).

### 1A.4 Move `iteration-unbounded` and `eval` `untrusted-instruction` Checks Earlier
- **Defect 1:** `iteration-unbounded` (`SPEC.md:550-551`) only fires at runtime after the initial predicate fails, not during `natlang check`.
- **Defect 2:** `spec/ext/refinements.md:92-93` notes that `eval` one-shot `nl("... ${x}")` interpolations bypass `untrusted-instruction`.
- **Fix:**
  1. In `compiler/policy.ts`, statically inspect `.iterateOn(...)` chains whose `.until(...)` argument is a TS arrow/function without `.withMeasure(...)` or `.withLimit({ maxSteps })` and report `iteration-unbounded` at `natlang check` time (keeping the runtime check as defense-in-depth).
  2. In `compiler/eval-check.ts`, run the `untrusted-instruction` interpolation check on `eval` snippets using the scope's known `Untrusted<T>` bindings.
- **Targeted tests:** `ts-host/test/policy.test.mjs` and `ts-host/test/untrusted.test.mjs`.

---

## 3. Phase 1B — Ambitious Semantic-FP Language Leaps (Wave 1–2)

### 1B.1 Opt-In `edit_code` (Read-Only Callable Codebases by Default)
- **Problem:** Offering `edit_code` by default on every `.nl` function with a companion folder (`native/agent.ts:1275-1279`) causes stuck models to rewrite their own helpers/verifiers (`skills/natlang-authoring/references/language.md:75`), forcing authors to move pure TypeScript logic out into opaque host services.
- **Design:**
  - Add frontmatter key `mutable: true` (default `false`) to `.nl` files and `codeEdits: 'deny' | 'allow'` runtime option (changing default from `'allow'` to `'deny'` unless `mutable: true` is declared).
  - Read-only `read_code` remains always on so models freely explore rich codebases.
  - Self-modification remains first-class through explicit `mutable: true` or transactional `Folder` reducers (`folder.propose` / `folder.accept`).
- **Opt-out:** Runtime option `codeEdits: 'allow'` or env `NATLANG_LEGACY_EDIT_CODE=1`.
- **Spec & test updates:** Update `spec/SPEC.md` ("Model surface"), `loader.ts`, `agent.ts`, `conformance/programs/17-reopen-refine.yaml` (add `mutable: true`), and `ts-host/test/code-edits.test.mjs`.

### 1B.2 First-Class Higher-Order Semantic Functions, Factories & Currying
- **Problem:** Writing a function that accepts a semantic policy lambda or compiles prose into a typed semantic closure (`compileRule(doc: string) => Promise<(x: Event) => Promise<Action>>`) hits three walls:
  1. `compiler/inline.ts:369-390` rejects factory arrows returning uncalled `nl` (`nl-not-called`).
  2. `native/values.ts:200-203` checks `wanted.kind === 'lambda'` with only `typeof raw === 'function'`, ignoring arity and parameter/return types.
  3. Named `.nl` functions have no `.partial(...)` currying helper, and `return_result` cannot return a function directly (only `eval { finish: true }`).
- **Design:**
  1. **Relax `nl-not-called`:** Permit an uncalled `nl` tag whenever it is in return position of a function/arrow, an object/array literal property typed as a function, or passed to a higher-order parameter expecting a callable (`compiler/inline.ts`). Keep `nl-not-called` only for bare expression statements (`nl\`...\`;`).
  2. **Typed callable wrappers at runtime:** Tag compiled inline `nl`, named `.nl`, and TS callable exports with their runtime `Type` (`{ kind: 'lambda', params, returns }`) via a non-enumerable symbol `NATLANG_SIGNATURE`. In `values.ts:200`, when a value carries `NATLANG_SIGNATURE`, verify `fitsType(gotSig, wanted, env)` contravariantly! For plain JS functions without metadata, wrap them in a **boundary-checking proxy** that validates incoming arguments and outgoing returns against `wanted` at call time.
  3. **`.partial(...boundArgs)` on `.nl` callables:** Add `fn.partial(boundRecord)` (or positional partial binding) to `NatlangCallableMethods` (`runtime/callable.ts`, `compiler/intrinsics.ts`), preserving `.iterateOn`, `.in`, `.on`, and the narrowed runtime signature.
  4. **Clear `eval` guidance for returning functions:** When a `.nl` function declares a lambda return type (`returns: (e: Event) => Decision`), `agent.ts` tells the call to build and return the function via `eval({ code: "return nl<...>`...`;", finish: true })` (omitting `value` schema confusion on `return_result`).
- **Targeted tests:** `ts-host/test/higher-order.test.mjs` — passing inline `nl` into a `.nl` function, returning a compiled `nl` closure from a `.nl` function and calling it from host TS, contravariant signature rejection, and `.partial(...)`.

### 1B.3 Parametric Polymorphism (Monomorphized Generics) for `.nl` and Inline `nl`
- **Problem:** `compiler/targets.ts:127` rejects all TypeScript `TypeParameter`s and `loader.ts:272-282` restricts `generic:` to `R: T | Neuralese<T>`. Authors cannot write `@natlang/std` combinators (`semanticFilter<T>`, `semanticMerge<T>`, `semanticReduce<S, A>`).
- **Design:**
  - Extend frontmatter `generic:` to declare arbitrary bounded or unbounded type parameters:
    ```yaml
    generic:
      T: unknown
      K: string
    args:
      items: T[]
      rubric: Record<K, string>
    returns: Record<K, T[]>
    ```
  - **Call-site monomorphization (compile-time):** Just as `checkGenericResults` already instantiates `R: T | Neuralese<T>` per call site (`compiler/neuralese.ts`), resolve concrete type arguments `T`, `K` from TypeScript's call-site inference (`checker.getResolvedSignature(call)`), substitute them into a monomorphized `TargetDescriptor`, and pass `{ typeArgs: { T: 'Ticket', K: '"bug" | "feature"' } }` to `invokeDefinition`.
  - **Generic TS wrapper functions around inline `nl`:** Allow a generic TS helper like `export async function semanticFilter<T>(items: T[], criterion: string): Promise<T[]>` when `nl` is either passed an explicit runtime type descriptor (auto-injected by the compiler at monomorphic call sites via a compiler intrinsic `typeOf<T>()`) or when `T` is concrete at the lowered call site!
  - **Zero extra model burden:** At execution time, the model sees the concrete instantiated types (`Ticket[]`, `Record<"bug" | "feature", Ticket[]>`), never unresolved type variables.
- **Opt-out:** `--no-generic-nl` compiler flag during rollout.
- **Targeted tests:** `ts-host/test/generic-nl.test.mjs` covering monomorphic instantiation, constraint violations, nested generic types, and call-record provenance (recording instantiated signature alongside definition ID).

### 1B.4 Real Acyclic Module System (`uses:` v2)
- **Problem:** Companion folders force every shared helper into brittle basename-only package-local `uses:` lists (`loader.ts:290-292, 341-373`); callable-folder `.ts` cannot use `uses:` (`modules.ts:256-258`); npm semantic libraries are impossible; `.d.nl.ts` explodes into 10 KB single-line intersections (`project.ts:160-193`).
- **Design:**
  1. **Aliased & package-scoped `uses:` map** (backward-compatible with existing list form):
     ```yaml
     uses:
       git: "@natlang/lib-git/inspect"
       rubric: "./shared/rubric"
       verify: "../checks/citations"
     ```
     - Relative paths (`./`, `../`) resolve from the `.nl` file within the package (or workspace).
     - Bare package specifiers (`@scope/pkg/item` or `pkg/item`) resolve via standard Node/package `exports` + `natlang.json` manifest of the dependency package.
     - Explicit alias keys eliminate basename collisions.
  2. **Acyclic DAG guarantee preserved statically:** The loader builds the complete module call graph across companion folders + `uses:` (both `.nl` and `.ts` modules) and rejects any cycle (unless allowed by structural recursion in 1B.5) with the exact cycle path.
  3. **Callable-folder `.ts` imports:** Permit relative imports within the package (`../shared/util.js`, `./sibling.nl`) and `@scope/pkg` natlang exports in callable-folder `.ts`, subject to the same static DAG + finite-iteration policy (`modules.ts`).
  4. **Clean `.d.nl.ts` emit (`project.ts:160-193`):** Emit `import type` references to child declarations instead of inlining the entire transitive tree into one giant intersection type.
- **Opt-out:** Existing list `uses: [a/b]` continues working identically.
- **Targeted tests:** `ts-host/test/uses-modules.test.mjs` (aliases, relative paths, cross-package npm `.nl` import, cycle detection, callable-folder `.ts` shared imports, compact `.d.nl.ts` snapshot).

### 1B.5 Structural Semantic Recursion for `.nl` Functions
- **Problem:** `runtime/kernel.ts:543-545` unconditionally rejects any `.nl` re-entry, even when the recursive call is on a strictly smaller subtree/slice/integer — forcing authors to manually defunctionalize tree walks and divide-and-conquer into giant `iterateOn` state machines.
- **Design:**
  1. Add optional frontmatter key `recursive: structural` (or `measure: <expr>`, e.g., `measure: node.depth` or `measure: items.length`). Without it, `.nl` functions remain strictly non-recursive by default (preserving existing guarantees and error messages).
  2. In `runtime/kernel.ts:543-545`, when `definition.recursive` is enabled, reuse `runtime/context.ts:223-258`'s exact structural-descent checker across the active `.nl` call chain: each self-call must strictly decrease along the same parameter (`part` reachable via property/element, shorter array/string, smaller non-negative integer) or strictly decrease the declared nonnegative integer `measure`. Non-decreasing or repeated arguments throw `NatlangRecursionError`.
  3. Expose the function's own name in its eval scope when `recursive` is declared so eval code can invoke `await self(child)` directly (including concurrently across disjoint children with `Promise.all(node.children.map(c => summarize(c)))`!).
  4. Replace the fixed `MAX_AD_HOC_NL_DEPTH = 5` (`kernel.ts:524-529`) with a soft descent rule for ad-hoc `nl` (allow deeper nesting when total serialized input size strictly shrinks by ≥ 25%, capped at a safety ceiling of 16; keep 5 as fallback when inputs do not shrink).
- **Opt-out:** `NATLANG_ALLOW_NL_RECURSION=0` runtime switch.
- **Targeted tests:** `ts-host/test/nl-recursion.test.mjs` (recursive tree summarizer, parallel divide-and-conquer, rejection of equal/growing argument, measure exhaustion, mutual cycle still rejected unless measured).

### 1B.6 Relational / Dependent Refinements & `ensures:` Postconditions
- **Problem:** `native/refinement.ts:94-97` shows the judge only the isolated field value (`renderForJudge(value)`), making relational invariants (`quote: Is<string, "verbatim substring of source">`) impossible to check even though `skills/natlang-authoring/references/refinements.md:8` teaches them.
- **Design:**
  1. **Frontmatter `ensures:` contracts** on `.nl` functions (both crisp TS expressions and natural-language predicates with access to `result` and all parameters):
     ```yaml
     args:
       source: Untrusted<string>
       question: string
     returns: CitedAnswer
     ensures:
       - check: "result.quotes.every(q => source.includes(q))"
         message: "Every quote must be a verbatim substring of source."
       - predicate: "result.answer is supported solely by result.quotes and answers question"
         against: [question, source]
     ```
  2. **Contextual `Is<T, P>` resolution:** When `Is<T, P>` appears inside a return type or `refine(value, predicate, { context })`, allow passing `{ context }` bindings to `refinement.ts` so the judge sees `{ value, context }` cleanly separated (with `Untrusted<T>` fencing preserved!).
  3. **Execution & repair loop:** `ensures:` checks run after structural type-checking at `return_result` / `eval { finish: true }`:
     - Crisp `check:` runs first (0 tokens, deterministic, constrained TS expression evaluated in a read-only scope holding `result` + args).
     - Semantic `predicate:` runs via cached `readout: decision` judge (or crisp override from `refinements.ts`).
     - Failures feed the exact `message` / predicate back to the model inside the existing repair budget (`maxFailureRepairs`), throwing `RefinementCallError` if exhausted.
- **Opt-out:** `refinements.ensures: 'off'` in `natlang.json`.
- **Targeted tests:** `ts-host/test/relational-refinements.test.mjs` (crisp `ensures`, semantic `ensures` with `against:`, repair loop recovery, `Untrusted<T>` fencing inside judge prompt, caching key includes context hash).

---

## 4. Acceptance Gates for WS1

- `npm test` in `ts-host` passes 100%, including all 7 new targeted test suites above.
- Browser bundle build + `npm run test:browser` pass with identical diagnostics and runtime semantics.
- `spec/SPEC.md`, `spec/ERRORS.md`, `spec/ext/refinements.md`, and `spec/ext/directory-reducers.md` are updated in the same pushes as each feature lands.
- Zero regressions on existing `conformance/programs/` and `applications/` test suites.
