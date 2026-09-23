---
name: natlang-authoring
description: Author, refactor, review, and validate natlang code — inline `nl` calls in TypeScript, named `.nl` functions with their callable folders, and `natlang.d/` application context. Use when implementing semantic algorithms, reducers, generated programs, or application logic with natlang, including long-running work intended for small interpreter models.
---

# Author natlang code

You are the capable program author; the interpreter may be a small model. Put the algorithm, decomposition, data contracts, and recovery decisions in source so execution does not depend on the interpreter inventing an architecture.

Natlang is TypeScript with natural-language functions. Ordinary TypeScript owns exact computation, I/O, and orchestration. A natural-language function owns a judgment: interpretation, planning, semantic merging, choosing among options, explaining. Choose the boundary deliberately. A host that computes everything and asks natlang to approve wastes it; making the model do arithmetic by hand wastes its context.

## Establish the contract

Locate the natlang checkout or installed `@natlang/node` / `@natlang/browser` version first. Use its declarations (`dist/index.d.ts`), tests, and `natlang check` as the authority; if prose here disagrees with executable behavior, flag the disagreement rather than inventing behavior. Outside a checkout, rely on these bundled references and the installed package.

Read [language and source contracts](references/language.md) before creating or restructuring code, [algorithm patterns](references/patterns.md) for loops, reducers, reconciliation, and generated methods, and [verification and diagnosis](references/verification.md) when testing or debugging. The [review example](assets/review/review.nl) shows a named function whose callable folder holds a semantic helper and an exact aggregation; copy the whole `assets/review/` directory.

## Build a real program

1. State the behavior as typed inputs, results, and observable scenarios, including ambiguity, empty inputs, and partial failure. Types check structure, not meaning.
2. Write the orchestration in TypeScript. Call natural language where judgment is needed: inline with `` nl`…` `` for a one-off decision, or a named `.nl` function when the instruction deserves its own file, its own helpers, or reuse.
3. Give each natural-language function one coherent responsibility, a precise signature, and the helpers it may call in its callable folder (`foo/` beside `foo.nl`, or `natlang.d/` for inline calls in application code).
4. Keep exact work exact: helpers in callable folders are ordinary TypeScript under a finite-iteration policy; application code outside them is unrestricted.
5. Run `natlang check` (types, `nl` signatures, callable-folder policy), then exercise the real source through the runtime. Distinguish checks, scripted wiring, live-model runs, and semantic evaluation.

## Preserve the execution model

- A natural-language call is an ordinary awaited function call returning a checked, typed value; a failure rejects with `NatlangCallError` (`outcome`, `detail`, `trace`). Catch it where the application can recover.
- An inline `nl` reads the variables its instructions mention by exact name, live at call time. Mutable captures (`let`) are written back after a successful eval; a concurrent change makes that eval fail with `capture-conflict` so it retries against the current value. Rebinding a captured `const` is a compile error.
- Signatures are inferred before the model runs: from the contextual type, immediate call arguments, and later uses. When nothing determines the result type, `natlang check` reports `nl-unknown-return`; annotate the binding or use `nl<T>`.
- A function may not appear in its own chain of callers: no direct or mutual recursion among natural-language functions and callable-folder TypeScript. Concurrent sibling calls (`Promise.all`) and repeated sequential calls are fine.
- Callable-folder TypeScript and eval code use finite iteration: `for...of`, counted `for (let i = 0; i < n; i++)`, and array methods. `while`, `do`, `for...in`, open `for(;;)`, and generators are rejected. Open-ended refinement uses `iterateOn`, which records every step and reviews progress.
- Host capabilities come from services: `import { wiki } from 'natlang:services'` in callable-folder code, or the same names as bindings inside eval. Every service call is traced as an effect and is not rolled back.
- Directory reducers (`kind: directory-reducer`) receive a `Folder` as their first argument and file tools rooted there. A direct call returns the typed value and discards file changes; `folder.apply(reducer, ...args)` retains the committed changes.
- Long productive runs are intended. Do not add arbitrary turn, token, or nesting limits to make a test finish; deployment budgets are runtime options.
- Report real blockers. Do not return invented success, manufacture receipts, weaken assertions, or treat an unknown effect outcome as safe to retry.

For embedding in an application, pair this skill with `natlang-integration` when available; this skill is usable on its own.
