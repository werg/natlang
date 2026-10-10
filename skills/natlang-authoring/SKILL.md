---
name: natlang-authoring
description: Author, refactor, review, evaluate, and adapt natlang code — inline `nl` calls in TypeScript, named `.nl` functions with their callable folders, and `natlang.d/` application context. Use for semantic algorithms, reducers, application logic, or measured instruction and program-guidance optimization, including programs intended for small interpreter models.
---

# Author natlang code

You are the capable program author; the interpreter may be a small model. Put the algorithm, decomposition, data contracts, and recovery decisions in source so execution does not depend on the interpreter inventing an architecture.

Natlang is TypeScript with natural-language functions. Ordinary TypeScript owns exact computation, I/O, and orchestration. A natural-language function owns a judgment: interpretation, planning, semantic merging, choosing among options, explaining. Choose the boundary deliberately. A host that computes everything and asks natlang to approve wastes it; making the model do arithmetic by hand wastes its context.

## Establish the contract

Locate the natlang checkout or installed `@natlang/node` / `@natlang/browser` version first. Use its declarations (`dist/index.d.ts`), tests, and `natlang check` as the authority; if prose here disagrees with executable behavior, flag the disagreement rather than inventing behavior. Outside a checkout, rely on these bundled references and the installed package.

Read [language and source contracts](references/language.md) before creating or restructuring code, [algorithm patterns](references/patterns.md) for loops, reducers, reconciliation, and generated methods, and [verification and diagnosis](references/verification.md) when testing or debugging. The [review example](assets/review/review.nl) shows a named function whose callable folder holds a semantic helper and an exact aggregation; copy the whole `assets/review/` directory.

Put a property of a value in its type rather than in a guard sentence: read [constraints belong in types](references/refinements.md) for `Is<T, "predicate">`, when to use it and how to phrase the predicate.

To let the runtime keep an intermediate value as Neuralese between stages, write the chain so its data flow is explicit: read [orchestrators that can fuse](references/fusion.md).

For evaluation suites, instruction optimization, or trainable program guidance, read [adaptation and evaluation](references/adaptation.md). It covers component selection, independent scoring, fresh fixtures, bounded search, report review, and artifact activation. Ordinary authoring does not require an optimization run.

## Build a real program

1. State the behavior as typed inputs, results, and observable scenarios, including ambiguity, empty inputs, and partial failure. Types check structure, not meaning.
2. Write the orchestration in TypeScript. Call natural language where judgment is needed: inline with `` nl`…` `` for a one-off decision, or a named `.nl` function when the instruction deserves its own file, its own helpers, or reuse.
3. Give each natural-language function one coherent responsibility, a precise signature, and the helpers it may call in its callable folder (`foo/` beside `foo.nl`, or `natlang.d/` for inline calls in application code). Put only the program's own code there; the systems it acts on (stores, boards, simulators, verifiers, APIs) are services with declarations, scoped to the functions that should use them.
4. Write each instruction so a small interpreter can take the next step: name the data and how to reach it, state the step a loop repeats, and keep the words for finishing statuses ("missing", "cannot") for real blockers. See [patterns](references/patterns.md).
5. Keep exact work exact: helpers in callable folders are ordinary TypeScript under a finite-iteration policy; application code outside them is unrestricted.
6. Run `natlang check` (types, `nl` signatures, callable-folder policy), then exercise the real source through the runtime. Distinguish checks, scripted wiring, live-model runs, and semantic evaluation.

## Improve instructions with measured adaptation

Use adaptation when the callable contracts and decomposition are sound and the task is to improve instructions or program guidance against explicit scenarios. Fix missing capabilities, types, capture names, or helper structure in source first; adaptation changes static instruction text, not those contracts. Label authored inline sites when stable selection matters, keep expected answers outside model-visible code, and use fresh fixtures with independent scoring. Review held-out gates, helper coverage, actual usage and uncertainty before adopting an artifact. A retained baseline is a valid result.

Keep search overlays separate from authored source. Load a compatible artifact explicitly, use `adaptation: null` for a baseline task, and revalidate after source or model changes. Read the [adaptation reference](references/adaptation.md) for SDK and CLI examples; use `natlang-integration` when wiring an application host or browser deployment.

## Preserve the execution model

- A natural-language call is an ordinary awaited function call returning a checked, typed value; a failure rejects with `NatlangCallError` (`outcome`, `detail`, `trace`). Catch it where the application can recover.
- An inline `nl` reads the variables its instructions mention by exact name, live at call time. Mutable captures (`let`) are written back after a successful eval; a concurrent change makes that eval fail with `capture-conflict` so it retries against the current value. Rebinding a captured `const` is a compile error.
- Signatures are inferred before the model runs: from the contextual type, immediate call arguments, and later uses. When nothing determines the result type, `natlang check` reports `nl-unknown-return`; annotate the binding or use `nl<T>`.
- In eval, an inline result annotation can use only type names declared in that eval scope. If `nl-undeclared-type` reports that a name is missing, declare the type in scope or write the needed structure directly in the annotation; an alias mentioned only in earlier instruction text does not type the child.
- A natural-language function may not appear in its own chain of callers. A TypeScript function may call itself only on a smaller argument: a part of its input, a shorter array or string, or a smaller non-negative integer (walking a tree recursively is fine). Concurrent sibling calls (`Promise.all`) and repeated sequential calls are fine.
- Callable-folder TypeScript and eval code use finite iteration: `for...of`, counted `for (let i = 0; i < n; i++)` (the bound is read once, when the loop starts; `&&` may join it to an early exit such as `i < n && !found`), and array methods. `for...of` takes an array, string, Map or Set, plus `entries()`, `keys()` and `values()` of those and `string.matchAll(regex)`, which are finite views of a fixed collection; spread any other iterable first (`for (const x of [...iterable])`). `natlang check` reports a refused loop source with its location. `while`, `do`, `for...in`, open `for(;;)`, generators, hand-written iterators and `setInterval` are rejected. Open-ended refinement uses `iterateOn`, which records every step and reviews progress; a step can wait with `await new Promise(r => setTimeout(r, ms))`. Timers still pending when a call finishes are cleared. `for await` consumes async iterables that services, packages or `fetch` provide.
- Await the natural-language calls you start. A call that fails stops the calls it started; in eval, calls still running when the eval ends, or when its `timeout_ms` passes, are stopped (quietly for the losers of `Promise.race`). Service calls already made stay made.
- Host capabilities come from services: `import { wiki } from 'natlang:services'` in callable-folder code, or the same names as bindings inside eval. Every service call is traced as an effect and is not rolled back. A service's declaration is what the model knows of it; a scoped service is usable only in its functions' calls.
- Nothing the interpreter can reach is private: it reads callable items, service declarations, and packages with `read_code`, and edits callable items with `edit_code`. Never let correctness depend on it not reading source; limit what it can use (services, scopes) instead.
- Directory reducers (`kind: directory-reducer`) receive a `Folder` as their first argument and file tools rooted there. A direct call returns the typed value and discards file changes; `folder.apply(reducer, ...args)` retains the committed changes.
- Long productive runs are intended. Do not add arbitrary turn, token, or nesting limits to make a test finish; deployment budgets are runtime options.
- Report real blockers. Do not return invented success, manufacture receipts, weaken assertions, or treat an unknown effect outcome as safe to retry.

For embedding in an application, pair this skill with `natlang-integration` when available; this skill is usable on its own.
