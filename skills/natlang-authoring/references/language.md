# Language and source contracts

This guide describes natlang's scope-eval model-facing architecture. Resolve implementation details using the actual loader, runtime, and tests.

## Source layout

```text
review/
  types.ts
  review.nl
  review/
    assess.nl
    summarize.ts
```

A function sees helpers in its same-named companion directory and explicit `uses` links. It does not inherit the caller's helpers. Folder `types.ts` supplies aliases; inherited aliases and function `types` support shared contracts. Example frontmatter:

```yaml
---
description: Review evidence against a stated criterion.
args:
  observations: string[]
  criterion: string
returns: Report
uses:
  summarize: ../shared/summarize
---
```

`uses` paths are relative to the source file; inspect the target layout before choosing one. In supplied definition graphs, `uses` targets are definition names instead of filesystem paths. Use aliases with legal identifiers. Native packages resolve and install before a run; `uses` never downloads a missing source while a lambda is executing.

A crisp `.ts` module is an ordinary TypeScript module with a default-exported
function:

```ts
export default function summarize(observations: string[], criterion: string): Report {
  return { count: observations.length, criterion };
}
```

Use normal parameters, local variables, imports, and `return`. Natural-language
functions use frontmatter for their signature and instruction lines for their
body. In model eval, typed parameters and imported functions are ordinary
lexical names; the final eval expression can supply the natural-language
function's result.

## Types

Use ordinary TypeScript types: `string`, `number`, `boolean`, `null`, records,
arrays (`T[]`), `Record<string, T>`, named aliases, and literal unions such as
`"supported" | "contradicted" | "uncertain"`. Optional record and parameter
names use `?`. Quote YAML type strings containing record syntax or other YAML
punctuation. Folder aliases use `export type Name = ...;` or `type Name = ...;`.

Types are checked structurally at function boundaries, after eval transactions,
and before completion. Use real TypeScript function declarations for crisp
helpers. Runtime host objects remain host-owned and should be accessed through
an imported helper or, for reducer file operations, the folder API.

Partial return records can be built incrementally; completion requires a complete result of the declared type. Wrongly typed fields are rejected. A validated record can still be semantically false. Exact count checks, semantic rubrics, and external effect receipts address different claims.

## Calls, values, and reuse

In source, write `const assessment = await assess(observation, criterion)` using ordinary positional calls:

```text
const assessment = await assess(observations[0], criterion);
```

File names remain ordinary values:

```text
const evidence = await workspaceRead(args.state.head, "evidence/observations.json");
```

The callee must actually exist with those parameters. Use `read_value` for scope inspection and `read_function` to inspect an imported function. Use `edit_function` and `diff_functions` to inspect and improve existing codebase functions when that would help the task. Directory reducers alone receive file tools and `folder.fs`; paths are relative to the reducer's explicit input folder. Reuse computed values by variable reference.

Imported functions have stable names and signatures. The function tools inspect or edit existing source; the fixed function set cannot be created, deleted, moved, or renamed. A directory reducer's first parameter is an explicit `Folder` value. It can create, edit, move, and remove files within that folder. A direct call `await reducer(folder, ...args)` returns only the typed value and discards file changes. Use `await folder.apply(reducer, ...args)` when selected file changes should be retained in the folder; `folder.dir(path)` selects a subfolder. For generated programs, validate and execute the new artifact before claiming it works.

## Combinators and completion

- Repeated work uses ordinary control flow, for example `await Promise.all(items.map(item => assess(item, criterion)))`, or an explicit loop when order matters.
- A natural-language loop keeps its state in ordinary scope variables and closes each source line with `mark_lines`. Continuation is represented by the persistent scope and line marks, with no separate model command.
- The final eval expression that matches the declared return type completes the function. Chat prose does not set the result. `report_blocker` and `report_error` terminate the episode with a diagnostic.

Completion marks denote executed or skipped program lines. Mark actual work only after success; a range is inclusive, not a pair of isolated line numbers. Avoid using line marks as proof that a semantic result is correct.

## Repository anchors

When a checkout is available, inspect `natlang/codebase.py`, `natlang/types.py`, `natlang/surface.py`, `natlang/runtime.py`, `ts-host/src/native/source-core.ts`, `ts-host/src/native/agent.ts`, and `ts-host/src/native/runtime.ts`. Executable examples include `examples/triage/`, `codebases/dependency_plan/`, and `codebases/order_saga/`. Design/spec context lives in `spec/SPEC.md`, `spec/CODEBASES.md`, and `TYPES.md`; report discrepancies against executable behavior.
