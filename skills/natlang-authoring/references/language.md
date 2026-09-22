# Language and source contracts

This guide describes the scope-eval-v1 model-facing architecture. Resolve implementation differences using the actual loader, runtime, and tests.

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
  observations: Text[]
  criterion: Text
returns: Report
uses:
  summarize: ../shared/summarize
---
```

`uses` paths are relative to the source file; inspect the target layout before choosing one. In supplied definition graphs, `uses` targets are definition names instead of filesystem paths. Use aliases with legal identifiers. Native packages resolve and install before a run; `uses` never downloads a missing source while a lambda is executing.

A crisp file uses `/*---` through `---*/` frontmatter followed by a function body. Inputs are `args`, the body returns its value with `return`. Inline `run_code` instead returns its final expression. Exact portable JS bodies can omit `engine` for the compatibility loader behavior; select `engine: typescript-host` when using the native TS host. Python's default evaluator is `quickjs-isolated`; additional executor names exist only when the embedding registers them. Inline eval must supply the offered engine when the current schema requires it.

## Types

Use `Text`, `Num`, `Bool`, `Null`, records, `T[]`, `Dict<T>`, named aliases, and unions such as `"supported" | "contradicted" | "uncertain"`. Optional record/parameter names have `?`. Quote YAML type strings containing record syntax or other YAML punctuation. Folder aliases use `export type Name = ...;` or `type Name = ...;`. Nested records accept commas or semicolons.

This is a structural subset, not the TypeScript compiler's type system. Do not assume `any`, interfaces, imported types, generics, methods, branded host objects, or automatic inference. Even when a type name such as `Blob` exists, check the target's representation before using it. Host buffers, processes, database connections, and DOM nodes normally remain host-owned and are accessed by portable IDs or crisp environment code.

### Host-backed dictionaries

An embedding can bind a lazy, read-only implementation of the existing
`Dict<T>` type. Natlang source does not declare `LazyDict`, `Tree`, a provider,
or a special namespace. This is useful when enumerating or materializing the
whole input would be expensive. A project-aware function can declare:

```yaml
---
args:
  files: Dict<ProjectFile>
types:
  ProjectFile: '{ kind: "text", text: Text, bytes: Num } | { kind: "binary", bytes: Num }'
returns: Report
---
```

The root lists at `args/files`; branches and leaves are read through ordinary
paths such as `args/files/docs/design.md/text`. A filesystem adapter supplies
the record at the leaf and leaves binary bytes outside model state. The host
loads each observed directory or leaf at most once for that bound value, so a
run sees a stable observation even when the backing store later changes.

Treat this like any other argument. A helper that needs the mapping declares a
compatible `Dict<ProjectFile>` parameter and receives it through
`call.inputs`, for example `{"files":"args/files"}`. It is not ambient state
and should not be copied leaf by leaf. Eager dictionaries continue to behave as
before.

The lazy implementation is read-only and not a portable snapshot. Returning or
persisting selected typed leaves is supported; process restart requires the
host to bind the provider again. Crisp code that needs arbitrary filesystem or
database access should use its real host API. A crisp function cannot receive a
host-backed dictionary as an argument. Inline `run_code` remains usable for
other values, but provider-backed fields are omitted from its portable scope.
Do not force the runtime to materialize a native provider merely to pass it
into eval.

Partial return records can be built incrementally; completion requires a complete result of the declared type. Wrongly typed fields are rejected. A validated record can still be semantically false. Exact count checks, semantic rubrics, and external effect receipts address different claims.

## Calls, values, and reuse

In source, write `const assessment = await assess(observation, criterion)` using ordinary positional calls:

```text
const assessment = await assess(args.observations[0], args.criterion);
```

File names remain ordinary values:

```text
const evidence = await workspaceRead(args.state.head, "evidence/observations.json");
```

The callee must actually exist with those parameters. Use `read_value` for scope inspection and `read_file` for source or data files. Reuse computed values by variable reference. For a host-backed dictionary, pass the dictionary value to a compatible callee and read only the branches or leaves needed for the decision.

To change a function, edit the contents of its existing writable codebase source and use the normal import/reload path. The codebase file set is fixed during a run: do not create, delete, or move codebase files. Directory-reducer project folders have the broader file lifecycle when a task needs generated files or structural refactoring. For generated programs, validate and execute the new artifact before claiming it works.

## Combinators and completion

- Repeated work uses ordinary control flow, for example `await Promise.all(items.map(item => assess(item, criterion)))`, or an explicit loop when order matters.
- A natural-language loop keeps its state in ordinary scope variables and closes each source line with `mark_lines`. Continuation is represented by the persistent scope and line marks, with no separate model command.
- The interpreter finishes by writing the typed return and ending its tool episode. Chat prose alone is not a return value. `report_blocker` and `report_error` terminate without a completed return.

Completion marks denote executed or skipped program lines. Mark actual work only after success; a range is inclusive, not a pair of isolated line numbers. Avoid using line marks as proof that a semantic result is correct.

## Repository anchors

When a checkout is available, inspect `natlang/codebase.py`, `natlang/types.py`, `natlang/surface.py`, `natlang/runtime.py`, `ts-host/src/native/source-core.ts`, `ts-host/src/native/agent.ts`, and `ts-host/src/native/runtime.ts`. Executable examples include `examples/triage/`, `codebases/dependency_plan/`, and `codebases/order_saga/`. Design/spec context lives in `spec/SPEC.md`, `spec/CODEBASES.md`, and `TYPES.md`; report discrepancies against executable behavior.
