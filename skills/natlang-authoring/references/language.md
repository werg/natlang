# Language and source contracts

This guide describes the implementation inspected in September 2026. Resolve version differences using the actual loader, runtime, and tests. Some older specification passages still describe removed local/function limits and older tool schemas.

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
database access should use its real host API. Do not force the runtime to
materialize a native provider merely to pass it into eval.

Partial return records can be built incrementally; completion requires a complete result of the declared type. Wrongly typed fields are rejected. A validated record can still be semantically false. Exact count checks, semantic rubrics, and external effect receipts address different claims.

## Calls, values, and reuse

In source, write `assessment = assess(observation, criterion)` and make argument intent unambiguous. The interpreter can implement this with:

```text
call(function="assess", to="let/assessment",
     inputs={"observation":"args/observations/0", "criterion":"args/criterion"})
```

Mixed literal/path binding:

```text
call(function="workspace_read", to="let/evidence",
     inputs={"head":"args/state/head"},
     values={"path":"evidence/observations.json"})
```

The callee must actually exist with those parameters. `read(path="args/state")` reads interpreter state; it does not retrieve an application artifact. To reuse a computed value, use `write` with `source` or bind a path to a call rather than regenerating the value as tokens. Use full/ranged reads when the state listing says a value is only a preview. For a host-backed dictionary, pass the dictionary path directly to a compatible callee and read only the branches or leaves needed for the decision. Directory listings discover names without loading every leaf.

To vary a function, copy `Function<helper>` into a local, edit that copy's instructions, and call `let/copy`. The checked original remains immutable. For larger generated programs, use the application's versioned source loader and child execution API; validate and execute the new artifact before claiming it works.

## Combinators and completion

- Map: `call(function="assess", to="let/results", over="args/observations", inputs={"criterion":"args/criterion"})`. Leave the per-item parameter unbound. Output order follows input order.
- Fold: with `over` and `init`, leave `item` and `acc` unbound; the helper returns the next accumulator.
- Iterate: bind fixed parameters and leave one state parameter for `init`; `until` names a checked Boolean predicate. The explicit `max` is the Iterate contract, not an implicit cap on every program. Choose it from the algorithm's requirements. Long loops can also keep explicit state in a natlang function.
- Calling the same unfinished function destination with only `function` and `to` requests resumption. Distinguish this from making a new call or repeating an effect.
- The interpreter finishes by writing the typed return and ending its tool episode. Chat prose alone is not a return value. `report_blocker` and `report_error` terminate without a completed return.

Completion marks denote executed or skipped program lines. Mark actual work only after success; a range is inclusive, not a pair of isolated line numbers. Avoid using line marks as proof that a semantic result is correct.

## Repository anchors

When a checkout is available, inspect `natlang/codebase.py`, `natlang/types.py`, `natlang/surface.py`, `natlang/runtime.py`, `ts-host/src/native/source-core.ts`, `ts-host/src/native/agent.ts`, and `ts-host/src/native/runtime.ts`. Executable examples include `examples/triage/`, `codebases/dependency_plan/`, and `codebases/order_saga/`. Design/spec context lives in `spec/SPEC.md`, `spec/CODEBASES.md`, and `TYPES.md`; report discrepancies against executable behavior.
