# Law-based compiler rewrites

Version 1, 2026-10-03. Normative for the natlang compiler (`ts-host/src/compiler/`).
Part of the Neuralese section of [SPEC.md](SPEC.md).

The Neuralese combinators approximately satisfy algebraic laws. The compiler uses
them, in the spirit of GHC rewrite rules, to make programs cheaper.

## Laws

"≈" is measured through `read` and downstream consumers, never as vector equality.

| Law | Form |
| --- | --- |
| Map identity | `read(map(v, x => x)) ≈ read(v)` |
| Map fusion | `map(map(v, g), f) ≈ map(v, x => f(g(x)))` |
| Read/map commutation | `read(map(v, f)) ≈ f(read(v))` |
| Combine associativity | `combine(combine(a, b), c) ≈ combine(a, combine(b, c))` |
| Combine identity | `combine(v, empty()) ≈ v` |
| Split/zip | `split(zip(a, b)) ≈ [a, b]` |

## Rules

| Rule | Rewrite | Saves |
| --- | --- | --- |
| `map-fusion` | `map(map(v, g), f)` → `map(v, compose(f, g))` | one model call and one stored block |
| `read-map` | `read(map(v, f))` → `f(read(v))`, or the reverse | the cheaper of an exact and a soft call |
| `map-identity` | `map(v, x => x)` → `v` | one model call |
| `combine-reassociate` | a chain of `combine` → a balanced tree | latency, through parallel calls |
| `combine-identity` | `combine(v, empty())` → `v` | one model call |
| `split-zip` | `split(zip(a, b))` → `{ 0: a, 1: b }` | two model calls |

A rule matches only syntactic occurrences within one function body whose
intermediate values have no other use. It never moves a call across an effect,
an `await` of another call with effects, or a capture write-back.

## Composition

`compose(f, g)` is a system natural-language function with a trained soft body. For
text functions it produces the composed function's instructions; for soft functions
it is the same operator applied to soft bodies. Its result is a function-typed value
in the caller's context, so it adds no executable node.

## Enabling

- A rule is enabled automatically, per model and dialect version, once a
  whole-program comparison of rewritten against unrewritten executions on held-out
  programs shows it does no harm. The comparison is re-run when the model or dialect
  version changes; until it has run, the rule is off.
- The enabled set is part of the runtime configuration and is recorded in each
  trace's manifest.
- Rewrites can be disabled per program or per call site (`// natlang-no-rewrite`),
  for evaluation and debugging.

## Tracing

Every applied rewrite is recorded as a `rewrite` node in the execution graph
([NEURALESE_GRAPH.md](NEURALESE_GRAPH.md)) with the rule, the source site, and the
comparison that enabled it, so replay and evaluation can compare rewritten and
unrewritten executions.
