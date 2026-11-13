# Tuned weights

Extension version **0.5-draft**, extending the [core specification](../SPEC.md) (0.5-draft). An extension is optional: a program or host that does not use it is unaffected.

**Scope.** How a program's tuned weight release binds to its calls. Producing a release is `natlang tune`
([docs/TUNING.md](../../docs/TUNING.md)).

A program may carry a tuned weight release
([docs/TUNING.md](../../docs/TUNING.md)). Its binding manifest declares an adapter
scope tree over the program's bound definitions, not over the context DAG: a
program root, folder nodes along source directories, and one lambda node per
named definition or persistent inline site. A call runs the stack of the
callee's own binding (gated families, then its scope ancestry from the root to
its lambda); it replaces the caller's stack and is never inherited from the
caller. Aliases of one bound definition share its binding; a rebound function
(`foo.in(context)`) has no tuned binding unless the release declares one for that
context, and then runs the base. Explicit `withNativeAdapters` scopes append
after the lexical stack in nesting order. A task pins one release for all its
calls. A stack the backend cannot serve falls back to the base executor and is
recorded as degraded, never as tuned execution; strict resolution throws
`weights-unavailable`, `weights-incompatible` or `weights-stale` instead.
