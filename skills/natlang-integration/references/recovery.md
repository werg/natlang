# Effects, concurrency, and recovery

## Values, live handles, and native lifetime

Portable values (records, arrays, strings, finite numbers, booleans, null) are copied into an invocation and checked at its boundary. Everything else — functions, class instances, DOM nodes, database clients, processes, folders — is passed by reference as a live handle and stays owned by the application. The model sees a preview and can call methods; the handle is released with the task. Do not invent a universal serialization protocol for native objects.

## What is and is not rolled back

| Change | When it takes effect | On failure |
|---|---|---|
| Captured `let` reassigned in eval | After the eval succeeds, if nobody else changed it | Discarded; a concurrent change is a `capture-conflict` and the eval retries |
| Property write on a live object | Immediately | Stays; traced |
| Service method call | Immediately | Stays; traced as an effect |
| Directory reducer file edits | On `commit`, installed by `folder.apply` | Discarded with the transaction |

For an operation that must not repeat:

1. Persist a stable operation ID and the exact request before executing it.
2. Record the observed result with that ID.
3. If interrupted without a result, keep the outcome unknown and reconcile with the external system before any retry.

A timeout can mean the operation is still running; a rejected return can follow a successful mutation. Neither justifies an unconditional retry. The workflow application (`applications/workflow/`) shows intents written before effects, idempotent receipts, and reconciliation after a lost acknowledgement.

## Concurrency

Independent tasks run concurrently, and sibling natlang calls in one task (`Promise.all`) run in parallel. Only folder writers serialize. A model backend may queue requests; that is resource scheduling, not a semantic limit. Order effectful work explicitly where it matters; several tool calls in one model response are an ordered, non-atomic batch.

## Cancellation and limits

`runtime.run(fn, { signal })` and `EventLoop.cancel()` abort the task; natlang calls in flight reject. Native work already started keeps running unless the application cancels it. `limits` (`maxDepth`, `maxActions`, `maxToolCalls`, `maxEpisodes`) and model options are deployment policy.

## Continuation and restart

Long invocations roll the conversation over (`segmentTurns`, `segmentMessages`) while the eval scope and line marks carry progress. That is not process restart recovery: persist application state, event IDs, operation receipts, and seeds, and rebuild native resources through the application's own recovery path.

## Traces and reproducibility

Each invocation produces a trace (`trace` sink on the runtime or task): manifest, presented messages and tools, actions, observations, effects, state, outcome, and the call tree. Inspecting a trace, reconstructing its final value, and replaying effects are different operations; only the first two are implemented. Model seeds derive from invocation identity; keep world randomness in a separate seeded generator. Reproducibility needs the same source, model, template, sampling, inputs, ordering, and relevant environment, and still differs across inference backends.
