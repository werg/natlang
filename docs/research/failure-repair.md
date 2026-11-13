# Failure-aware scope eval and repair

Natural-language functions can recover from a failed `eval` in the same model episode. The runtime retains a read-only diagnostic snapshot and gives the model a bounded opportunity to inspect it and submit a corrected eval. This works in Node and the browser, including when ordinary validation feedback is returned to the caller.

## Failure path

1. Before each scope eval, capture the committed input and local values. Compilation failures retain TypeScript diagnostics; execution failures also retain captured console output and the original JavaScript stack. Boundary failures retain their rejection diagnostics.
2. Record a `scope_failure` trace event. The snapshot contains the attempted code and a compact recent execution trace, including eval, action, invocation, host, and effect events. It is exposed on the next eval as immutable `debug` (or `__natlangDebug` if that name is already in the function scope).
3. The agent's system prompt and failed tool result identify the binding and tell the model to inspect the snapshot, account for any host effects, and repair the code. A successful diagnostic probe leaves the snapshot available. Setting a compatible function result clears it.
4. The default allowance is three failed repair attempts per unresolved failure episode. `maxFailureRepairs` can change that allowance; a diagnostic probe does not reset it. The episode's existing turn, token, and wall-clock budgets still apply.

The snapshot has `kind` (`compile`, `runtime`, or `boundary`), `message`, `code`, `scope.inputs`, `scope.locals`, `diagnostics`, `logs`, `stack` when available, and `trace`. The snapshot is a view of the committed state *before* the failed eval. Failed evals do not commit portable locals or the function result. Host effects are **not rolled back**; the model must inspect the trace before retrying an effectful call.

Example recovery sequence:

```ts
// This fails when items[9] is absent.
result = String(items[9].value)

// In the next eval, inspect the snapshot without setting the function result.
console.log(debug.message, debug.scope.inputs.items.length, debug.stack)

// Then submit corrected code.
result = String(items[0].value)
```

## Boundaries and follow-up

The snapshot is intentionally bounded: source code, stack frames, logs, and recent trace events are truncated. It is a repair aid, not a deterministic replay or transaction rollback. The trace identifies prior effects but cannot prove an external effect was undone. A future source-map-aware diagnostic pass could annotate the failing source expression and callee path more precisely; the current implementation preserves original stacks and compiler diagnostics without adding another model call on every failure.

Tests cover runtime and compilation failures, immutable diagnostic probing, scope-name collisions, browser behavior, caller-feedback repair, and retry exhaustion. A live teacher inference check is optional and should be run only while the Bonsai server is available; deterministic tests do not require the teacher.
