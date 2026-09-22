# Effects, streams, concurrency, and recovery

## Separate portable state from native lifetime

Keep typed natlang state portable: records, lists, strings, finite numbers, Booleans, null, and checked supported values. Keep file handles, media bytes, processes, database clients, model instances, and DOM elements in the host environment. Expose direct operations in crisp code or portable IDs with lookup helpers as appropriate. Do not create a universal host protocol unless concrete portability requirements justify one.

`mode:'fresh'` resets eval globals; `mode:'retained'` preserves them for the environment's lifetime. Both can expose the same native host object by identity. `args`, `locals`, and `self` are portable snapshots rather than writable aliases to interpreter state. Returned values cross validation even if the native operation already mutated something.

Declared `fx` calls use a registered capability and a function's `effects` declaration; they record request/outcome in an effect journal. Direct `host` calls are permitted by the shared evaluator's authority and do not automatically become declared `fx` entries. Hosts can expose observation events for tracing, but observations are not rollback machinery. State the actual authority of each embedding.

## Durable effects and cancellation

For an operation that could be retried after failure:

1. Persist a stable logical operation ID and the exact request before execution when durability is required.
2. Record the actual observation/result with that identity.
3. If interrupted without a result, keep an unknown state and reconcile with the host before replay.
4. Reuse an ID only for the identical logical operation. Use the external system's idempotency/query/cancellation mechanisms where available.

A timeout can mean the operation is still executing. A rejected return can follow a successful native mutation. Neither is safe evidence for unconditional retry. Use read-only operations, temporary destinations, explicit user authorization, or reversible effects appropriate to the task when validating an integration.

## Streams and parallelism

Python `load_fold(step_file, init, source)` expects `acc` and `item` parameters with the accumulator type matching the return. The source can be an iterable or a polling stream source. A waiting source is distinct from closed or failed; preserve its accumulator and position.

Native hosts support a root Fold with `streams:{over: asyncIterable}`. Waiting for the next event need not consume a model turn. Browser `run()` returns when that stream completes; for a view after every event use `BrowserNatlangApplication`'s per-event reduction queue.

Map can exploit independent work subject to target host constraints. Node's public options include `mapWorkers` and `parallelMapSafe`; shared native objects/retained environments constrain safety. Do not assert parallel safety merely to obtain speed. Effectful/stateful ordering remains part of the program contract. A single threaded target is a valid embedding.

Multiple calls in one model response are an ordered non-atomic batch, not proof of parallel execution. A later action needing an earlier observation belongs after that observation. Native asynchronous jobs can be launched/inspected through the host while natlang decides when and how to proceed.

## Continuation versus durable restart

Both main agents support conversation rollover through `segment_turns` and `segment_messages`. Defaults at the inspected revision are six work turns and twelve messages; either can trigger a checkpoint. Setting one to null does not disable the other. Python uses `None`. A short note plus serialized typed scope, pending calls, and line marks continues the same task. Model-run budgets default to unbounded, while the checkpoint note itself has a separate allowance. Inspect truncation/finish reason if a note loses necessary information.

In-memory continuation is not process restart recovery. Persist the actual source identity, state/pending nodes when supported, effect observations, invocation seed policy, and ownership metadata. Reconstruct native resources through an application recovery contract. `dump_state` / `load_program` in Python and native state serialization support runtime state; they do not snapshot arbitrary host objects or provider sessions.

This includes host-backed `Dict<T>` inputs. Their observed leaves can inform
portable results and traces, but the provider itself remains native. Record the
provider root or revision needed by the application, revalidate that identity,
and bind a new provider on restart. Do not silently substitute a changed
working directory when deterministic resumption matters.

## Trace and reproducibility

Retain a run ID, source revision, parent/call identity, actual model/template/settings, seed policy, engine mode/authority, observations, outcomes, and continuation boundaries. Node can write `tracePath`; browsers return traces in memory for explicit persistence/export. Python has `TraceRecorder`/`TraceReader`; shared readers can inspect portable reduction events.

Distinguish inspecting a trace, reconstructing a portable value, resuming a pending node, and replaying external effects. Only promise what is implemented for that operation/engine. Time travel UI can inspect historical state without executing old effects.

Derived model seeds are tied to invocation identity. World randomness belongs to a separate seeded world RNG. For semantic merge replicas also agree on source, model, template, sampling, inputs, ordering, and relevant environment. Shared seed/model is part of the reproducibility contract; measure backend agreement rather than promising mathematical convergence.
