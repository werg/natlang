# Native observation tracing for unsupported JS/TS

Use `installObservedNativeTracingEnvironment(onTrace)` from `@natlang/node`
to run ordinary Node/TypeScript natlang evaluations and receive an execution
record even when the owned interpreter does not support their language
constructs.

```ts
import {installObservedNativeTracingEnvironment} from '@natlang/node';

installObservedNativeTracingEnvironment(record => {
  // Persist start, completed and failed records to a separate observation store.
  writeObservation(record);
});
```

Each record is `natlang-native-observation/1` and contains the phase
(`start`, `completed`, `failed`), evaluation kind, mode, code, path and an
optional error message. Synchronous and asynchronous evaluations also return
a `native-observed/1` provenance event on success. Retained callable-folder module **bootstrap** is recorded via the callback.
Top-level own-data callable exports are also observed when called. Each
`exportCall` record names the export and an invocation ID, with `start`,
`completed` (synchronous return, including a returned Promise), or `failed`
(synchronous throw). When the returned value is a native Promise, a later
`settled` record reports `fulfilled` or `rejected` with the same invocation ID;
the original Promise identity is preserved. These wrappers preserve `this` forwarding and constructor
invocation, but intentionally change exported function identity.

**Limit:** This is an *execution-boundary trace*, not a semantic instruction
trace. It does not record individual expressions, reads, writes, dynamic
calls, callback bodies, nested calls, asynchronous scheduling,
host effects, snapshots or replayable state transitions. In particular,
calling a top-level exported function and native Promise fulfillment/rejection are observed,
but its internal instructions, nested callbacks and subsequent export reassignment are **not** traced.
Therefore this mode does not meet the requirement that every executed JS/TS
operation be traced, and observations must not be passed to megatoken as
owned/replay-verified training episodes.

Trace records contain source text and may include sensitive information;
protect their storage. Observer failures propagate and can prevent an
execution from starting or turn a successful native evaluation into a
failed API call; native side effects already performed are not rolled back.

## Path to semantic coverage

1. Instrument the compiler's JS/TS AST into explicit runtime hooks for
   expression evaluations, reads/writes, calls, returns and throws.
2. Propagate trace identity across promises, timers, workers and callbacks;
   trace callable exports on each invocation, not just module bootstrap.
3. Journal host effects with stable operation identifiers and versioned
   payload schemas.
4. Verify traces against observable outcomes, add differential fixtures,
   and keep native observational traces separate from owned replay archives.
5. Extend the owned interpreter to absorb supported native-observed cases,
   with independent exact replay and corpus admission checks.

The distinct `installHybridTracingEnvironment` currently refuses native
execution to satisfy its trace-required policy. Choose observation mode only
when boundary-level coverage is preferable to refusal.
