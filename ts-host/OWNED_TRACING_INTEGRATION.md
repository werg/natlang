# Opt-in owned JavaScript tracing evaluator wiring (Node)

This document describes the bounded native reference adapter. The asynchronous
selected-model integration, its supported native resources, module lifecycle,
event ingress and recovery contract are documented in
[SELECTED_SEMANTIC_INTEGRATION.md](SELECTED_SEMANTIC_INTEGRATION.md).

Natlang's Node entry point now exports `installOwnedTracingEnvironment(factory)`.
The installer replaces **both** evaluator seams:

- The fresh per-task factory used by `NatlangTask.environment()`.
- The retained callable-folder module realm installed through `setModuleRealm`.

There is **no implicit fallback** to natlang's `TypeScriptEnvironment` when an
owned factory refuses guest syntax, imports, host callbacks or resources.
The factory must return an `EvalEnvironment` whose `authority` begins with
`owned-` and whose `mode` matches the requested `fresh` or `retained`
mode. A native `shared-node-host` evaluator is explicitly rejected.

## Wiring both evaluation seams

An embedding application that has installed or linked the separately built
`tracing-js-self` package can opt in before creating natlang tasks or loading
callable-folder modules:

```ts
import {installOwnedTracingEnvironment} from '@natlang/node';
import {EvalFailure} from '@natlang/node';
import {createBoundedNatlangEnvironment} from 'tracing-js-self/natlang';

installOwnedTracingEnvironment(mode =>
  createBoundedNatlangEnvironment({
    EvalFailure,
    mode,
    onTrace(bundle,request,lowering) {
      // Persist only separately validated, exact-replayed guest archives.
      archiveTrace(bundle,request.path,lowering);
    }
  })
);
```

`installOwnedTracingEnvironment` is **process-wide** for new tasks and new
module-realm instances. It is not a per-task toggle, and already-instantiated
modules are not migrated. The application supplies the implementation, making
the tracer/host connection explicit rather than depending on a hidden JS
runtime fallback. Pass the actual natlang `EvalFailure` constructor to retain
`instanceof` compatibility.

## Current acceptance boundary

This is **integration wiring, not a production drop-in JS/TS replacement**.
The owned adapter currently supports a bounded strict JS/TS subset, data-only
live `__live` graph reconciliation, a data-only CommonJS-shaped retained
module boundary, internal guest promises, and explicitly versioned native
service effects. It cannot execute the existing callable-folder compiler's
general `require`, package imports, `nl`, `iterateOn`, callbacks, service
objects, class/metaprogramming syntax or arbitrary host resources.

Because callable-folder module evaluation supplies `exports`, `module`,
`require`, `__natlang`, `__natlang_context`, `nl` and `iterateOn`
by live reference, the bounded tracer **correctly refuses** many such calls.
Do not silently install the native TypeScript evaluator when it does.

A genuine natlang production cutover still requires independently verified
execution of actual compiler-generated programs, native callback reentry and
resource authority, retained module identity and caching, asynchronous guest
integration, source provenance and Test262/module conformance. The bounded
`tracing-js-self/evaluated-modules` API implements synchronous ESM cycles
and static live namespace property reads, but it is separate from natlang's
CommonJS-like callable-folder module realm, which still requires further
semantics.

The test `ts-host/test/owned-tracing-wiring.test.mjs` covers the opt-in
fresh-evaluator installation and rejects a native-host factory. It is a wiring
regression, not application compatibility evidence.
