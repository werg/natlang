# Compatibility-first hybrid JS tracing (opt-in, untraced native route)

`installHybridTracingEnvironment` in `@natlang/node` installs both per-task
and retained callable-folder evaluator factories. Unlike
`installOwnedTracingEnvironment`, it can route evaluation to natlang's
ordinary `TypeScriptEnvironment` even when the owned tracing interpreter
does not support the code. The default route is **native** so existing
callable-folder modules and host services have their usual compatibility.

**A native execution is not a semantic trace.** Native `execute` and
`executeAsync` results prepend an
`{operation:'execution.provenance',authority:'native-untraced/1', ...}`
event. For `evaluateModule`, which returns only the module completion
and has no event list, provide `onNativeExecution` to record the same
provenance before running the module. The provenance event itself is not
proof that nested calls are traced. Never feed these native executions to
megatoken as owned, replay-verified episodes.

```ts
import {installHybridTracingEnvironment, EvalFailure} from '@natlang/node';
import {createBoundedNatlangEnvironment} from 'tracing-js-self/natlang';

installHybridTracingEnvironment(
  mode => createBoundedNatlangEnvironment({
    mode, EvalFailure,
    onTrace(bundle, request, lowering) {
      // Only the explicitly owned route produces this callback.
      saveCandidateForIndependentAdmission(bundle, request, lowering);
    }
  }),
  {
    route({kind, code, path}) {
      // Compatibility for normal callable-folder modules.
      if (kind === 'evaluateModule') return 'native';
      // Choose the owned interpreter for known supported fixture paths.
      return path?.startsWith('owned://') ? 'owned' : 'native';
    },
    onNativeExecution(event) {
      recordUntracedExecution(event);
    }
  }
);
```

Routing is decided **before** either interpreter runs. There is no automatic
catch/retry, because an unsuccessful invocation might already have mutated
live bindings, performed I/O, or scheduled external work. Routing can be
changed for new evaluations by the supplied selector but is not a proof
of compatibility or tracing coverage.

This mode intentionally weakens runtime isolation. The native route executes
guest JS/TS in natlang's existing Node VM host boundary, with its normal
module, callback, network and resource behavior. It is not a security
sandbox; do not run untrusted programs, and do not assume that VM timeouts
roll back side effects. It does **not** make the tracer a complete JS engine,
supply independently admitted training data, or produce owned instructions
for code that runs on the native route.

Use `installOwnedTracingEnvironment` when fail-closed training capture is
required. Keep native and owned execution counts separate in dashboards,
corpora and model evaluation.
