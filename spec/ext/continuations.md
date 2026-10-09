# Continuations

Extension version **0.5-draft**, extending the [core specification](../SPEC.md) (0.5-draft). An extension is optional: a program or host that does not use it is unaffected.

**Scope.** Continuing a long invocation in a fresh model conversation while carrying scope, staged result, child state and folder overlay.

A long invocation may continue in a fresh model conversation. The runtime
carries the scope, staged result, child state, and folder overlay; earlier
conversation text is not copied. A short working note may carry unresolved
reasoning.
