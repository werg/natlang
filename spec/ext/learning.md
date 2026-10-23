# Learning: grad, objectives and native state

Extension version **0.5-draft**, extending the [core specification](../SPEC.md) (0.5-draft). An extension is optional: a program or host that does not use it is unaffected.

**Scope.** `natlang:learning` (gradients over soft values, objectives, optimisers, and `save`) and stored positional state with `withNativeStates`. Learning produces new immutable state values and never mutates model weights. Depends on [Neuralese](neuralese.md).

**Learning.** `natlang:learning` exports `grad`, `valueAndGrad`, `stopGradient`,
objectives (`crossEntropy`, `selfDistill`, `conditionedDistill`, `logLikelihood`, `law`, `klPrior`, `decision`), optimisers,
`withNativeStates`, and `nativeStates.gradient` / `nativeStates.optimize`, to callers given the `natlang:learning` service. `grad(f, a)`
differentiates a loss with respect to soft arguments by recording `f(a)` and
replaying it ([NEURALESE_GRAPH.md](../NEURALESE_GRAPH.md)); discrete choices are
held fixed and trained through `logLikelihood`. Nested `grad` is first-order
unless `{ order: 2 }` is given. Training steps are ordinary step functions run
with `iterateOn`; learning produces new values, never mutates model weights, and
is promoted by binding a context that contains them.

**Native state.** A native state is an immutable positional KV and hybrid-state
span produced by a model. `withNativeStates(ids, fn)` inserts native-state
references into a system message for every model call in `fn`. It never inserts
the archived source text as ordinary prompt text. The server consumes the
actual state at its causal position and records the imported state in the call
capture. The runtime requires the state writer's
exact checkpoint, tokenizer, embedding and architecture identity to match the
current reader. To learn a state, request `native-state` gradients for the
captured graph with `nativeStates.gradient(graph, stateId, identity)` and pass
the returned gradient to `nativeStates.optimize`; that returns a new state
artifact with verifiable parent, gradient and optimizer ancestry.

This updates the prompt's state tensors for a call. It does not modify the
model's weights or reproduce the old behavior of applying a low-rank weight
adapter to every layer. The old weight-adapter and projected adapter-code
interfaces are retired from the current runtime.
