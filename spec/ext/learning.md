# Learning: grad, objectives and adapters

Extension version **0.5-draft**, extending the [core specification](../SPEC.md) (0.5-draft). An extension is optional: a program or host that does not use it is unaffected.

**Scope.** `natlang:learning` (gradients over soft values, objectives, optimisers, `save`) and `Adapter` values with `withAdapters`. Learning produces new values and never mutates model weights. Depends on [Neuralese](neuralese.md).

**Learning.** `natlang:learning` exports `grad`, `valueAndGrad`, `stopGradient`,
objectives (`crossEntropy`, `selfDistill`, `conditionedDistill`, `logLikelihood`, `law`, `klPrior`, `decision`), optimisers,
`withAdapters`, `adapters.create`, and `save`, to callers given the `natlang:learning` service. `grad(f, a)`
differentiates a loss with respect to soft arguments by recording `f(a)` and
replaying it ([NEURALESE_GRAPH.md](../NEURALESE_GRAPH.md)); discrete choices are
held fixed and trained through `logLikelihood`. Nested `grad` is first-order
unless `{ order: 2 }` is given. Training steps are ordinary step functions run
with `iterateOn`; learning produces new values, never mutates model weights, and
is promoted by binding a context that contains them.

**Adapters.** An `Adapter` is a soft value holding the coefficients of a tiny
weight adapter of the serving model. Its block's dialect states its structure
(kind, rank, layers, targets, base-weight hash; `model/tiny_adapters.py`), and it
runs only on that base. `withAdapters(adapters, fn)` makes adapters active for
every model turn inside `fn`: generation and decision readouts run the adapted
model, and recorded turns replay with the same adapters, so `valueAndGrad(f,
adapter)` trains an adapter like any soft value. A zero adapter is the base model.
An adapter is a value, not a change to the model: binding it is how it is
promoted, and it ships with the program that uses it
([LEARNING_CONTINUUM.md](../../plans/neuralese/LEARNING_CONTINUUM.md) §6).
