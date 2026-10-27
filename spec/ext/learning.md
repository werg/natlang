# Learning: grad, objectives and native state

Extension version **0.5-draft**, extending the [core specification](../SPEC.md) (0.5-draft). An extension is optional: a program or host that does not use it is unaffected.

**Scope.** `natlang:learning` (gradients over native state and native adapter coefficients, objectives, optimisers, and `save`) and stored positional state with `withNativeStates`. Learning produces new immutable artifacts and never mutates a loaded model's weights in place. Depends on [Neuralese](neuralese.md).

**Learning.** `natlang:learning` exports `grad`, `valueAndGrad`, `stopGradient`,
`trajectory`, objectives (`crossEntropy`, `selfDistill`, `conditionedDistill`,
`decision`, `expectedReward`, `policyGradient`, `logLikelihood`, `law`, `sum`,
`scale`), optimisers (`sgd`, `adam`), `withNativeStates`, `withNativeAdapters`,
`withSystemPrompts`, `nativeStates.gradient` / `nativeStates.optimize`,
`nativeGraph.gradient`, `nativeAdapters.create`, `save` and `load`, to callers
given the `natlang:learning` service. `grad(f, a)` differentiates a loss with
respect to the native roots in `a` by recording `f(a)` and replaying it
([NEURALESE_GRAPH.md](../NEURALESE_GRAPH.md)). A root is an archived native
state (`nzs1_`) or a native adapter coefficient artifact (`nnt1_`); all roots
of one gradient share one exact base checkpoint identity. Any other soft value,
including a retired projected `nz1_` value, is rejected with
`learning-native-target`. Each gradient entry names a `state-gradient` or
`adapter-gradient` tensor artifact. Discrete choices are held fixed and trained
through `logLikelihood` or `policyGradient`. Nested `grad` is first-order
unless `{ order: 2 }` is given; order 2 is the deepest supported. Training steps
are ordinary step functions run with `iterateOn`; learning produces new values
and is promoted by binding a context that contains them.

**Objectives.** Every objective scores exact recorded native model turns; the
server lowers the terms into one differentiable graph over those turns. The
term kinds are:

- `crossEntropy(output, expected)`: mean negative log-likelihood of the
  expected reply as the output's continuation.
- `selfDistill(output, withFullSource)` and `conditionedDistill(student,
  teacher, options?)`: distillation of the student turn toward a teacher turn
  that saw more context; the teacher must see privileged context the student
  did not (`learning-teacher-unconditioned`, `learning-privilege-leak`). On a
  decision readout, `conditionedDistill` distils the option distribution.
- `decision(output, expected, rule?)`: a proper scoring rule (`logLoss`,
  `brier`, `rps`) on a decision readout. `expectedReward(output, reward)` is
  the expected reward under the readout's option distribution.
- `logLikelihood(trajectory, weight?)`: summed log-probability of the recorded
  turns' sampled tokens, at temperature 1 (`model-conditional`).
  `policyGradient(samples, options?)` weights each sample's log-likelihood by
  its reward minus a baseline and uses the recorded sampling temperature and
  token masks (`sampled-policy`); it needs a positive-temperature sampled
  trajectory.
- `law(name, ...args)`: a combinator law as a cross-entropy consistency loss,
  measured through `read` (names `mapIdentity`, `mapFusion`,
  `readMapCommutation`, `combineAssociativity`, `combineIdentity`, `splitZip`).
- `sum` and `scale` combine terms.

There is no prior or density term over stored state: native KV has no Gaussian
prior, and the former `klPrior` objective and sampled-payload likelihood are
retired.

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
artifact with verifiable parent, gradient and optimizer ancestry. An optimizer
step on a state root inside `grad` does the same. This updates the state
tensors a call imports. It does not modify the model's weights.

**Native actor coefficients.** `nativeAdapters.create` creates an exact-checkpoint
coefficient artifact over declared native model parameters. `withNativeAdapters`
binds that actor to calls. `nativeGraph.gradient(f, target, identity)` returns
gradients through the recorded producer/consumer execution graph for a
`native-state`, `adapter` or `model-parameters` target, and an optimizer step
(`nativeAdapters.optimize` in the module, `nativeAdapterOptimizerStep` for
hosts) returns a new coefficient artifact with optimizer ancestry. This changes
the parameters a native call executes with; it is distinct from native-state
scope, which changes only the imported call context. Native graph policy
learning remains a separate operation: its exact-actor coefficient bindings
receive gradients through the recorded whole graph and update through the
native optimizer. Legacy projected adapter codes and generic tiny-adapter/LoRA
transport are retired; they are not substitutes for either native-state
learning or exact native actor learning.
