# Native learner and service decomposition (2026-10-15)

## Policy

The native learner operates on one declared native invocation graph and one exact native checkpoint. It validates that every graph row, imported state, action coordinate, and parameter belongs to the selected model identity; it evaluates the declared objective in one shared graph forward; it returns gradients or commits an optimizer update only at the requested order and update boundary. An update continues the same native model state and records the changed checkpoint identity. Decisions score candidate continuations from one cached native prompt; batches preserve per-item results. Guidance accepts a token only when the configured natural-language constraint accepts its actual decoded prefix, and rejected candidates restore the exact native cache and sampling record before resampling. Unsupported legacy projected operations remain explicit errors until an equivalent native operation exists.

## Functions and exact mechanisms

1. **Native request validation** — one shared typed service function verifies schema, source split, exact checkpoint/tokenizer/embedding/architecture identity, action-to-query coordinates, external native-state references, and objective binding. Crisp validation is justified at the trust boundary.
2. **Native graph objective** — reuse `data.native_graph.parse_graph`, `model.shared_state` graph execution, and the registered `NativePolicyObjective`; never flatten multi-tensor state into a legacy block or replay separate producer forwards. The objective remains a declared policy input; Python only dispatches registered objective kinds.
3. **Differentiation** — a shared native learner function computes first derivatives from the complete graph objective. Higher-order derivatives are advertised only when the same model-family numerical tests establish them. Exact autograd, graph retention, and finite checks are crisp tensor mechanics.
4. **Optimizer and continuation** — one optimizer function updates named native trainable parameter tensors with explicit checkpoint identity and resumable state; commits are atomic and produce a new immutable checkpoint receipt. Reuse shared optimizer factories and parameter naming, not a separate optimizer implementation; explicit SGD/Adam service requests use the shared standard optimizer factory and production MuonSR uses the native trainer factory. Updating shared backbone weights requires the same runtime memory policy and live model handoff as training.
5. **Adapter values** — adapters are values only when they identify an actual native trainable component and can be applied by the normal native inference path. They preserve architecture/checkpoint identity and tensor inventory. Legacy projected adapters are never reinterpreted as native parameters.
6. **Decision scoring** — one shared inference function prefills each distinct typed prompt once, splices exact native imports, and scores each candidate continuation from its actual predecessor logits. Batching changes scheduling only, not item semantics.
7. **Guided generation** — the natural-language guidance policy judges decoded prefixes; crisp code snapshots/restores the complete native cache and records only actually sampled tokens and applied masks. Existing sampling capture schema remains the source of truth.
8. **Capabilities and errors** — `/info` lists only implemented native operations and their exact order/adapter modes. Structured errors name the missing operation; capability denial is honest status, not a completion claim.
9. **Natlang surface and measurement debt** — TS learning functions and native service routes preserve typed native graph/parameter identities end to end. New signatures, errors, and instruction text update both authoring/integration skills in the same push and are recorded in `plans/MODEL_FACING_CHANGES.md` for live measurement before adoption.

## Qualification

Exercise existing `neuralese-learning.test.mjs` contracts against native LFM and Mellum services, including first/second derivatives where advertised, optimizer update plus forward change and resume, adapter application/gradient, single-vs-batched candidate scoring, and guided cache rollback with exact `/2` capture. Compare graph outputs and gradients against the explicit cached native reference; do not loosen tolerances around an unexplained discrepancy. Base-relative KL is telemetry only. The shared full-check service and independent review own integrated qualification; focused DGX probes run through the memory ledger.

## Live implementation status

Native cached `/decide` and `/score` now use the shared renderer and the native
Engine cache/step path. Named native tensor artifacts use a distinct content
identity and the existing durable owner/pin/collection store. A composite
checkpoint binds an immutable full native parent, named native parameter
patch, and optimizer tensor state without rewriting the parent. Adapter application/creation, second-order execution, and guidance
remain open until their actual client paths and both-backbone validation land;
capability denial is not counted as implementation.

Native `/grad` accepts `natlang.native-gradient-request/1` with a native graph
row, explicit `model-parameters` or `native-state` target, and either shared
graph CE or the registered native policy objective bound by exact spec-file
SHA and actor identity. It rejects held rows, checks imported native state,
executes the shared packed-graph objective, and returns named parameter or
consumed KV/hybrid gradients in a stored `nnt1_` artifact. Policy gradients use
the same `NativePolicyObjective` as the trainer; a focused test scores and
differentiates imported attention KV and finite-convolution state. It exposes
first-order gradients only. The `/info` surface reports `grad_order: 1` for
native models.

TypeScript memory, persistent-file and HTTP stores now preserve the same
`nnt1_` safetensors format and Python content identity. This is a named tensor
archive domain; `get()` never turns it into a flat Neuralese value. The
composite writer requires the parent model to have been restored from the exact
manifest parent and rejects undeclared parameter-version changes. Native `/optim`
applies SGD or Adam to explicitly named backbone or interface parameters, stores
named per-parameter optimizer state, and publishes a new immutable effective
checkpoint. It updates a detached copy of selected native parameters, sharing
untouched storage, so concurrent generation and failed requests never observe
transient updated weights. The caller reloads the returned checkpoint before
continuing, so one identity never names two weight states. Composite manifests
use one stable update directory and restore iteratively from oldest to newest;
continuing updates do not grow directory depth or Python recursion depth. An HF
base-directory override changes only the location of source model files.

The TS host exports explicit native graph-gradient and optimizer clients; they
archive named gradients, parameter updates and optimizer state, and restore the
state artifact before continuation. Identity equality compares exact fields
independently of JSON property order. This does not reinterpret legacy flat
value gradients as native graph requests. Adapters/direct-gradient/higher-order/
guidance still need native implementations. Capability denial remains explicit;
requesting unsupported order/derived computations or adapter application is
rejected before any native graph computation rather than silently ignored.
Actual LFM and Mellum tests cover gradients, immutable full-backbone/interface
updates, reload and alternating-target SGD/Adam continuation. Focused evidence
is not cutover or task-quality qualification.

## Full named updates and exact continuation

The existing composite checkpoint now also supports explicit full-model named
parameter patches. The optimizer receives exact native graph gradients; it uses
the shared train/optim standard SGD/Adam factory and shared gradient/commit
helpers. Production MuonSR continues through the native trainer, not an SGD alias.
A lightweight copy of the module tree clones only selected parameters, including
tied aliases. Untouched tensors remain shared with an unchanged verified parent.
Publication verifies every untouched parameter identity and the parent's loaded
checkpoint/version binding, then writes a named patch without mutating the live
actor. Reload uses the normal shared native checkpoint loader for both models.
Per-parameter Adam steps/moments and SGD momentum persist even when a subsequent
request targets different parameters. RNG and continuation metadata use the
existing full training-state capture and durable immutable sidecar. Changed
weights invalidate prior semantic adapters; clients must refresh the adapter
against the resulting checkpoint before semantic promotion.
