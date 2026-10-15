# Native learner and service decomposition (2026-10-15)

## Policy

The native learner operates on one declared native invocation graph and one exact native checkpoint. It validates that every graph row, imported state, action coordinate, and parameter belongs to the selected model identity; it evaluates the declared objective in one shared graph forward; it returns gradients or commits an optimizer update only at the requested order and update boundary. An update continues the same native model state and records the changed checkpoint identity. Decisions score candidate continuations from one cached native prompt; batches preserve per-item results. Guidance accepts a token only when the configured natural-language constraint accepts its actual decoded prefix, and rejected candidates restore the exact native cache and sampling record before resampling. Unsupported legacy projected operations remain explicit errors until an equivalent native operation exists.

## Functions and exact mechanisms

1. **Native request validation** — one shared typed service function verifies schema, source split, exact checkpoint/tokenizer/embedding/architecture identity, action-to-query coordinates, external native-state references, and objective binding. Crisp validation is justified at the trust boundary.
2. **Native graph objective** — reuse `data.native_graph.parse_graph`, `model.shared_state` graph execution, and the registered `NativePolicyObjective`; never flatten multi-tensor state into a legacy block or replay separate producer forwards. The objective remains a declared policy input; Python only dispatches registered objective kinds.
3. **Differentiation** — a shared native learner function computes first derivatives from the complete graph objective. Higher-order derivatives are advertised only when the same model-family numerical tests establish them. Exact autograd, graph retention, and finite checks are crisp tensor mechanics.
4. **Optimizer and continuation** — one optimizer function updates named native trainable parameter tensors with explicit checkpoint identity and resumable state; commits are atomic and produce a new immutable checkpoint receipt. Reuse the native trainer optimizer factory and parameter naming, not a separate optimizer implementation. Updating shared backbone weights requires the same runtime memory policy and live model handoff as training.
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
checkpoint binds an immutable full native parent, named interface parameter
patch, and optimizer tensor state without rewriting the parent. Native `/grad`,
`/optim`, adapter application/creation, second-order execution, and guidance
remain open until their actual client paths and both-backbone validation land;
capability denial is not counted as implementation.
