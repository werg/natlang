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
patch, and optimizer tensor state without rewriting the parent. Native `xs` and
`tiny` adapter application/creation and exact actor replay are integrated.
Independent root qualification on October16 passed40 actual CUDA/typed-HTTP,
optimizer-restart, heterogeneous-actor and counted-FP8 tests. The service now
advertises these adapter modes and same-base adapter-state consumption. This
evidence covers the implemented small-backbone paths, not full-size cloud
performance or task lift. Second-order execution remains open; capability
denial is not counted as implementation.

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
value gradients as native graph requests. Supported native adapter application,
coefficient gradients and immutable optimizer continuation use the shared
mechanisms. Guided generation is implemented below. Higher-order derived
computations still require implementation and both-backbone validation;
unsupported requests are rejected before native computation. The advertised
first-order boundary is a real remaining gap, not completion.
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

## Portable parameter/checkpoint application

1. S7 chooses the exact updated actor/participant to deploy; transport never
   chooses a replacement model or relabels an adapter.
2. A crisp exporter walks the composite's exact dependency chain and gathers
   each parent checkpoint, named parameter patch, optimizer artifact and training
   RNG sidecar. It verifies every declared SHA before streaming immutable copies
   into a portable content-addressed objects directory.
3. Keep original composite manifest bytes intact. Extend the existing shared
   ArtifactResolver with an explicit content root: a role pinned by SHA may use
   only that digest-named object, or its already verified original path. Never
   guess by filename suffix, shape, model family or a stale artifact identity.
4. The normal native checkpoint loader applies the same manifests from that
   bundle, using the separately pinned HF base files (an explicit location
   override may supply those identical files). Typed parameter/optimizer values
   retain their original content identities and stores can restore them normally.
5. Verify relocated load, next real gradient/update and per-request sampling
   against the original actor. Chat's explicit request generator is invocation
   state and does not advance inherited training RNG; deterministic SGD/Adam
   preserve that training boundary. Do not restore process-global RNG on actor
   startup or claim a model checkpoint resumes an unfinished chat invocation.

This supports portable native parameter checkpoint application. Legacy adapter
creation/direct application and higher-order graphs remain separate open
paths; the bundle does not advertise those capabilities. Native guidance is
implemented separately below.

The bundle may also include the exact semantic adapter receipt and every bound
participant tokenizer. The shared semantic loader uses the same role resolver
for its checkpoint, keeping adapter SHA, participant identities and tokenizers
unchanged after relocation. No semantic adapter is applied to changed weights
merely because its tensors fit. Streaming export and content addressing retain
one large full-checkpoint copy, and verification checks the complete declared
native plus semantic dependency inventory before deployment.

## Native guided generation and exact replay

1. Use the existing Guide constraints and retry policy over the actual decoded
   reply, including native values as the same `__nz` syntax placeholder.
2. Save one complete prompt-cache anchor (positional attention plus convolution/
   hybrid state) and the actual generated own-token/producing-role log. On a
   rejection, replay only the retained prefix with native_step, restoring every
   own position, producer capture, write reference and sample coordinate; rejected
   artifacts cannot remain in the emitted reply or sampling receipt. This is
   inference rollback, not recursive training graph replay.
3. Reuse one shared normalized masked log-distribution for sampling and native
   RL scoring. Capture sparse banned-token IDs in the existing /2 sampling
   receipt; forced prefix inputs remain forced rather than fabricated actions.
4. Buffer guided output until its final committed reply, because the streaming
   contract has no retraction event. Delayed guided deltas still use the existing
   text/native-state events, and rejected text/references never escape.
5. Verify both actual model ports, complete cached-prefix equivalence, producer
   rollback, stochastic behavior-logp replay and gradient parity before enabling
   the guidance capability. Adapter/direct and order2 support remain separate.
