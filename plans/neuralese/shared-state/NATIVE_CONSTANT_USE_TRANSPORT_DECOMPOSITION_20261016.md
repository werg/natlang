# Native constant uses: transport and gradient semantics

The natlang operation is the existing `stopGradient(value)`: retain the same
value and actual forward execution, while treating this particular use as a
constant during differentiation. It does not make every equal artifact constant.
No new model-facing operation or prompt is introduced.

1. The typed value retains the immutable artifact ID plus optional true-only
   constant-use metadata. Context snapshots, soft function bodies and `.nz`
   preserve that use metadata. Distinct constant/live contexts have distinct
   context identities; the underlying native-state artifact remains unchanged.
2. The existing private sentinel transports that metadata to structured native
   content parts. Compiler literals, contextual soft bodies and archive loading
   preserve it. It never becomes a token presented to the model.
3. The shared renderer produces the same text and tokens and a parallel use
   flag. Engine generation/decision/contextual initialization records the flag
   on the actual consumed state receipt and capture, retaining the original
   actor/source provenance. Invalid explicit false or nonboolean metadata fails
   verification instead of being silently reinterpreted.
4. Whole-episode verification binds each captured use to its actual block-read
   event. Replay places the flag on the existing `ImportSpan`; the shared graph
   attention and convolution gather mechanisms already implement the exact
   per-use constant boundary. All producer/consumer occurrences remain present.
5. Adapter coefficient uses reuse the same declared adapter mechanism. Only
   that use's coefficient tensors are detached; numerical effective actor
   identity is unchanged. The invocation cache and row grouping also bind
   derivative-use flags, preventing a constant computation from replacing a
   live one with identical weights. Gradient artifact provenance retains the
   exact invocation adapter declarations.

These are exact compiler/transport/AD mechanisms. Decisions about where to use
constants remain in the natural-language program. Public recorded higher-order
dispatch remains guarded; this component does not claim complete semantic-codec
RL or repeated occurrence lowering. Those are the next production blockers.
Legacy projected-value channels are not a replacement implementation.

Qualification: targeted CPU replay/coverage and actual both-port captures,
Node typed value/context/eval/archive roundtrips, and actual CUDA per-use source
and coefficient adjoints. Existing default artifact and row identities are not
changed. Root independently reviews before adoption. Frozen cloud source/cache
policy still needs its explicit continuation review before deployment.

Focused evidence (2026-10-16): CPU replay/coverage32 passed; final integrated
capture/typed HTTP/archive/whole-episode suite41 passed45.37s; Node typed
values/agent/term-provenance25 and contexts/inline source verification26 passed;
build and layering passed; quick passed. CUDA ledger11 passed10.42s (actual
CUDA both-port adapter-use cases and Engine captures; replay/cached unit fixtures
in that file remain CPU). Earlier shared-core CUDA qualification separately
covers the masked graph stop adjoints. No production cloud enablement is claimed.
