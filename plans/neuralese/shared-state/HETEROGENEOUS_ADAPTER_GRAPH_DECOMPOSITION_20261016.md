# One-pass native graphs with distinct invocation actors

Policy: a typed native graph gradient operation may bind a different exact
adapter actor to each invocation. Internal source-to-consumer communication
must retain producer gradients and run the graph once, layer by layer. Actors
share the same immutable native base and tokenizer/layout contract; this is not
cross-base semantic interoperability. Existing single-actor requests retain
exact behavior and identities.

The service is a typed graph-to-gradient operation. Natural-language instructions
specify the intended producer/consumer roles; crisp mechanisms verify exact
bindings, execute the differentiable graph and preserve its immutable artifacts.
No stage chooses an actor implicitly, replaces an invocation with text or replays
a producer trajectory recursively.

1. Extend the existing graph request with an explicit invocation-name-to-adapter
   binding map. An omitted invocation inherits the request actor; an explicit
   empty list means the unadapted base. Reject unknown invocation names, foreign
   bases, missing coefficient/basis artifacts and inconsistent target leaves
   before modifying model objects. Bind the declaration into gradient receipts.
2. Resolve each distinct actor once through the existing adapter resolver and
   effective-weight construction. One operation owns the engine lock, coefficient
   leaves and a lifetime lease through forward, checkpoint recomputation and
   backward. Each actor scope restores exact original module/parameter/buffer
   objects after the selected operation; expired banks fail closed.
3. Extend the existing layer-major executor with a physical-own-token actor map
   derived from GraphPlan.node_ids. Group rows by actor for the existing
   _layer_inputs projections, scatter Q/K/V or convolution inputs to physical
   order, then perform the existing single shared causal attention or predecessor
   gather. Group rows by actor for _finish projections/FFN and scatter outputs.
   This is still one graph forward: every physical token is processed once per
   layer, imported aliases gather the producing row, and no invocation runs its
   own recurrent forward. Actor grouping reuses the actual model projections,
   expert routing, FP8 backend and precision/cotangent contracts.
4. Keep shared positional rotations, masks, source aliases and imported-state
   identities exact. Convolution kernel weights are not current admissible
   adapter targets; adapted convolution input/output projections execute in the
   correct actor groups. If a future target changes this fact, extend the
   mechanism explicitly rather than assuming one kernel for all actors.
5. Store each captured producer state with its actual invocation actor. Preserve
   source and consumer actor identities in graph, archival and objective receipts.
   Include all coefficient and frozen-basis dependencies recursively. Default
   strict transport remains strict; explicit same-base transport retains both
   actors without text reencoding.
6. Reuse existing graph CE and policy objectives. Differentiate selected actor
   coefficient leaves in the same backward, including consumers reaching source
   coefficients through internal KV/hybrid state. Current-policy updates retain
   immutable behavior bindings and durable moments/RNG; no actor-wide parameter
   reset or stale actor relabeling.
7. Validate both actual ports against explicit cached per-actor execution:
   gathered hidden/logits/loss, all selected coefficient adjoints, source-to-reader
   gradients, repeated actors, empty imports, external raw state, checkpoints,
   actual HTTP services, two optimizer updates, archival/restart and unchanged
   base state. Require executed FP8 counters for claims about that backend.
   Existing strict numerical gates apply; cloud runtime qualification remains
   distinct from local qualification. Advertise only after independent review.

Implementation is pending. This decomposition does not claim that per-invocation
actors currently execute, nor that existing unsupported routes are complete.

Internal executor implementation now exists in `model/native_actor_graph.py`.
Every effective matrix is materialized once from captured base objects before
any functional scope. Nested adapted scopes and use after the operation lease
expires fail closed. The ordinary single-actor path and actor banks share the
same effective-weight constructor. Graph attention stays combined; only layer
projections and feed-forward are grouped, with a linear physical-row index.

Local mechanism evidence: `native-invocation-actors-engine-oracle-20261016`
passed4 cases in37.61s, both real CUDA ports with checkpointing on/off, all
admissible layer matrices, different XS/tiny actors and a repeated actor. The
independent incremental oracle executes actual Engine._native_forward per token
and the same exact native splice mechanism on differentiable source captures.
All hidden/logit/loss and coefficient-gradient gates remain strict, including
consumer loss reaching the producer's coefficients. This is an oracle only;
the graph executor does not invoke it. Source/target identity receipts, public
service admission/gradient artifacts, immutable optimizer continuation and
heterogeneous RL capture/replay remain unfinished and unadvertised.

`native-invocation-actors-qualified-20261016` passed25 CUDA cases in91.60s
(new executor plus existing all-target/FP8/backend carrier coverage) and quick.
Forward instrumentation verifies exactly one physical-row visit per layer and
two distinct-actor groups, not three invocation replays. Failure/lifetime tests
verify exact restoration. `native-invocation-actor-request-bindings-20261016`
passed the focused request-binding reuse regression and quick: canonical actor
matrices may be reused while each invocation retains its own request spelling,
including explicit empty versus inherited base bindings.

Exact receipt/service continuation is implemented: each result and captured
producer binds the actual invocation actor; incorrect caller-supplied capture
identity is rejected. Versioned `natlang.native-invocation-actors/1` receipts
bind every invocation and the exact common base. Graph gradients and immutable
coefficient updates retain this receipt plus its SHA, with no invented single
effective model identity. Updating one coefficient rewrites every affected
invocation binding while retaining other actors, scale/order, moments and all
unselected coefficients. Python and TypeScript native tensor dependency walkers
archive every actor's coefficients and frozen bases; .nz and wire roundtrips
retain the complete closure.

The typed graph-CE service accepts `invocationAdapters`; source, split and
artifact declarations are checked before execution. Real consumer-only CE reaches
the producer value-projection coefficients. The final-layer producer query
projection is structurally absent from exported same-layer KV in the tiny LFM
fixture, so its zero gradient is not used as an update test. The value-projection
test requires actual nonzero gradients and changed coefficients on every update.
Unknown invocations and held rows fail closed. An existing nested provider-error
parser bug initially dropped recovery text as `[object Object]`; the native
client now shares the completion protocol unwrapper and preserves real codes.

Terminal evidence: `native-invocation-actor-receipts-20261016` 6 passed38.60s,
including real CUDA ports, exact actor capture and cross-language wire/.nz
closure. `native-invocation-actor-errors-typed-final-20261016` built the typed
client, passed both real HTTP cases17.06s (two Adam updates, archive reopen,
remote eviction/restoration, untouched actor/base isolation, source-name and
held rejection), and quick. Existing service regression coverage passed25 cases
in the preceding integration; its two newly added error-text tests exposed and
then verified the parser repair. `native-invocation-actors-executed-fp8-20261016`
passed2 cases17.69s: current/current-bf16-dx each execute actual grouped FP8
projections in graph, cached reference and actual Engine oracle, checkpointed,
all selected coefficient adjoints held to the existing strict gates. These are
local Torch2.11/cu130/SM121 tests, not full-size or cloud qualification.

Remaining explicit work: heterogeneous native-policy behavior admission,
immutable per-turn actor bundles and current/behavior replay bindings. That
objective path rejects requests instead of silently using one actor. Public
capability flags remain unadvertised. Full-size memory/throughput is unqualified:
current matrices are materialized once per distinct actor; repeated actors share
matrices, but many distinct actors targeting complete expert banks increase
memory. Any bounded/tiled materialization must derive from immutable base values
and preserve exact rounded forward/adjoints under the same references.

The receipt digest is mandatory and verified by both artifact codecs and their
dependency walkers; a valid actor receipt cannot retain a stale graph digest.
Final `native-invocation-actor-receipt-hash-final-20261016` rebuilt TypeScript,
passed the actor-wire/digest forgery test and both real typed service/restart
cases, then quick. Existing corpus/checkpoint/recipe identity formats are
unchanged; this is a new explicit invocation-actor receipt binding.
