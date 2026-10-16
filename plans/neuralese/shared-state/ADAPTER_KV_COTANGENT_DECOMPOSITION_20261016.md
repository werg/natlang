# Exact-forward native cached KV cotangent carriers

Contract: an explicitly FP32 adapter-gradient operation keeps the existing
BF16 forward values, while accumulating shared KV cotangents in FP32 before
returning them to the producing activation. This applies to both graph and
cached execution. Default storage arithmetic and inference remain unchanged.

The original actual incremental-vs-flat certificate failed despite bitwise hidden,
logits, action logps and loss. At the final layer dQ is exact, while dK/dV differ
before earlier-layer dQ diverges. The FP64 adapter Jacobian contracts the actual
weight-cotangent gap to the observed coefficient gap. This is not permission to
relax its gate or describe fixed-observation replay as runtime qualification.

1. Crisp state plumbing stores exact already-rounded/rotated BF16 K/V values
   in FP32 carriers only inside the explicit high-precision gradient scope.
2. One shared attention operation consumes carriers. Its forward casts inside
   custom autograd and executes the existing sparse native Flex forward.
3. Its backward calls the existing native sparse Flex derivative directly,
   retaining the fixed delta and ordered masking. A private, explicitly marked
   lowering changes only dK/dV output storage to FP32, before graph/cache fan-in.
   The compiler template and all expected allocation sites are verified exactly;
   unknown runtime versions fail closed. No global registry mutation during
   compilation, recursive graph replay, dense fallback or surrogate gradient.
4. Check bitwise unchanged forward, exact dequantized/cast derivative oracles,
   actual per-token cached versus flat coefficient gradients and both ports.
   Record time/memory and target-runtime scope; do not advertise capability until
   the complete mode passes and independent review is finished.

This policy is the declared precision contract; crisp code implements exact
arithmetic, identities, storage and verification. Teacher feedback debt applies
only if the public error/setting contract changes. Heterogeneous per-invocation
weights remain a separate open requirement.

Implementation uses a pure `torch.compiler.allow_in_graph` boundary around the
existing Flex backward HOP: Dynamo does not inspect its reusable FX scalar score
graphs; AOTAutograd traces all explicit tensor/mask/bound inputs and Inductor
still lowers the same sparse native operation. No eager/dense fallback or
compiler-global setting mutation occurs. Forward and backward share one causal
predicate, kernel option constructor, tensor-layout contract and compiled-kernel
factory. Allocation interception verifies exact dK/dQ/dV AST signatures, not
only call counts. Unknown names/order/shapes/strides/dtype/device expressions or
conditional allocation sites fail closed.

CUDA checkpoint recomputation in the actual threaded HTTP service executes on
a different Python thread. A shared callback wrapper captures only the exact
native actor frames and re-enters them during recomputation. A live-operation
lease prevents deferred backward from falsely claiming an expired actor scope;
precision selection matches the actual model port even inside nested actors.
This also preserves context-dependent expert coefficient dtype lookup. The
callback includes backward and the functional substitution lock remains held.

Primitive evidence: bitwise original forward and dQ; F32 dK/dV round to the
original BF16 gradients exactly; whole-query cotangents match the FP64 sum of
per-query unrounded cotangents. Actual model evidence: the initial 18-case strict
both-port CUDA test passed (87.79s), including two real threaded HTTP updates and
actual incremental Engine gradient parity. Follow-up all-target/FP8/backend,
allocation guard and lifetime regressions are recorded in the terminal ledger
logs before publication. Local environment is Torch2.11.0+cu130, CUDA13.0,
SM121. Cloud NGC2.14/H100 and full-size memory/throughput remain separately
unqualified. Default storage/inference arithmetic is unchanged; feature
capabilities remain unadvertised until independent integration review.

Final local terminal evidence: `native-kv-carrier-qualified-contracts-20261016`
29 passed (78.68s), including actual executed FP8 projections and all selected
parameter targets; `native-kv-carrier-final-policy-20261016` 6 passed (49.96s),
including strict actual Engine current-policy gradients after two durable Adam
updates on both ports. The original Mellum coefficient max gap 3.9903447e-5
is reduced to 3.7252903e-9, relative L2 4.6071529e-8; no gate changed.
`native-kv-all-call-guard-20261016` 6 structural cases passed, including an
extra nested allocation call that would otherwise escape direct-assignment
inspection. Every allocation Call must be one of the three signed sites.

Typed contextual initialization validates the source-stream receipt against
both actual archived actors, the explicit caller declaration, exact logical
position, source token count and matching import observation. The shared native
restore helper restores source coefficient/basis closure before raw state;
recording preserves declared modes. Actual Python HTTP → Node client tests cover
both ports, forged receipt fields, .nz portability and remote eviction followed
by exact restoration. This is exact same-base transport, not cross-base semantic
compatibility. Heterogeneous per-invocation graph weights remain open.

`native-adapter-state-recorded-closure-20261016` rebuilt the actual TypeScript
client and passed both actual CUDA HTTP → Node cases in15.04s, including mode
recording and complete remote-eviction restoration. Its adjacent ledger log
records the final quick check. Prior initial client fixtures omitted the required
.nz dialect and failed only during serialization; those runs are not claimed
as passing. The corrected fixture asserts portable source and target actors.
