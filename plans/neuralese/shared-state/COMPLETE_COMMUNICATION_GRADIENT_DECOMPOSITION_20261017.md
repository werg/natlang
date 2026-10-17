# Complete communication derivative qualification

The supervising recipe continues useful training while exact qualification
records what has and has not been proved. Existing channel checks compare
producer/consumer outputs, consumer CE, adapter derivatives and native splice
execution. They are real evidence, but they do not prove every trained backbone
derivative. Cross-row causal-prefix sharing is a separate optional operation.

1. Bind the actual model/arithmetic/backend, complete declared production
   trainable inventory, original graph cases and consumer-loss normalization.
   Distinguish frozen vocabulary weights from learned sparse input/output rows,
   low-rank embedding parameters, backbone parameters and input embeddings.
2. Compare all used declared parameter derivatives and connected input
   embeddings between the existing one-pass graph executor and explicit cached
   execution. Record unused parameters honestly; require nonzero derivatives
   through the producer/input, adapter, backbone and learned input/output roles.
   Preserve absolute and relative/cosine gates; tiny absolute gradients alone
   cannot qualify a missing derivative.
3. Keep one full gradient set resident. During the cached backward, temporary
   leaf hooks compare each arriving gradient and release its old counterpart.
   Use autograd.grad exclusively: never accumulate parameter.grad or arm the
   in-backward optimizer. Preserve existing gradients, optimizer, RNG, backend
   state and train/eval/checkpoint modes. No giant disk copy or second full
   model/gradient allocation is needed.
4. Label old adapter-only checks as channel evidence. Full foundation/shared
   readiness requires a passed complete derivative proof, bound to the exact
   trained identity and inventory. A missing or failed complete check prevents
   readiness; it does not erase actual exposure or stop ordinary useful SGD.
5. A new explicit cotangent profile must pass these same checks on complete
   communication graphs, as well as its separate prefix and actual optimizer
   checks. Do not silently replace the active arithmetic or infer qualification
   from an ordinary SFT capacity measurement.

The numbered policy is natural language. Crisp code is limited to exact
inventory/case/identity verification, tensor derivative comparisons, temporary
hook plumbing and durable qualification receipts. Existing trainers, graph
execution, cached execution, objectives and readiness remain the shared
interfaces for both models.

The CPU audit at
`/home/werg/data/dgxops/native-communication-complete-gradients-20261017.json`
demonstrates why the distinction matters: on the real tiny Mellum
three-invocation graph BF16 outputs/loss/input adjoints match bitwise, while
some per-invocation versus full-row BF16 backbone weight derivatives differ.
This is not a full-size FP8 failure claim or a reason to interrupt active runs.

Stateful backends execute the flat forward and its full VJP first. Only then
restore the exact pre-forward backend state and RNG and construct the cached
forward/VJP lazily. This prevents the cached forward from changing scaling
history retained by a flat backward, and avoids retaining both activation
graphs. The outer shared qualification restores backend extras, counters,
execution flags and RNG on success and failure. A stateful-autograd regression
rejects any intervening backend mutation before its VJP.

Ordinary and joint semantic publication use the same complete qualifier under
their declared live backend. CPU fixtures explicitly declare FP8 off; they
cannot certify the production Mellum FP8 default by replacing its loader.
Historical row credits and full checkpoints remain unchanged when a complete
proof is absent or fails. No CPU fixture is a full-size production capacity or
FP8 qualification. The retained-gradient and comparison-pair byte counts are
not peak memory measurements: cached partial accumulations, autograd return
tuples, metric temporaries and actual destination activations also consume
memory and must be measured before a full-size cloud proof.

Actual trained LFM diagnosis (2026-10-17)
---------------------------------------

The full step9478 parent is
`f07a57e3d98224360b8a509f8f410be3de2fd37830e176c989164d2e83ebcb89`
(2,523,898,927 bytes), restored by independent private GET. Its complete
communication check compares 150 declared trainable parameters plus connected
input embeddings. Under the recorded production arithmetic, 100 of these 151
comparisons fail the existing strict absolute/relative elementwise gates; all
are finite. Forward difference is 1.43e-6 and consumer CE difference 1.14e-5.
Selecting IEEE linear arithmetic alone does not resolve this (101 failures).
These are failed readiness checks, not evidence that actual SGD should stop.

Intermediate tracing localizes the first forward discrepancy in layer zero:

- Adapter output, layer input and elementwise squared norm input are bitwise
  equal between flat and cached execution.
- The FP32 `mean(-1)` reduction first differs by 1.164e-10 when invocation
  shapes differ. Reciprocal square root then differs by 3.815e-6, and the norm
  output by 4.768e-7. This occurs with both recorded and IEEE GEMM arithmetic.
- Calling that same norm one token row at a time gives bitwise matching flat
  and cached norm outputs. This diagnostic isolates reduction scheduling;
  it is not a proposed serial implementation for production.

A subsequent local diagnostic reuses the existing shared Triton RMSNorm on
all 45 HF LFM norms, asserting FP32 inputs/weights to preserve their dtype
semantics. The complete check still fails 96/151 comparisons, with forward
1.07e-6 and CE 1.24e-5 differences. Therefore making the first reduction
independent of row count alone is not a complete repair. No replacement norm,
changed threshold, optimizer update, checkpoint rewrite or production
qualification was adopted from these diagnostics. Complete derivative and
actual optimizer-update checks remain necessary; do not describe the failed
checks as a proven model limitation or an established task-regression cause.

Durable owner-host receipts in `/home/werg/data/dgxops/` are
`lfm9478-complete-gradient-diagnostic-root-v2-20261017.json`,
`lfm9478-norm-reduction-divergence-root-20261017.json`, and
`lfm9478-existing-fusednorm-gradient-diagnostic-root-20261017.json`.
The restored parent remains assigned to this unresolved investigation.
