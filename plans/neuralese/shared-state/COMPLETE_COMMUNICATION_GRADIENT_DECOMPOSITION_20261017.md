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
