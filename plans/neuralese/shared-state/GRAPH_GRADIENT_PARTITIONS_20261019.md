# Graph gradient reduction partitions

Status: implementation integrated locally; full-suite and production cloud
qualification pending. No live cutover or production numerical-pass claim.

The one layer-major graph forward can produce exact logits and input gradients
while a combined FP32 weight-gradient reduction falls on the other side of a
BF16 rounding midpoint from the cached invocation reductions. The measured
seed-67 34-row versus 20+14-row Linear reproducer isolates this mechanism;
reverse invocation reductions match the cached accumulator and delivered BF16
parameter gradient bit for bit. This does not explain every production mismatch
until the complete comparison passes.

The policy remains the owner's full recipe on both students. This change is
exact arithmetic plumbing, not a new learned policy or a different objective.

1. Carry immutable invocation row partitions through the flat forward and its
   activation-checkpoint recomputation. Capture them in each autograd operation;
   never consult a later mutable context during backward.
2. Preserve the existing forward and input-adjoint kernels. Reduce each weight
   contribution in cached invocation order before its once-only parameter cast.
3. Preserve explicit row provenance through actor subsets and FFN chunks. Run
   the existing token-local FFN chunks at invocation-local boundaries: expert
   routing then constructs exactly the cached invocation-local padding, and each
   physical FFN row still executes once per layer. Attention and state-producing
   layers remain one layer-major graph traversal. This avoids a global padded
   expert reduction with zero masks, which is not arithmetically equivalent.
   Small invocations may add FFN launches; measure useful-update throughput on
   each actual GPU before claiming a performance gain.
4. Cover embedding frequency rules, norm/head axes and finite convolution state,
   in addition to ordinary dense projections. Retain higher-order functional
   arithmetic and exact optimizer/RNG/exposure restoration.
5. Verify the midpoint reproducer and varied genuine producer/consumer graphs
   against independent cached execution, including actual low-precision
   parameter gradients. Check the whole production inventory and capacity on
   each cloud model before replacing its useful diagnostic fallback.

No tolerance relaxation, precast-only qualification, recursive graph-forward
replay or hidden omission of trainable parameters is authorized by this work.

The branched CUDA checkpoint checks exposed a second independent mechanism:
checkpoint recomputation could lose the ContextVar carrying the exact parameter
cotangent scope. Capture that live scope alongside immutable row partitions;
refuse recomputation after its owning operation closes. This preserves the same
forward save/restore path instead of disabling checkpoint metadata checks.

Independent review reproduced an interleaved-actor fan-in bug with a nonzero
prior microbatch accumulator. Actor buckets A={0,2}, B={1} deliver B1,A2,A0
rather than cached A2,B1,A0. Shared graph execution therefore keeps invocation
order within each layer's actor-local projections/FFN, even when identities
repeat. Every physical token still runs once per layer. This also keeps actor
adapter routes exact; it does not merge or relabel actors.

Focused integration evidence (2026-10-18 UTC): the complete-gradient/shared-state/
backend/convolution set passed 136 tests; checkpoint/functional higher-order and
port/actor set passed 102; the actor ordering/graph set passed 37 after updating
its structural assertion to invocation-local visits; final context, embedding
frequency, convolution and actor regression set passed 31. The independent actor
review supplied the nonzero prior-accumulator counterexample and verified its
actual shared graph dispatch regression. Production all-parameter comparison and
capacity remain required before either live runtime changes.
