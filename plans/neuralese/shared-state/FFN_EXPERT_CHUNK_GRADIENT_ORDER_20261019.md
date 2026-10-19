# FFN expert-gradient chunk invariance

Status: implementation and focused CPU/CUDA validation are complete in an
isolated worktree. The production 0-to-4096 Mellum transition remains failed;
no tolerance waiver or live cutover is claimed.

The training policy remains the existing whole-graph objective: each physical
FFN row executes once in the same layer-major forward, and each trainable expert
parameter receives the cached-execution derivative. FFN chunk size is a memory
setting, not an objective change.

The exact shared mechanism will preserve each token-local chunk's parent
invocation, absolute row offset, and physical length through checkpoint
recomputation. Expert dW contributions will retain bounded references to chunk
gradient and routed-activation operands until the corresponding parameter's
existing cotangent anchor runs. At that anchor, the implementation will restore
reverse cached-invocation order, stable expert row order, and the original
per-expert 32-row tile alignment. A tile spanning chunks enters the grouped dot
once as one operand tile; splitting it into two dots and summing partials is not
equivalent. CUDA gathers tiles directly from chunk-backed operands, without
concatenating a full invocation. The CPU reference path packs one invocation to
preserve its existing matrix multiplication law. The existing FP32 accumulator
and CPU-file/device-FP32 storage choices remain the only parameter accumulators.

The implementation avoids a whole-invocation concatenation or hidden forward
replay on CUDA, preserves activation gradients, and releases staging references
when the anchor flushes or the scope closes. Stable tie ordering is required at
the router because concatenated chunk-local unstable sorts do not recreate the
whole-invocation expert row order. The rule applies to each shared expert
projection implementation used by both models, including checkpointed and
cached execution.

Focused validation covers uneven boundaries (including 4099 rows split
4096+3), multiple rows per expert, two invocations in one backward,
microbatch accumulation, base-plus-adapter routing, exact forward/input
gradients, and the grouped operand order. CUDA tests compare deferred projection
gradients with the existing grouped kernel; both-model flat/cached graph tests
cover chunked and unchunked execution with and without activation
recomputation. These focused tests pass. Full M315 production qualification
remains required before calling the runtime transition fixed.

Only the owner-set qualification policy selects full-graph versus cached
execution. Crisp code is limited to immutable row provenance, bounded operand
staging, and the grouped device kernel needed to preserve the exact numeric
reduction law.
