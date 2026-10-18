# Finite-convolution cotangent order

## Design

The cached native reference applies a finite-convolution operation once per
physical token. Autograd visits those independent operations in reverse token
order. The layer-major graph applies the same operation to a token sequence in
one call, so its conventional row reduction can produce a different rounded
BF16 parameter gradient near a rounding midpoint.

Preserve the cached operation order as exact gradient plumbing while keeping
the graph forward single-pass. When graph token provenance is active, compute
each token's FP32 weight product and fold it into the existing shared
accumulator in reverse physical-token order. Start from the accumulator's
current value, which may contain earlier microbatches. Bias follows the same
order. Input and carry gradients retain the existing formulas.

## Implementation boundaries

1. The finite-convolution custom backward calls one shared ordered accumulator
   helper only when exact graph token provenance is present. Without provenance,
   the existing path is unchanged.
2. The file-backed CPU backend batches product construction to at most 16 MiB,
   transfers each batch once, then adds its rows in reverse order directly to
   the existing accumulator. This bounds temporary memory and preserves
   microbatch state.
3. The `device_fp32` backend launches a Triton elementwise kernel. Each lane
   owns one accumulator coordinate and loops over tokens in reverse order,
   with floating-point fusion disabled. It performs no per-token host transfer.
4. This changes only the reduction order required to match cached execution.
   It does not alter model policy, forward values, activation gradients,
   checkpoint state, or the accumulator backend setting.

## Verification

The focused CPU and CUDA fixture exercises the BF16 midpoint that distinguishes
flat from cached execution, verifies the preexisting-accumulator/microbatch
case, and compares weight and bias gradients. Both accumulator backends must
match the cached per-token reference exactly. It also asserts unchanged forward
values. Test evidence is recorded with the change; it is not a production
qualification by itself.
