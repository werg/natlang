# Bounded expert weight preparation

Policy: retain every graph, token and gradient; remove redundant immutable weight
preparations within one token-local FFN call. This is exact execution plumbing,
not a change to task selection, model prompts, arithmetic or optimizer policy.

1. `token_local_chunks` owns one short-lived preparation scope for its callbacks,
   including checkpoint recomputation. Nested calls own independent scopes.
2. Expert FP8 preparation reuses the cast weight, quantized weight and scales only
   for the same tensor object, version, shape, stride, dtype, device and format.
   Keep the original tensor alive so object identifiers cannot be recycled.
3. Release the scope after the FFN call; autograd retains only the tensors needed
   by its actual saved contexts. Never reuse across updates or unrelated calls.
4. Compare actual chunked expert outputs, input/weight gradients and saved weight
   storage with reuse disabled, including checkpoint recomputation. Preserve the
   existing shared cotangent and invocation reduction mechanisms.

Mellum V5 reached30469 then OOM during FP8 expert recomputation despite passing
maximum-graph capacity checks. Duplicate preparation is a concrete allocation
defect; its contribution to the production failure requires a real continuation.
Do not claim this alone explains the complete capacity-check/runtime discrepancy.
