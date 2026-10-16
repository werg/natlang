# Reuse exact graph convolution history indices

The existing natural-language training recipe still decides which complete
graphs to train and which numerical checks qualify an implementation. This
change is crisp graph-execution plumbing: the causal predecessor relation is
already fixed by the immutable graph plan, convolution width and device.

1. Construct that exact integer gather index once for each width in a graph
   forward pass, including explicit zero-padding and external-state positions.
2. Reuse it across layers and activation-checkpoint recomputation in that pass.
3. Keep convolution products, FP32 reduction order, weights, gradients and state
   transport unchanged. Do not reuse an index across different graph plans.

Verify ordinary and imported-state execution against the existing cached
reference, including gradients and checkpoint recomputation. Measure eliminated
host construction separately; no full-weight cloud throughput claim follows
from a local index benchmark. This does not hold up the current cloud handoff.
