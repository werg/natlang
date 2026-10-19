# LFM real-update profile optimization

## Natural-language design

During each ordinary training update, the shared trainer should apply the same graph, adapter, loss, and optimizer decisions it already makes, while reducing avoidable launch and dispatch overhead in the measured linear and attention paths. It must preserve the existing tensors, precision, reduction order, checkpoint state, and per-row exposure. Architecture-specific attention may remain architecture-specific; any new dispatch or batching rule belongs in shared mechanisms and must not change model-visible behavior.

## Exact mechanism

1. Bind every optimization to the supplied four-update LFM profile and identify the source callsites behind the measured `_grouped_dense_mm` and flex-attention kernels.
2. Gather a graph's distinct supervised hidden rows once and split that selected tensor into loss chunks. That replaces repeated full-hidden scatter/zero work in each chunk's backward with one gather backward per graph. If physical query indices repeat within one graph, keep the old chunked gather path so repeated-index accumulation order stays unchanged.
3. Avoid per-graph device-to-host barriers for loss telemetry. Preserve each graph's exact GPU loss denominator and defer reading all graph weights back until the batch is complete; skip training-only accuracy telemetry because the update loop does not consume it. Keep the public loss helper's full metrics as the default.
4. Remove the per-call all-zero expert-index tensor from the statically single-expert grouped-matmul path. Leave routing tensors and arithmetic unchanged for actual expert-grouped operations.
5. Implement each mechanism in shared code, preserving the old gather and expert-index representations as numeric references.
6. Compare exact losses and all parameter gradients on ordinary and shared-prefix graphs for both architecture families. Measure launch/time reduction under the memory ledger.
7. Keep runtime configuration and checkpoint lineage unchanged; production adopts the shared code only after independent review and normal qualification.

## Justification

The supplied real-update profile records 35.13 seconds across four updates, 22.30 seconds of GPU work, 12.83 seconds of GPU idle, 11.30 seconds in 11,650 grouped dense matrix kernels, and 6.18 seconds in 24 flex-attention backward kernels. Source inspection found `packed_graph_loss` gathers each graph's hidden rows again for every CE chunk, creating repeated backward scatter/zero work; it also converts denominator and accuracy to Python floats inside the graph loop, synchronizing the device before backward. Standard training only consumes graph weights for boundary telemetry. The grouped-matmul profile also has 11,662 integer-fill kernels, matching its one-expert expert-index setup. The optimization preserves the exact loss denominator on GPU and only changes when its telemetry copy completes; it does not change training math.
