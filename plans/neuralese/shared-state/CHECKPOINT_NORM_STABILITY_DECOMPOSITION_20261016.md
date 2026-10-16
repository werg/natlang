# Preserve the norm graph through checkpoint recomputation

The existing training policy and native graph objective stay unchanged. Exact
checkpoint replay is crisp tensor-execution plumbing: a compiler fallback must
not change which intermediates a checkpoint saves between forward and backward.

1. Reproduce the observed compiled FP32 RMSNorm→eager fallback across a native
   checkpoint with real CUDA tensors and both native ports.
2. Extend the existing eager-norm scope with a compiled-only setting. In that
   scope FP32 dynamic compilation is bypassed consistently on forward/recompute;
   the existing BF16/FP16 fused kernel remains selected exactly as before.
3. Apply that scope inside the existing shared checkpoint layer function, so
   ordinary execution, actor re-entry, higher-order scopes, backend contexts,
   CPU/file input storage and RNG replay retain their existing owners.
4. Compare actual outputs and every input/parameter adjoint against the same
   native graph without checkpointing under the matching arithmetic scope;
   assert BF16 fused dispatch remains active during checkpoint recomputation.

This fixes a shared first-order checkpoint correctness issue, not a numerical
qualification waiver or a different training objective. The old cached-port
scope already avoids this failure; ordinary uncheckpointed dispatch is retained.

Evidence: deterministic real CUDA compiled→eager fallback reproduced the saved
metadata mismatch before repair. Repaired focused CUDA suite passed 13 cases
(9.50 s, 2.1 GiB), including the exact standing-suite physical-index failure,
both ports and FP32/BF16 bitwise outputs plus every input/parameter adjoint.
Ten additional backend-scope/offload/reference-HVP/fused-norm cases passed on
actual CUDA (7.88 s, 2.2 GiB). The BF16 test asserts real fused-kernel calls on
both forward and recompute. Ordinary uncheckpointed dispatch is unchanged;
FP32 checkpoint execution uses the same eager rule already used by cached-port
checkpointing, consistently rather than depending on mutable compiler caches.

The full suite then exposed a second consequence: checkpoint-only eager FP32
norms differ from compiled cached norms, and diagnostic TF32 projections amplify
the rounding difference. Native model construction now selects the existing
eager FP32 rule on its own Maple norm modules for every path, including cached
serving, reference execution, final readout and uncheckpointed graph execution.
This is per-model mechanism state derived from the native wrapper; ordinary
non-native Maple models retain compiled dispatch. BF16/FP16 custom fused norms
remain unchanged. Tests cover real cached action execution as well as the graph
oracle, with unchanged forward/gradient tolerances and compiler independence.
