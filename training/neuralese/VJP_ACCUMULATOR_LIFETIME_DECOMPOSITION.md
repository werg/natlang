# Audited VJP accumulator lifetime

Policy: preserve complete gradients and recoverable useful training while removing storage whose only consumer has finished. A VJP scope represents one forward/backward invocation. Update scopes may span several backward windows and keep their accumulators until finalization. Explicit higher-order reference paths retain their differentiable graph.

1. Each low-precision leaf receives the fully combined carried gradient. Compare it with the single cast of its FP32 accumulator using the existing exact audit.
2. After the audit succeeds, retire only that VJP accumulator. Keep contribution counts, anchor completion and audit metadata through finalization; retirement grants no training or readiness credit.
3. Reject any later contribution, anchor or gradient audit that attempts to reuse a retired VJP entry. A retained graph must obtain a fresh scope; it must never silently allocate a replacement and lose prior contributions.
4. Preserve update-mode multi-window accumulation and differentiable higher-order/reference behavior. Prove parameter/state/optimizer equality and verify actual live allocation peaks before production selects device FP32 storage.

These are crisp graph/storage lifetime verifiers, not a new sampling or training policy. Both model ports use the shared carrier. Numerical execution and whole-row capacity remain separately qualified on the actual successor runtime.
