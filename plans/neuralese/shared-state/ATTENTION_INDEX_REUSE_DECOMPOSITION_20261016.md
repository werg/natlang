# Reuse attention workspace indices within a graph execution

Supervision observed the same integer physical-row index rebuilt at each
attention layer and activation recomputation in both active trainers. CPU stack
sample shares include synchronization and do not establish GPU time attribution.

The exact mechanism prepares the existing query workspace and physical-row index
once per graph forward, then shares those immutable values with each attention
layer and its checkpoint recomputation. Ownership ends with that forward's
backward graph. No global cache, graph identity, tensor arithmetic, attention
kernel, mask, position, exposure or optimizer rule changes. Reference arithmetic
and non-graph/cached execution retain their existing paths. This is crisp shared
allocation plumbing, with no new natural-language policy or model-facing text.

Verify actual outputs and adjoints against the original rebuilding path on both
ports. Measure normal production batches before claiming throughput improvement.

The diagnostic's gradient-name join is exact verifier plumbing: accept the
optimizer's canonical parameter names and the older model-name aliases, joining
both by parameter identity. Reject any ambiguous name rather than guessing.
Both attention diagnostics share this join; no training behavior changes.
