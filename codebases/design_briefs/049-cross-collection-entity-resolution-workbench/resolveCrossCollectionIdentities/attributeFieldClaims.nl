---
args: { clusters: ConstrainedIdentityClusters, resolution: EntityResolutionRequest }
returns: AttributedCanonicalView[]
---
Resolve each attribute independently using field-specific authority and effective time. Keep historical and current values separate. Preserve aliases, source citations, and competing claims when evidence conflicts; never overwrite source records or choose one global winner for all fields. If a source is revoked, remove its contribution and return affected links for reassessment.
