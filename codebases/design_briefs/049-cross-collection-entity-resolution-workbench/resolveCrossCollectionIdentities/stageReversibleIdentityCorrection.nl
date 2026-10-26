---
args: { canonical: AttributedCanonicalView[], clusters: ConstrainedIdentityClusters, resolution: EntityResolutionRequest }
returns: IdentityCorrectionPlan
---
For each proposed merge or split record the prior identity graph, exact changed edges, validity interval, downstream consumer links, and a reverse operation. When a previous merge is overturned, propagate the split to every dependent record and preserve the old cluster history. Block promotion until consumers can reverse or reconcile the correction.
