---
args: { request: SemanticWorkOrderRequest, proposal: SemanticWorkOrderProposal, receipt: AnalysisMaterializationReceipt, expectedGraphDigest: string }
returns: MaterializedRowCoverage
---
1. Require receipt.requestId, graphDigest and schemaSnapshotVersion to match this work order's authorized execution and the supplied graph digest. Missing or stale identity makes all output unpublishable.
2. Compare actual receipt row IDs with proposal.execution's included, excluded, inaccessible, and missing populations. No inaccessible, excluded, unresolved, or missing row may appear as an included result. Any unexplained population difference is a coverage gap.
3. Require the actual rows and cardinality evidence to preserve proposal.graph.outputGrain and every declared join relationship. A fanout or grouping change is a different analysis, not a complete run.
4. Return protected row IDs and exact coverage/grain reasons. Do not infer a zero count for inaccessible records.
