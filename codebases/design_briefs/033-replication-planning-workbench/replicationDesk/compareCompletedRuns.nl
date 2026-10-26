---
args: { tasks: ReplicationTask[], receipts: ReplicationJobReceipt[], completeness: ArtifactCompleteness[] }
returns: ReplicationComparison[]
---
Compare only completed, artifact-complete jobs bound to the expected source, dataset, environment and seed plan. Separate execution differences from data/version adaptations and incomplete crashes; never describe unverified partial output as replication evidence.
