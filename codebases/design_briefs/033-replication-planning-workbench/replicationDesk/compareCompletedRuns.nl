---
args: { tasks: ReplicationTask[], receipts: ReplicationJobReceipt[], completeness: ArtifactCompleteness[] }
returns: ReplicationComparison[]
---
Compare only completed, artifact-complete jobs bound to the expected source, dataset, environment and seed plan. Separate execution differences from data/version adaptations and incomplete crashes; never describe unverified partial output as replication evidence.
For each pair of actual reference and replicated run receipts for the same task, bind comparison = compareOneCompletedRun(task, referenceReceipt, replicationReceipt, completeness). Append a comparison only when both receipts are complete and their source, dataset, environment and seed bindings match; otherwise retain the precise mismatch as unresolved.
