---
args: { task: ReplicationTask, referenceReceipt: ReplicationJobReceipt, replicationReceipt: ReplicationJobReceipt, completeness: ArtifactCompleteness[] }
returns: ReplicationComparison
---
Compare these actual run receipts only after their expected artifact manifests verify complete. Bind both to the task's source, dataset, environment and preregistered seed. Preserve the exact output hashes and describe a result difference without recasting an adaptation as exact reproduction. If either run is incomplete or a provenance binding differs, keep the pair unresolved and state the missing artifact or mismatched identity.
