---
args: { tasks: ReplicationTask[], receipts: ReplicationJobReceipt[] }
returns: ArtifactCompleteness[]
---
For each job, compare the exact expected artifact set to completed output receipts, hashes and task identity. A crashed run with partial outputs is incomplete and not reusable unless every required artifact and provenance binding is verified.
