---
args: { prior: ReplicationCheckpoint|null, request: ReplicationPlanRequest, evidence: ReplicationEvidenceBundle, frame: ReplicationFrame, procedure: MethodProcedure, graph: ReplicationGraph, seeds: SeedSensitivityPlan, dataset: DatasetPin, plan: ReplicationTask[], artifacts: ArtifactCompleteness[], comparisons: ReplicationComparison[] }
returns: ReplicationCheckpoint
---
Preserve source method editions, dependency versions, preregistered seeds and dataset hashes. Record each execution plan against that exact configuration. Count a run complete only when every expected artifact and provenance identity verifies; crash outputs remain incomplete. Keep adaptations separate from exact replication and do not rewrite historical comparisons after a method or dataset change. Duplicate job receipt replay is a no-op.
