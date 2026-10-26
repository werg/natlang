---
args: { request: ReplicationPlanRequest, procedure: MethodProcedure, graph: ReplicationGraph, seeds: SeedSensitivityPlan, dataset: DatasetPin }
returns: ReplicationTask[]
---
Compile only method steps with resolved commands, inputs, source hashes, environment dependencies, preregistered seeds and expected artifact contracts. Preserve the original method requirement even where a dependency substitution is proposed; leave non-executable or ambiguous steps out of runnable tasks and return them as blockers.
