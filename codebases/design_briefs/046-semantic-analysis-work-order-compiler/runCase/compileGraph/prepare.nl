---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: DataSuitability[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for compileGraph. Construct a typed DAG of exact filters, joins, bounded semantic classification, temporal matching, and aggregation. Push exact restrictive filters before semantic work when meaning is preserved; attach provenance and missing-data rules to every edge and reject unsupported operations. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
