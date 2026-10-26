---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: AnalysisWorkGraph[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for executeAndAudit. Execute finite partitions, cache semantic judgments by record and rubric version, and reconcile every input disposition. Audit sampled positive and negative labels, exact join cardinalities, temporal boundaries, and denominators before materializing the final aggregate. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
