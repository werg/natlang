---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: IdentityJudgment[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for reconcileClusters. Build clusters using constrained graph operations rather than unconditional transitive closure. Before merging components, test all relevant cannot-link and temporal constraints; route inconsistent triangles to review and preserve uncertain alternatives. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
