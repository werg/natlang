---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: CompetencyMapping[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for assessSufficiency. Evaluate breadth, recency, independence, and consistency under the supplied assessment standard. Prevent double-counting one artifact as multiple independent demonstrations and preserve contradictory evidence rather than averaging it away. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
