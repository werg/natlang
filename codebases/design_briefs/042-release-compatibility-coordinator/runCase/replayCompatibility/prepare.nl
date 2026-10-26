---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ConsumerDependency[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for replayCompatibility. Execute representative consumer scenarios against baseline and candidate in isolated environments. Compare intended outcomes rather than raw output equality alone, and identify whether a difference is acceptable, breaking, or dependent on an unresolved contract. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
