---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: DebtWave[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for verifyImprovement. Compare post-change workflow evidence with the original problem definition, accounting for changes in workload and observation. Distinguish successful delivery from actual improvement and reopen mechanisms that the completed work did not address. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
