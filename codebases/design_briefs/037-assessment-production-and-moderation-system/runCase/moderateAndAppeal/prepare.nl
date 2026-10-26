---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ItemChallenge[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for moderateAndAppeal. Compare scorer disagreements and appeals against the released rubric and actual response. Identify systematic rubric defects, propose scoped corrections, and apply approved changes consistently to all affected learners rather than only those who appealed. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
