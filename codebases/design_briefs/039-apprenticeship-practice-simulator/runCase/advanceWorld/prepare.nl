---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: LearnerAction[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for advanceWorld. Apply the exact simulator’s legal transition rules, then formulate context-appropriate responses from actors who know only their own information. Preserve consequences across turns and do not reveal hidden facts merely to help the learner succeed. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
