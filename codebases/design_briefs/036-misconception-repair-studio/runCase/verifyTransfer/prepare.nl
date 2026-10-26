---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: LearningIntervention[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for verifyTransfer. Assess performance on new tasks that change surface details and require the intended concept. Distinguish independent transfer from success under hints, update the learner model conservatively, and retain evidence for instructor review. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
