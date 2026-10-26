---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: EvidenceTask[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for prepareAssessment. Assemble a reviewable recommendation with evidence, gaps, and proposed next work. Allow the learner to correct factual attribution, preserve assessor disagreements, and record the authorized decision separately from the program’s recommendation. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
