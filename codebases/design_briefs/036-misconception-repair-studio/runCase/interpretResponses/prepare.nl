---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for interpretResponses. Inspect the learner’s response against the task and accepted solution space. Distinguish conceptual errors, arithmetic slips, misunderstood wording, and incomplete work; preserve alternative correct strategies rather than forcing one canonical explanation. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
