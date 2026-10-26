---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: PracticeScenario[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for interpretAction. Interpret the learner’s natural-language response against the available action set. Distinguish a proposed action from an executed one, ask for clarification when ambiguity affects consequences, and avoid rewarding jargon without the corresponding operational behavior. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
