---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: AssessmentBlueprint[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for generateItems. Create varied tasks that elicit the target evidence rather than superficial recall alone. Specify required context and resources, avoid hidden assumptions, and generate plausible alternative interpretations for subsequent review. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
