---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: AssessmentItem[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for constructRubrics. Define observable scoring criteria and partial-credit boundaries, separating conceptual correctness from presentation preferences. Include correct alternative methods and common errors, and state how ambiguity or missing evidence should be handled. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
