---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ReuseConcept[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for planInvestigations. Identify the surveys, specialist reviews, or evidence needed to distinguish feasible concepts. Prioritize investigations by their ability to change a decision, and specify how positive, negative, or inconclusive findings would alter the concept. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
