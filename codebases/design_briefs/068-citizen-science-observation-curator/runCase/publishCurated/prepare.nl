---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ObservationQuestion[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for publishCurated. Prepare a curated record with confidence categories grounded in evidence, preserved alternatives, and project-specific disclosure controls. Publish provisional and verified statuses distinctly and update derivatives when experts or observers provide corrections. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
