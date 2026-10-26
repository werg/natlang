---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: LocalProcedureConstraint[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for constructVariant. Rewrite steps for the local context using approved terminology and resources. Preserve sequence, guards, stop conditions, and verification; represent genuine deviations explicitly rather than hiding them in friendlier wording or inferred substitutions. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
