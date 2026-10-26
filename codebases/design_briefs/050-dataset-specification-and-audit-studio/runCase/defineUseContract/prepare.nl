---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for defineUseContract. Interpret the intended application and decisions the dataset will support. Define target population, task boundaries, prohibited uses, and meaningful evaluation slices; avoid treating easily available data as representative of the intended use. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
