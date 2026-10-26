---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: AnalysisSlice[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for explainResult. Formulate the result using exact computed values and the approved question interpretation. Explain exclusions, uncertainty, and unanswerable components; attach the executable work order and lineage so another analyst can reproduce or revise the analysis. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
