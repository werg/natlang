---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: DatasetUse[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for writeAnnotationSpec. Draft operational labeling instructions with examples, counterexamples, and uncertainty handling. Separate observable labels from inferred attributes, define adjudication, and test whether independent annotators can apply the rule consistently. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
