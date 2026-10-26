---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: AssetObservation[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for selectProcedure. Match the asset variant and observed condition to approved procedures, checking revision, qualification, and required isolation or access conditions. Escalate unsupported cases rather than inventing maintenance instructions from general knowledge. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
