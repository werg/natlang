---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: DefectObservation[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for traceExposure. Traverse exact lot, material, and process lineage to identify confirmed and precautionary exposure. Interpret process changes and rework histories, and avoid narrowing the scope merely because records are missing for some potentially affected units. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
