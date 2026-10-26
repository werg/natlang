---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: QualityActionPlan[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for verifyEffectiveness. Compare implementation evidence and subsequent production observations with the predicted corrective effect. Detect recurrence and unintended consequences, retain unresolved exposure, and require qualified approval before lifting holds or closing the investigation. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
