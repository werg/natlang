---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: HarvestReadiness[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for recoverCommitments. Reconcile delivery agreements and customer preferences to identify quantity, quality, and timing requirements. Separate permitted substitutions from assumptions and retain dependencies such as processing or storage windows that constrain otherwise feasible harvest choices. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
