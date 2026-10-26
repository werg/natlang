---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: EnvironmentalMeasure[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for reconcileAttribution. Allocate shared activities using supplied rules and exact calculations, checking for duplicate records or overlapping contractor reports. Distinguish a transferred responsibility from a duplicated physical activity and preserve unresolved attribution rather than balancing by assumption. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
