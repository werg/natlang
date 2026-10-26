---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: GardenPreference[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for modelPlots. Reconcile plot records, field observations, and supplied gardening guidance to describe known conditions and constraints. Preserve uncertainty about soil or other untested properties, and route consequential interventions to appropriate local expertise. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
