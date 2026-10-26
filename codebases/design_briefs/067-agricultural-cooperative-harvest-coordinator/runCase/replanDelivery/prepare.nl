---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: HarvestPlan[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for replanDelivery. Process changed field observations, resource failures, or customer responses as scoped deltas. Reallocate only affected lots, obtain consent for material substitutions, and reconcile actual harvested and delivered quantities before declaring commitments fulfilled. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
