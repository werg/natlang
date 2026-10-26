---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: HarvestMatch[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for scheduleResources. Use exact scheduling and capacity services to coordinate labor, equipment, harvest windows, storage, and collection. Construct contingency plans for uncertain readiness and avoid assigning the same resource to overlapping lots or ignoring transport bottlenecks. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
