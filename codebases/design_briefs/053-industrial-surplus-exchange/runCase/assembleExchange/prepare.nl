---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ReuseMatch[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for assembleExchange. Construct feasible multi-party exchanges using exact quantity, reservation, and transport services. Account for dependent legs, storage windows, and adaptation costs, and preserve unallocated surplus and unmet needs explicitly. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
