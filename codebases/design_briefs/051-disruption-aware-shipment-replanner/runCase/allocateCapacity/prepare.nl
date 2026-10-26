---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ShippingAlternative[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for allocateCapacity. Submit candidate plans to exact capacity and timing services, then compare feasible allocations under supplied priorities. Reserve selected capacity before promising it, retain unfulfilled demand explicitly, and detect conflicts across shipments sharing scarce resources. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
