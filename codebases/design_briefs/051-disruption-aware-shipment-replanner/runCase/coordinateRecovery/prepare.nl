---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ShippingAllocation[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for coordinateRecovery. Sequence approved bookings and communications, reconcile carrier acknowledgments, and track actual shipment progress against revised promises. Replan only affected legs when new events arrive and explain residual uncertainty rather than claiming recovery after booking alone. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
