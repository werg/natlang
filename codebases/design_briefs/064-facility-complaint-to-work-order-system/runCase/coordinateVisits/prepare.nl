---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: FacilityWorkOrder[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for coordinateVisits. Schedule visits using exact availability and resource constraints, obtain required access consent, and coordinate multiple trades. Preserve cancellations and failed-access events, and update occupants with truthful windows rather than unsupported precise arrival promises. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
