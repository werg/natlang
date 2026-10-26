---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: FacilityIssueGroup[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for formWorkOrders. Choose approved inspection or repair task types and required trade qualifications. Sequence dependencies, include access arrangements and occupant impact, and route safety-sensitive or uncertain cases to qualified personnel rather than generating improvised repairs. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
