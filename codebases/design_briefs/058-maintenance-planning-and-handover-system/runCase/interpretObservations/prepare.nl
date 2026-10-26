---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for interpretObservations. Reconcile inspection notes, service history, and operator reports to identify observed conditions and unresolved discrepancies. Separate symptoms from diagnosis, retain intermittent behavior, and identify information needed for routing under approved maintenance procedures. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
