---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: BehaviorObservation[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for classifyMismatch. Reconcile requirements, release decisions, tests, and observed behavior to determine which artifact is wrong. Do not automatically rewrite documentation to match a bug; preserve unresolved intent questions for the responsible product or engineering owner. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
