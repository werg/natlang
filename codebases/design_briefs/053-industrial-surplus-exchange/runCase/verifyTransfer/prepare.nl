---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ExchangePackage[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for verifyTransfer. Coordinate approved transfers and verify quantity, condition, ownership, and recipient acceptance at handoff. Record rejected or partially accepted material and update future matching evidence without falsely reporting a completed reuse benefit. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
