---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: WarehouseWorkOrder[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for reconcileInventory. Compare completed tasks with actual scans and custody events, apply approved exact adjustments, and verify conservation across locations. Preserve unresolved differences and distinguish a found item from a fulfilled customer order. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
