---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: WarehouseFinding[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for planResolution. Create bounded search, recount, relabel, inspection, or transfer tasks using approved warehouse procedures. Sequence dependencies and require independent confirmation for consequential inventory adjustments or uncertain identity changes. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
