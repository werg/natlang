---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: StockObservation[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for matchIdentity. Compare descriptions, packaging, serials, lot information, and movement history to identify candidate stock. Distinguish a mislabeled item from a misplaced one, and request a discriminating scan or inspection rather than choosing the closest description. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
