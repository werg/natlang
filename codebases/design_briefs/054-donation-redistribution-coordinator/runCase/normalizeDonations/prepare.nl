---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for normalizeDonations. Interpret offers and condition evidence, distinguishing available items from tentative promises. Record quantities, collection constraints, and any permitted-use restrictions; request missing information that changes suitability rather than accepting vague offers into inventory. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
