---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: FabricationRoute[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for calculateQuote. Use exact costing services with documented quantities, rates, waste assumptions, and supplier quotes. Distinguish fixed price from allowance or estimate, expose unknowns, and prevent a creative design change from silently invalidating the cost basis. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
