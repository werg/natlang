---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: StockIdentityFinding[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for classifyException. Interpret the discrepancy against confirmed identity and movement evidence. Separate damage from quantity shortage and pending receipt from true absence; identify which inventory and fulfillment commitments may be affected without broad unsupported holds. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
