---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Interpret the discrepancy against confirmed identity and movement evidence. Separate damage from quantity shortage and pending receipt from true absence; identify which inventory and fulfillment commitments may be affected without broad unsupported holds.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
