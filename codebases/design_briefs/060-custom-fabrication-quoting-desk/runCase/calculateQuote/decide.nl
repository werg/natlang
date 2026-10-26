---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Use exact costing services with documented quantities, rates, waste assumptions, and supplier quotes. Distinguish fixed price from allowance or estimate, expose unknowns, and prevent a creative design change from silently invalidating the cost basis.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
