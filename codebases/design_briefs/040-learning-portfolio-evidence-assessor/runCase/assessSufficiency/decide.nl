---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Evaluate breadth, recency, independence, and consistency under the supplied assessment standard. Prevent double-counting one artifact as multiple independent demonstrations and preserve contradictory evidence rather than averaging it away.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
