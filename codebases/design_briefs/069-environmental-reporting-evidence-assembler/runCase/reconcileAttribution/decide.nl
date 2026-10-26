---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Allocate shared activities using supplied rules and exact calculations, checking for duplicate records or overlapping contractor reports. Distinguish a transferred responsibility from a duplicated physical activity and preserve unresolved attribution rather than balancing by assumption.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
