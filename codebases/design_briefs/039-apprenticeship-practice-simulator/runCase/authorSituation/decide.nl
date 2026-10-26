---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Construct a realistic situation requiring the supplied procedure and meaningful information gathering. Separate facts visible at the start from facts revealed by appropriate questions, and include complications that test judgment without requiring unauthorized actions.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
