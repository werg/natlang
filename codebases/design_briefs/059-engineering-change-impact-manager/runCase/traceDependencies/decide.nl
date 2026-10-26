---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Traverse bills of material, interfaces, process instructions, and supplier commitments through exact dependency graphs. Judge whether each indirect dependency is materially affected and retain uncertain impacts for engineering review instead of broad automatic replacement.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
