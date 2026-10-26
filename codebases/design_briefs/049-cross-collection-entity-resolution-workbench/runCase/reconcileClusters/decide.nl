---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Build clusters using constrained graph operations rather than unconditional transitive closure. Before merging components, test all relevant cannot-link and temporal constraints; route inconsistent triangles to review and preserve uncertain alternatives.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
