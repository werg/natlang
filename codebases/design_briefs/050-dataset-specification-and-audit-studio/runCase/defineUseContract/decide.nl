---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Interpret the intended application and decisions the dataset will support. Define target population, task boundaries, prohibited uses, and meaningful evaluation slices; avoid treating easily available data as representative of the intended use.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
