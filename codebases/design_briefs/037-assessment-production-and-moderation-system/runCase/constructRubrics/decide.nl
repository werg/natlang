---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Define observable scoring criteria and partial-credit boundaries, separating conceptual correctness from presentation preferences. Include correct alternative methods and common errors, and state how ambiguity or missing evidence should be handled.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
