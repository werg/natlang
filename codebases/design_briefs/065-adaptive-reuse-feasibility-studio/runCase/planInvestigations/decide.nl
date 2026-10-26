---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Identify the surveys, specialist reviews, or evidence needed to distinguish feasible concepts. Prioritize investigations by their ability to change a decision, and specify how positive, negative, or inconclusive findings would alter the concept.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
