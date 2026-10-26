---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Rewrite steps for the local context using approved terminology and resources. Preserve sequence, guards, stop conditions, and verification; represent genuine deviations explicitly rather than hiding them in friendlier wording or inferred substitutions.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
