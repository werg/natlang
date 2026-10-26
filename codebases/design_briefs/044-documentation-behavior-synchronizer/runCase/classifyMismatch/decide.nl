---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Reconcile requirements, release decisions, tests, and observed behavior to determine which artifact is wrong. Do not automatically rewrite documentation to match a bug; preserve unresolved intent questions for the responsible product or engineering owner.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
