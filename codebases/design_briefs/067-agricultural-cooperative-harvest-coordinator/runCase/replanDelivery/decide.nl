---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Process changed field observations, resource failures, or customer responses as scoped deltas. Reallocate only affected lots, obtain consent for material substitutions, and reconcile actual harvested and delivered quantities before declaring commitments fulfilled.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
