---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Reconcile plot records, field observations, and supplied gardening guidance to describe known conditions and constraints. Preserve uncertainty about soil or other untested properties, and route consequential interventions to appropriate local expertise.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
