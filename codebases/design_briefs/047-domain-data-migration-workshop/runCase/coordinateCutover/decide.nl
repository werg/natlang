---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Freeze the approved mapping and cutover scope, apply final source deltas, and reconcile target results before switching consumers. Track quarantined records and post-cutover corrections explicitly, retaining a safe rollback or forward-repair plan for irreversible changes.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
