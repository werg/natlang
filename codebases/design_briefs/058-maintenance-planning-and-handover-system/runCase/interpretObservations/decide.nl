---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Reconcile inspection notes, service history, and operator reports to identify observed conditions and unresolved discrepancies. Separate symptoms from diagnosis, retain intermittent behavior, and identify information needed for routing under approved maintenance procedures.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
