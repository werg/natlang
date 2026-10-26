---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Sequence approved bookings and communications, reconcile carrier acknowledgments, and track actual shipment progress against revised promises. Replan only affected legs when new events arrive and explain residual uncertainty rather than claiming recovery after booking alone.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
