---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Reconcile surveys, field notes, and historical records into a baseline with explicit spatial, temporal, and methodological limits. Distinguish observed absence from unsurveyed areas and preserve rare or uncertain observations for specialist review.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
