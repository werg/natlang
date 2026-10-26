---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Choose approved inspection or repair task types and required trade qualifications. Sequence dependencies, include access arrangements and occupant impact, and route safety-sensitive or uncertain cases to qualified personnel rather than generating improvised repairs.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
