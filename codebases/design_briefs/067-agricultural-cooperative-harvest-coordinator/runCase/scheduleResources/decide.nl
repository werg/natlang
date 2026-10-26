---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Use exact scheduling and capacity services to coordinate labor, equipment, harvest windows, storage, and collection. Construct contingency plans for uncertain readiness and avoid assigning the same resource to overlapping lots or ignoring transport bottlenecks.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
