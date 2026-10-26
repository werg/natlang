---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Apply the exact simulator’s legal transition rules, then formulate context-appropriate responses from actors who know only their own information. Preserve consequences across turns and do not reveal hidden facts merely to help the learner succeed.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
