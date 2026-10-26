---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Arrange consented collection or delivery, verify availability before dispatch, and track actual recipient acceptance. Reallocate declined or unavailable items through the same policy rather than informal side agreements, and preserve privacy in donor communications.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
