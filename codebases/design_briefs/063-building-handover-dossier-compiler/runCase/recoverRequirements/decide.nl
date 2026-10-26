---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Read the supplied handover specification, asset schedules, and contractual requirements to identify required documents, tests, training, and approvals. Separate documentary requirements from technical sign-off and preserve conditions specific to asset classes or building areas.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
