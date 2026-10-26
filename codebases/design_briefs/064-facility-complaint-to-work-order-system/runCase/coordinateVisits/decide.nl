---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Schedule visits using exact availability and resource constraints, obtain required access consent, and coordinate multiple trades. Preserve cancellations and failed-access events, and update occupants with truthful windows rather than unsupported precise arrival promises.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
