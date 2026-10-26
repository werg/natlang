---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Read the request and prior attempts to separate observations from the requester’s diagnosis. Identify approved handling restrictions and information needed for safe routing, without inventing technical causes or giving unqualified hazardous repair instructions.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
