---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Read member requests and prior agreements to identify desired uses, availability, and explicitly stated access needs. Separate preferences from accepted communal obligations and avoid inferring personal circumstances or assigning work without consent.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
