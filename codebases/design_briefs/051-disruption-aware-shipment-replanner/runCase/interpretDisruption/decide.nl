---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Read carrier notices and operational reports to determine what is unavailable, for how long, and with what certainty. Distinguish a delayed service from a closed route and preserve exceptions that may allow some shipment classes to continue.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
