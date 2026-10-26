---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Partition input and state space by behaviorally meaningful distinctions, including invalid inputs and interacting requirements. Identify boundary cases and previously untested combinations rather than generating many superficial variants of the same happy path.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
