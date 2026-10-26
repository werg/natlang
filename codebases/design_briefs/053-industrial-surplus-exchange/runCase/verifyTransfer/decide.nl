---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Coordinate approved transfers and verify quantity, condition, ownership, and recipient acceptance at handoff. Record rejected or partially accepted material and update future matching evidence without falsely reporting a completed reuse benefit.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
