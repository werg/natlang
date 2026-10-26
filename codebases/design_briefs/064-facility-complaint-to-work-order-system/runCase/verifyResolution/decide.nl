---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Compare technician completion evidence and subsequent occupant observations with the original complaint. Distinguish a replaced component from resolution of the symptom, detect recurrence, and reopen only the relevant issue when work does not achieve the intended outcome.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
