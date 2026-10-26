---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Assemble a canonical view without overwriting source records or selecting one source globally for every field. Resolve attributes using field-specific authority and effective time, preserve aliases and conflicting claims, and separate historical from current values.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
