---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Materialize a staged target snapshot, validate referential integrity and business invariants, and compare representative source-target round trips. Rehearse rollback and delta capture so cutover does not lose changes made after the initial extraction.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
