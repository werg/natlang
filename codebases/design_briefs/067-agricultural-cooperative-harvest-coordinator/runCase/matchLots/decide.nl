---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Compare lot characteristics and readiness with customer requirements, identifying compatible combinations and gaps. Preserve uncertain quality findings and request targeted verification rather than allocating a lot solely because its crop name matches.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
