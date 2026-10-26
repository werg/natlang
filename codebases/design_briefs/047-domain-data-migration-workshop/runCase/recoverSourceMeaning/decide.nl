---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Reconcile dictionaries, sample records, application notes, and domain-owner explanations. Identify fields whose meaning depends on record type or history, and distinguish null, not applicable, unknown, and deliberately withheld values before proposing transformations.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
