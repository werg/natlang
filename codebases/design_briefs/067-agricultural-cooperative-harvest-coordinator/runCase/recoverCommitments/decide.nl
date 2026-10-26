---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Reconcile delivery agreements and customer preferences to identify quantity, quality, and timing requirements. Separate permitted substitutions from assumptions and retain dependencies such as processing or storage windows that constrain otherwise feasible harvest choices.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
