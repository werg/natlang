---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Reconcile order terms, customer messages, and product constraints to identify what must be preserved. Separate contractual deadlines from preferences, recognize dependencies such as installation appointments, and avoid ranking customers by unstated commercial assumptions.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
