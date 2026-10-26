---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Reconcile measurements, invoices, contractor statements, and field records with their actual methods and periods. Preserve estimates versus direct measurements, identify missing coverage, and retain original units and uncertainty for exact conversion services.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
