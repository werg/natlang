---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftShippingCommitment
---
Construct a ShippingCommitment draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Reconcile order terms, customer messages, and product constraints to identify what must be preserved. Separate contractual deadlines from preferences, recognize dependencies such as installation appointments, and avoid ranking customers by unstated commercial assumptions.
