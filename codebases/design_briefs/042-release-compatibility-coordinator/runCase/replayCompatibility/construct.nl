---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftCompatibilityTest
---
Construct a CompatibilityTest draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Execute representative consumer scenarios against baseline and candidate in isolated environments. Compare intended outcomes rather than raw output equality alone, and identify whether a difference is acceptable, breaking, or dependent on an unresolved contract.
