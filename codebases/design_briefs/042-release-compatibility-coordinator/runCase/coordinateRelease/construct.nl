---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftReleaseReadiness
---
Construct a ReleaseReadiness draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Reconcile code, documentation, support guidance, and customer notices into one release state. Explain audience-specific consequences and verify that approved migrations and communications match the exact release artifact being promoted.
