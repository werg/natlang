---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftChangePackage
---
Construct a ChangePackage draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Prepare a coordinated scope, cost, timing, and responsibility package with the relevant evidence. Separate agreed changes from disputed claims, bind approvals to the exact revision, and update downstream plans only after the appropriate authorities accept the change.
