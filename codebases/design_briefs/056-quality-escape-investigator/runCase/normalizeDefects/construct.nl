---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftDefectObservation
---
Construct a DefectObservation draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Reconcile customer reports, operator notes, and inspection descriptions into comparable observed defects. Preserve measurement conditions and terminology differences, distinguish symptoms from suspected causes, and retain reports that may describe a different mechanism.
