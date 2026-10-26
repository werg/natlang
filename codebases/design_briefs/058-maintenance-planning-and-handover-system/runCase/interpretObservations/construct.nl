---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftAssetObservation
---
Construct a AssetObservation draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Reconcile inspection notes, service history, and operator reports to identify observed conditions and unresolved discrepancies. Separate symptoms from diagnosis, retain intermittent behavior, and identify information needed for routing under approved maintenance procedures.
