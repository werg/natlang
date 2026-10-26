---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftStockObservation
---
Construct a StockObservation draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Reconcile manifests, scans, picking notes, and receiving observations while preserving who observed what and when. Separate an item’s physical presence from its expected location and retain uncertainty when labels or serials are missing.
