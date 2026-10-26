---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftInventoryResolution
---
Construct a InventoryResolution draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Compare completed tasks with actual scans and custody events, apply approved exact adjustments, and verify conservation across locations. Preserve unresolved differences and distinguish a found item from a fulfilled customer order.
