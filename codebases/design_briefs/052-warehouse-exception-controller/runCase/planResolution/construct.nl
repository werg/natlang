---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftWarehouseWorkOrder
---
Construct a WarehouseWorkOrder draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Create bounded search, recount, relabel, inspection, or transfer tasks using approved warehouse procedures. Sequence dependencies and require independent confirmation for consequential inventory adjustments or uncertain identity changes.
