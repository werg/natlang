---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftFabricationEstimate
---
Construct a FabricationEstimate draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Use exact costing services with documented quantities, rates, waste assumptions, and supplier quotes. Distinguish fixed price from allowance or estimate, expose unknowns, and prevent a creative design change from silently invalidating the cost basis.
