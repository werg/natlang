---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftWarehouseFinding
---
Construct a WarehouseFinding draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Interpret the discrepancy against confirmed identity and movement evidence. Separate damage from quantity shortage and pending receipt from true absence; identify which inventory and fulfillment commitments may be affected without broad unsupported holds.
