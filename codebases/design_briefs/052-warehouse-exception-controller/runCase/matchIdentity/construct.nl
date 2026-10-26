---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftStockIdentityFinding
---
Construct a StockIdentityFinding draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Compare descriptions, packaging, serials, lot information, and movement history to identify candidate stock. Distinguish a mislabeled item from a misplaced one, and request a discriminating scan or inspection rather than choosing the closest description.
