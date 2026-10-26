---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftCategoryFit
---
Construct a CategoryFit draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Compare incoming records with category definitions and examples, not names alone. Distinguish a genuinely new concept from a poorly worded existing category, and retain ambiguous membership where the taxonomy permits multiple dimensions.
