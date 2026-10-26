---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftTaxonomyImpact
---
Construct a TaxonomyImpact draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Reclassify only potentially affected records under baseline and candidate definitions, compare changes, and inspect boundary cases. Build explicit bridges for historical reporting and retain cases that cannot be mapped uniquely after a split.
