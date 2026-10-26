---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftQualityExposure
---
Construct a QualityExposure draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Traverse exact lot, material, and process lineage to identify confirmed and precautionary exposure. Interpret process changes and rework histories, and avoid narrowing the scope merely because records are missing for some potentially affected units.
