---
args: { candidate: MeasureDefinition, decisions: DecisionsMeasureRelation, evidence: EvidencePage[] }
returns: DraftMeasureRelation
---
1. Construct an evidence-backed MeasureRelation record from established atoms, using the fields declared for MeasureRelation in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Compare constructs, scales, recall periods, and populations against the intended analysis. Separate exact unit conversion from inferential harmonization, and identify subsets where comparability holds without claiming it for the entire dataset.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
