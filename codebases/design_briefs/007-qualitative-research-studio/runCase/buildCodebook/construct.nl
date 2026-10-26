---
args: { candidate: SamplingPlan, decisions: DecisionsCodeDefinition, evidence: EvidencePage[] }
returns: DraftCodeDefinition
---
1. Construct an evidence-backed CodeDefinition record from established atoms, using the fields declared for CodeDefinition in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Draft codes from pilot material and research concepts, specifying boundaries through positive and negative examples. Separate descriptive labels from inferred explanations; reconcile overlapping codes without forcing genuinely multidimensional evidence into one category.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
