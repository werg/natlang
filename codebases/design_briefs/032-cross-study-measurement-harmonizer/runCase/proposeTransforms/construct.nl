---
args: { candidate: MeasureRelation, decisions: DecisionsHarmonizationRule, evidence: EvidencePage[] }
returns: DraftHarmonizationRule
---
1. Construct an evidence-backed HarmonizationRule record from established atoms, using the fields declared for HarmonizationRule in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Select documented transformations or category mappings and describe lost distinctions. Delegate arithmetic to exact services, refuse unsupported imputation, and preserve source variables so analysts can inspect or reverse the transformation where possible.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
