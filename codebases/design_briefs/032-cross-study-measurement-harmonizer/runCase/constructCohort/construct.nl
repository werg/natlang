---
args: { candidate: HarmonizationRule, decisions: DecisionsHarmonizedSlice, evidence: EvidencePage[] }
returns: DraftHarmonizedSlice
---
1. Construct an evidence-backed HarmonizedSlice record from established atoms, using the fields declared for HarmonizedSlice in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Define comparable cohorts before materialization, apply approved transformations in a sandbox, and reconcile row counts and exclusions. Keep missingness reasons and study membership available rather than flattening them into one generic null category.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
