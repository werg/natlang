---
args: { candidate: HarmonizedSlice, decisions: DecisionsHarmonizationAudit, evidence: EvidencePage[] }
returns: DraftHarmonizationAudit
---
1. Construct an evidence-backed HarmonizationAudit record from established atoms, using the fields declared for HarmonizationAudit in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Compare reasonable alternative mappings and cohort restrictions to reveal dependence on harmonization choices. Produce an analyst-facing report distinguishing stable findings from those driven by assumptions and identify measurements that must remain separate.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
