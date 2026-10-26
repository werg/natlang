---
args: { candidate: GrantAllocation, decisions: DecisionsActivityEvidence, evidence: EvidencePage[] }
returns: DraftActivityEvidence
---
1. Construct an evidence-backed ActivityEvidence record from established atoms, using the fields declared for ActivityEvidence in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Compare project records with milestone definitions and acceptance evidence. Distinguish activity completion from demonstrated impact, preserve partial delivery, and request missing evidence rather than embellishing a narrative to fit the grant’s aspirations.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
