---
args: { candidate: AuthorityDecision, decisions: DecisionsPagePatch, evidence: EvidencePage[] }
returns: DraftPagePatch
---
1. Construct an evidence-backed PagePatch record from established atoms, using the fields declared for PagePatch in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Draft the smallest coordinated changes that resolve the selected conflicts. Preserve unrelated claims and local audience framing; update examples, cross-links, and summaries that would otherwise repeat the corrected mistake.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
