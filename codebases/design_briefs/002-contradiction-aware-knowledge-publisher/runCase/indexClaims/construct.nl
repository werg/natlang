---
args: { candidate: KnowledgeRevision, decisions: DecisionsScopedClaim, evidence: EvidencePage[] }
returns: DraftScopedClaim
---
1. Construct an evidence-backed ScopedClaim record from established atoms, using the fields declared for ScopedClaim in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Extract independently checkable claims with qualifications and page context. Resolve entity references conservatively, attach author and effective period, and keep instructions separate from descriptions of historical behavior or quoted third-party views.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
