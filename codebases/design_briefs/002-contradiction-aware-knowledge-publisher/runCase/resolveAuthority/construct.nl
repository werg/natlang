---
args: { candidate: ClaimConflict, decisions: DecisionsAuthorityDecision, evidence: EvidencePage[] }
returns: DraftAuthorityDecision
---
1. Construct an evidence-backed AuthorityDecision record from established atoms, using the fields declared for AuthorityDecision in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Consult the configured publication hierarchy and explicit supersession records. Prefer authoritative material only inside its remit; if equally authoritative sources disagree, formulate a bounded clarification rather than selecting the newest or most fluent text.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
