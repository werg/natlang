---
args: { candidate: ScopedClaim, decisions: DecisionsClaimConflict, evidence: EvidencePage[] }
returns: DraftClaimConflict
---
1. Construct an evidence-backed ClaimConflict record from established atoms, using the fields declared for ClaimConflict in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Generate candidates through shared entities and facets, then compare meanings under aligned scope and time. Classify coexistence as legitimate when claims describe different editions or populations; record the minimum incompatible propositions when a real conflict exists.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
