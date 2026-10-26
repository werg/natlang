---
args: { candidate: PagePatch, decisions: DecisionsPublicationCheck, evidence: EvidencePage[] }
returns: DraftPublicationCheck
---
1. Construct an evidence-backed PublicationCheck record from established atoms, using the fields declared for PublicationCheck in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Re-extract claims from the proposed page set and repeat comparison around changed topics. Inspect whether simplification removed exceptions; build a review packet with residual disagreements and publish only after required owners approve the same patch set.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
