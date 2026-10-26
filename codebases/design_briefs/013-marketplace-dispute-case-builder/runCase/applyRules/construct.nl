---
args: { candidate: EvidenceFinding, decisions: DecisionsDisputeRecommendation, evidence: EvidencePage[] }
returns: DraftDisputeRecommendation
---
1. Construct an evidence-backed DisputeRecommendation record from established atoms, using the fields declared for DisputeRecommendation in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Apply supplied marketplace rules to established facts and explicitly conditional alternatives for unresolved facts. Explain how the outcome changes under plausible resolutions and route cases outside delegated interpretation to authorized reviewers.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
