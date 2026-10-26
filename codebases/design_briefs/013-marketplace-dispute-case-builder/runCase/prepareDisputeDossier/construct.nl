---
args: { candidate: DisputeRecommendation, decisions: DecisionsReviewDossier, evidence: EvidencePage[] }
returns: DraftReviewDossier
---
1. Construct an evidence-backed ReviewDossier record from established atoms, using the fields declared for ReviewDossier in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Assemble a balanced dossier with evidence, disputed questions, reasoning summary, and each party’s response opportunity. Redact unnecessary personal information while preserving contestable evidence and record the reviewer’s actual disposition separately from the recommendation.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
