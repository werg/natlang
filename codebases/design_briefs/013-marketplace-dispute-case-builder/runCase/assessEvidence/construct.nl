---
args: { candidate: DisputedQuestion, decisions: DecisionsEvidenceFinding, evidence: EvidencePage[] }
returns: DraftEvidenceFinding
---
1. Construct an evidence-backed EvidenceFinding record from established atoms, using the fields declared for EvidenceFinding in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Evaluate relevance, consistency, and provenance of each submitted item against a specific question. Distinguish evidence authenticity checks from semantic support, and discount multiple copies of the same record without dismissing independent corroboration.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
