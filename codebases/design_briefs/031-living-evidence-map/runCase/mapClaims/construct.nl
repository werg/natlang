---
args: { candidate: StudyObservation, decisions: DecisionsEvidenceClaim, evidence: EvidencePage[] }
returns: DraftEvidenceClaim
---
1. Construct an evidence-backed EvidenceClaim record from established atoms, using the fields declared for EvidenceClaim in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Group evidence by the actual proposition and relevant population rather than broad topic. Identify incompatible measurements and contradictory findings, qualify applicability, and expose evidence gaps without treating absent studies as proof of no effect.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
