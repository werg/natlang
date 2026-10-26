---
args: { candidate: EvidenceClaim, decisions: DecisionsEvidenceUpdate, evidence: EvidencePage[] }
returns: DraftEvidenceUpdate
---
1. Construct an evidence-backed EvidenceUpdate record from established atoms, using the fields declared for EvidenceUpdate in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Process new reports, corrections, and retractions as scoped changes. Re-evaluate linked studies and dependent claims, preserve prior review snapshots, and request expert review where the new evidence changes methodological interpretation rather than merely adding another citation.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
