---
args: { candidate: ExceptionRecord, decisions: DecisionsPrecedentCandidate, evidence: EvidencePage[] }
returns: DraftPrecedentCandidate
---
1. Construct an evidence-backed PrecedentCandidate record from established atoms, using the fields declared for PrecedentCandidate in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Search by the rationale and relevant circumstances rather than matching requested outcomes. Construct analogies and disanalogies, including expired and rejected cases when they explain the boundary of acceptable treatment.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
