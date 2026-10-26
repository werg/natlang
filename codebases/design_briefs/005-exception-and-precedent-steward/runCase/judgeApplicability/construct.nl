---
args: { candidate: PrecedentCandidate, decisions: DecisionsApplicabilityFinding, evidence: EvidencePage[] }
returns: DraftApplicabilityFinding
---
1. Construct an evidence-backed ApplicabilityFinding record from established atoms, using the fields declared for ApplicabilityFinding in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Test every limiting condition against current evidence and identify which differences change the rationale. If a decisive fact is unknown, request that fact; do not grant the exception by treating missing evidence as a match.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
