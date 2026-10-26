---
args: { candidate: StudyIdentity, decisions: DecisionsStudyObservation, evidence: EvidencePage[] }
returns: DraftStudyObservation
---
1. Construct an evidence-backed StudyObservation record from established atoms, using the fields declared for StudyObservation in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Apply eligibility to available evidence, escalating insufficient abstracts to full-text review. Extract outcomes, methods, and limitations with source locations; keep author interpretation separate from directly reported observations and exact numerical results.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
