---
args: { candidate: QualificationTask, decisions: DecisionsSubstitutionPacket, evidence: EvidencePage[] }
returns: DraftSubstitutionPacket
---
1. Construct an evidence-backed SubstitutionPacket record from established atoms, using the fields declared for SubstitutionPacket in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Compare qualified alternatives on verified cost, lead time, and operational consequences. Present the recommended substitution with adaptations and residual uncertainty, obtain technical and commercial approval, and record which uses the approval does not cover.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
