---
args: { candidate: CompatibilityFinding, decisions: DecisionsQualificationTask, evidence: EvidencePage[] }
returns: DraftQualificationTask
---
1. Construct an evidence-backed QualificationTask record from established atoms, using the fields declared for QualificationTask in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Design the smallest supplier clarification, sample inspection, or approved test that resolves a decisive unknown. Specify pass criteria before results arrive and distinguish documentary clarification from testing requiring technical specialists.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
