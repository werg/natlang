---
args: { candidate: ActivityEvidence, decisions: DecisionsGrantNarrative, evidence: EvidencePage[] }
returns: DraftGrantNarrative
---
1. Construct an evidence-backed GrantNarrative record from established atoms, using the fields declared for GrantNarrative in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Draft a coherent account aligned with actual spending, activities, and verified results. Explain deviations and unresolved conditions honestly, tailor the format to funder requirements, and cross-check narrative quantities against exact financial schedules.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
