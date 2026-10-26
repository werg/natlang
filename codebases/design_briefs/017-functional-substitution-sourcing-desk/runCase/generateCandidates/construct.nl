---
args: { candidate: FunctionalRequirement, decisions: DecisionsSubstituteCandidate, evidence: EvidencePage[] }
returns: DraftSubstituteCandidate
---
1. Construct an evidence-backed SubstituteCandidate record from established atoms, using the fields declared for SubstituteCandidate in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Search catalogs and supplier descriptions for equivalent functions, including alternative mechanisms where permitted. Explain how each candidate would serve the use, identify required adaptations, and avoid treating a familiar name or matching dimension as sufficient.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
