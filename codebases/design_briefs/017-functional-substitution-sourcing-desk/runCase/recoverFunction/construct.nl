---
args: { candidate: SubstitutionRequest, decisions: DecisionsFunctionalRequirement, evidence: EvidencePage[] }
returns: DraftFunctionalRequirement
---
1. Construct an evidence-backed FunctionalRequirement record from established atoms, using the fields declared for FunctionalRequirement in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Interpret why the original item is used, including interfaces, operating environment, and failure consequences. Separate critical properties from historical purchasing habits, and ask the requester to confirm distinctions that determine compatibility or safety.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
