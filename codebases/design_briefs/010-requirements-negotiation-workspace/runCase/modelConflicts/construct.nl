---
args: { candidate: StakeholderPosition, decisions: DecisionsRequirementConflict, evidence: EvidencePage[] }
returns: DraftRequirementConflict
---
1. Construct an evidence-backed RequirementConflict record from established atoms, using the fields declared for RequirementConflict in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Identify requirements that cannot jointly hold under the supplied technical, time, or resource constraints. Distinguish true contradictions from terminology differences and propose the smallest clarification or relaxation that would make the pair compatible.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
