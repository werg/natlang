---
args: { candidate: RequirementConflict, decisions: DecisionsCompromiseOption, evidence: EvidencePage[] }
returns: DraftCompromiseOption
---
1. Construct an evidence-backed CompromiseOption record from established atoms, using the fields declared for CompromiseOption in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Invent alternatives such as phased delivery, differentiated modes, or reduced scope while preserving mandatory constraints. Explicitly identify who gives up what and which new assumptions the compromise introduces; include a no-agreement option.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
