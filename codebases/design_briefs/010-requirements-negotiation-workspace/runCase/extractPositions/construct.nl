---
args: { candidate: RequirementsProject, decisions: DecisionsStakeholderPosition, evidence: EvidencePage[] }
returns: DraftStakeholderPosition
---
1. Construct an evidence-backed StakeholderPosition record from established atoms, using the fields declared for StakeholderPosition in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Read each request with its rationale and authority. Separate desired outcomes, proposed solutions, non-negotiable constraints, and preferences; do not upgrade emphatic language into a mandatory requirement without an explicit basis.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
