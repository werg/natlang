---
args: { candidate: ExposureFinding, decisions: DecisionsContainmentPlan, evidence: EvidencePage[] }
returns: DraftContainmentPlan
---
1. Construct an evidence-backed ContainmentPlan record from established atoms, using the fields declared for ContainmentPlan in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Select approved containment actions proportionate to the established issue and uncertainty. Coordinate holds, inspections, alternative supply, and reviewed communications while preserving unaffected orders and avoiding promises before capacity is reserved.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
