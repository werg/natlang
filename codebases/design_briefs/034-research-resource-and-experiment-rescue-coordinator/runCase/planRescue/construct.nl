---
args: { candidate: ExperimentSlot, decisions: DecisionsRescueOption, evidence: EvidencePage[] }
returns: DraftRescueOption
---
1. Construct an evidence-backed RescueOption record from established atoms, using the fields declared for RescueOption in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: When a resource or sample becomes unavailable, generate alternatives such as resequencing, qualified substitution, or postponement. Explicitly state scientific deviations and seek investigator approval where the original protocol would no longer hold.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
