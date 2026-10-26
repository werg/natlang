---
args: { candidate: ResourceCapability, decisions: DecisionsExperimentSlot, evidence: EvidencePage[] }
returns: DraftExperimentSlot
---
1. Construct an evidence-backed ExperimentSlot record from established atoms, using the fields declared for ExperimentSlot in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Submit hard resource and timing constraints to an exact scheduler, then evaluate feasible schedules against supplied priorities and scientific dependencies. Keep optional preferences visible and explain unavoidable conflicts instead of silently dropping experiments.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
