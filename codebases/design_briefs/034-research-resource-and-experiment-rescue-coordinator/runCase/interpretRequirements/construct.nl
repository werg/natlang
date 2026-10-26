---
args: { candidate: ResearchSchedule, decisions: DecisionsExperimentRequirement, evidence: EvidencePage[] }
returns: DraftExperimentRequirement
---
1. Construct an evidence-backed ExperimentRequirement record from established atoms, using the fields declared for ExperimentRequirement in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Read approved protocols and researcher requests to distinguish essential experimental conditions from convenient preferences. Identify sample timing, equipment capabilities, and required personnel without changing scientific procedures or inferring unapproved substitutions.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
