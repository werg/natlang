---
args: { candidate: ExperimentReview, decisions: DecisionsExperimentDesign, evidence: EvidencePage[] }
returns: DraftExperimentDesign
---
1. Construct an evidence-backed ExperimentDesign record from established atoms, using the fields declared for ExperimentDesign in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Read the original plan and subsequent amendments in temporal order. Distinguish preregistered criteria from retrospective choices, identify the assignment and observation units, and preserve deviations that could change interpretation even when the experiment ran successfully.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
