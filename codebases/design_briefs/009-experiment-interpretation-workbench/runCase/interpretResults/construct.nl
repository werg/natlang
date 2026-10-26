---
args: { candidate: MeasurementFinding, decisions: DecisionsResultClaim, evidence: EvidencePage[] }
returns: DraftResultClaim
---
1. Construct an evidence-backed ResultClaim record from established atoms, using the fields declared for ResultClaim in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Consume approved statistical outputs rather than calculating inference through prose. Relate results to the specified estimand, inspect multiplicity and subgroup selection, and distinguish evidence of little effect from an underpowered or compromised experiment.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
