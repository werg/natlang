---
args: { candidate: TriggerFinding, decisions: DecisionsObligationTask, evidence: EvidencePage[] }
returns: DraftObligationTask
---
1. Construct an evidence-backed ObligationTask record from established atoms, using the fields declared for ObligationTask in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Turn active obligations into bounded tasks with ownership, dependencies, and exact computed dates. Preserve legal interpretation separately from operational scheduling and identify impossible or conflicting task requirements before issuing commitments.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
