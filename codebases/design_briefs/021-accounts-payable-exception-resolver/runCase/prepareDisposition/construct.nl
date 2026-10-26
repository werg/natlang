---
args: { candidate: AllocationProposal, decisions: DecisionsPayableDisposition, evidence: EvidencePage[] }
returns: DraftPayableDisposition
---
1. Construct an evidence-backed PayableDisposition record from established atoms, using the fields declared for PayableDisposition in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Prepare an auditable recommendation to pay, hold, request correction, or apply a verified credit. Route by entity and delegated thresholds, and explain why undisputed portions may proceed without treating the whole invoice as resolved.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
