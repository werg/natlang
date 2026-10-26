---
args: { candidate: PayableDiscrepancy, decisions: DecisionsAllocationProposal, evidence: EvidencePage[] }
returns: DraftAllocationProposal
---
1. Construct an evidence-backed AllocationProposal record from established atoms, using the fields declared for AllocationProposal in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Construct a partial resolution for portions supported by delivery and authorization evidence. Preserve disputed balances, prevent the same receipt from supporting multiple payments, and request targeted confirmation for unresolved acceptance or scope.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
