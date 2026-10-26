---
args: { candidate: PayableDisposition, decisions: DecisionsPostingReceipt, evidence: EvidencePage[] }
returns: DraftPostingReceipt
---
1. Construct an evidence-backed PostingReceipt record from established atoms, using the fields declared for PostingReceipt in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Compare approved allocations with actual ledger and processor results. Reconcile lost acknowledgments by operation identity, detect subsequent credit notes, and maintain remaining obligations without retrying a payment whose outcome is unknown.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
