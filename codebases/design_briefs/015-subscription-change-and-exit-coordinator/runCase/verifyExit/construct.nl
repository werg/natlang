---
args: { candidate: SubscriptionOperation, decisions: DecisionsExitReceipt, evidence: EvidencePage[] }
returns: DraftExitReceipt
---
1. Construct an evidence-backed ExitReceipt record from established atoms, using the fields declared for ExitReceipt in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Compare actual subscription states and upcoming invoices with the selected option. Explain surviving obligations and access explicitly, provide a durable receipt, and reopen inconsistencies rather than declaring success after merely submitting a cancellation request.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
