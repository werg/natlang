---
args: { candidate: ExpenseQuestion, decisions: DecisionsExpenseDisposition, evidence: EvidencePage[] }
returns: DraftExpenseDisposition
---
1. Construct an evidence-backed ExpenseDisposition record from established atoms, using the fields declared for ExpenseDisposition in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Produce line-level recommendations and a clear explanation of reductions or holds. Support independent review and appeals with new evidence, then reconcile the approved amount to the reimbursement service without changing the historical original claim.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
