---
args: { candidate: ExpenseFinding, decisions: DecisionsExpenseQuestion, evidence: EvidencePage[] }
returns: DraftExpenseQuestion
---
1. Construct an evidence-backed ExpenseQuestion record from established atoms, using the fields declared for ExpenseQuestion in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Group unresolved questions to minimize claimant burden while keeping each decision testable. Ask only for evidence necessary to determine eligibility, explain the applicable rule, and route privacy-sensitive material through restricted upload channels.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
