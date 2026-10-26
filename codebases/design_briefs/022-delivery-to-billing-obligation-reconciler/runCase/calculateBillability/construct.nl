---
args: { candidate: AcceptanceFinding, decisions: DecisionsBillableAllocation, evidence: EvidencePage[] }
returns: DraftBillableAllocation
---
1. Construct an evidence-backed BillableAllocation record from established atoms, using the fields declared for BillableAllocation in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Submit established milestone states and exact schedules to the billing calculator. Account for prior invoices, credits, caps, and partial acceptance; explain remaining holds without inventing a commercial compromise not authorized by the contract.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
