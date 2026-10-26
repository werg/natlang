---
args: { candidate: PurposeAllocation, decisions: DecisionsExpenseFinding, evidence: EvidencePage[] }
returns: DraftExpenseFinding
---
1. Construct an evidence-backed ExpenseFinding record from established atoms, using the fields declared for ExpenseFinding in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Apply the policy edition effective for the expense and claimant context. Preserve category-specific limits, meal inclusions, and approval conditions; distinguish a documented exception from a persuasive but unauthorized justification.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
