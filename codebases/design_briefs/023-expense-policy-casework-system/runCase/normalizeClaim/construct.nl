---
args: { candidate: ExpenseCase, decisions: DecisionsExpenseLine, evidence: EvidencePage[] }
returns: DraftExpenseLine
---
1. Construct an evidence-backed ExpenseLine record from established atoms, using the fields declared for ExpenseLine in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Extract receipt lines and claimant explanations while preserving original currency and dates. Distinguish the merchant’s description from the claimed business purpose, and flag illegible or missing evidence rather than reconstructing amounts through guesswork.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
