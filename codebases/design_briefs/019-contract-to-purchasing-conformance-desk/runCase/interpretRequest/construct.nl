---
args: { candidate: PurchaseReview, decisions: DecisionsPurchaseIntent, evidence: EvidencePage[] }
returns: DraftPurchaseIntent
---
1. Construct an evidence-backed PurchaseIntent record from established atoms, using the fields declared for PurchaseIntent in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Read item descriptions with the requester’s purpose and supporting documents. Distinguish a legitimate functional need from a preferred brand or category label, and identify ambiguity that changes contract coverage or approval authority.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
