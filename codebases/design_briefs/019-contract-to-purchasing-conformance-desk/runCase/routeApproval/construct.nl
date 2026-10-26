---
args: { candidate: PurchasingOption, decisions: DecisionsPurchaseDisposition, evidence: EvidencePage[] }
returns: DraftPurchaseDisposition
---
1. Construct an evidence-backed PurchaseDisposition record from established atoms, using the fields declared for PurchaseDisposition in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Build a review packet with scope evidence, pricing, alternatives, and unresolved conditions. Route to the correct authority based on cumulative commitment and exception type; recheck terms and request versions immediately before order release.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
