---
args: { request: ProcurementReviewRequest, purchases: PurchaseReview[], evidence: EvidencePage[] }
returns: StageResultPurchaseIntent
---
Recover the requested purpose, line identity, quantity, entity, and essential function from the supported purchasing record. Separate brand preference from functional need. Ask only for missing details that can alter agreement coverage, scope, quantity, or approval authority.
