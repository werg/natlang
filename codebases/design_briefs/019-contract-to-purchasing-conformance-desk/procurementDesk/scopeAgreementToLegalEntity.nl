---
args: { request: ProcurementReviewRequest, intent: StageResultPurchaseIntent, terms: EvidencePage[], entityTerms: EvidencePage[] }
returns: StageResultAgreementScope
---
Match the exact purchasing legal entity, agreement and schedule, intended category/use, territory, effective dates, and amendments. Require an explicit affiliate schedule before applying a parent agreement to an affiliate. A catalog listing or shared portal cannot establish contract coverage.
