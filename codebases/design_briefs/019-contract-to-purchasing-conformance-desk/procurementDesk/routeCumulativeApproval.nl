---
args: { request: ProcurementReviewRequest, options: StageResultPurchasingOption, commitments: ProcurementGuardrail, agreement: StageResultAgreementScope, currentTerms: ProcurementGuardrail, policy: PolicySelection }
returns: StageResultPurchaseDisposition
---
Aggregate related commitments before choosing an approver so splitting one purchase cannot evade a threshold. Route by cumulative commitment and exception type. Before order release, require current agreement, price, entity, and catalog revision; expired terms keep the disposition blocked until refreshed.
