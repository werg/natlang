---
args: { request: NegotiationPackageRequest; packages: ClausePackageSet; redlines: RedlineReviewSet; rendering: NegotiationGuardrail; approval: NegotiationGuardrail }
returns: ClauseOfferPlan
---
Select only a complete package whose external rendering contains no internal comments and whose approval evidence binds the current exact draft hash. Keep internal fallback notes out of the external body while preserving their protected history. If any material condition, undefined term, approval mismatch or prohibited concession remains, return no offer with precise blockers. A prepared offer remains uncirculated and cannot imply counterparty acceptance.