---
args: { request: AmendmentReviewRequest, receipts: EvidencePage[] }
returns: AmendmentGuardrail
---
At cutover, recheck current effective scope for any task not yet executed. A pre-cutover approval does not authorize post-cutover execution where amendment changed that clause; preserve authorization history and request refreshed approval where needed.
