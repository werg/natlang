---
args: { request: GrantCycleRequest, submissions: EvidencePage[] }
returns: GrantCorrectionPlan
---
Bind a correction to the original submitted artifact, approval hash, funder portal receipt and changed claims. Return amendment only when the linked old submission and correction evidence are exact; otherwise leave it null and carry blockers. Preserve prior receipt references in oldReceiptRefs. Never replace or erase the original submission history.
