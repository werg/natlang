---
args: { request: ProcurementReviewRequest, commitments: EvidencePage[] }
returns: ProcurementGuardrail
---
Group related requests by requester, intended use, supplier, shared project and purchase window when evidence links them. Sum cumulative commitment for authority thresholds; preserve separate records and flag uncertain linkage rather than silently aggregating unrelated purchases.
