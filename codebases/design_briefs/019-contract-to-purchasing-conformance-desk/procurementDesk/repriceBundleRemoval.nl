---
args: { request: ProcurementReviewRequest, schedules: EvidencePage[] }
returns: ProcurementGuardrail
---
When a line is removed from a discounted bundle, recalculate price using the applicable schedule edition and remaining quantities. Preserve original bundle price, remove/reprice scenario, and any minimum quantity condition; stale unit pricing cannot be reused.
