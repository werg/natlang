---
args: { request: BidPreparationRequest, receipts: EvidencePage[] }
returns: TenderGuardrail
---
For a timed-out upload, query submission status by tender identity and idempotency key, then bind any acknowledgment to the frozen response hash. Retry only when authoritative portal status proves no submission was accepted; uncertainty blocks duplicate upload.
