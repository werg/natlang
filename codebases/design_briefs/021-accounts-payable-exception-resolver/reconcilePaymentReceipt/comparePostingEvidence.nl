---
args: {"instruction": "PaymentInstruction", "observation": "PostingObservation"}
returns: "string[]"
---
Compare operation identity, legal liability, document version, amount, currency and actual posting IDs. A processor acknowledgment is not necessarily a ledger posting; a network timeout is not proof of failure. Reject mismatched or reused receipt IDs. Return every unresolved discrepancy; an empty list is allowed only when the configured authoritative observations establish the claimed terminal state exactly.
