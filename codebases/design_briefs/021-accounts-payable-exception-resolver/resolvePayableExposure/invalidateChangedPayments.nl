---
args: {"versions": "LiabilityVersion[]", "ledger": "PayableBook", "documents": "EvidencePage[]"}
returns: "PaymentInstruction[]"
---
For each outstanding instruction compare documentVersion, credit identities, amount basis and posting period to the latest authoritative evidence. Invalidate only affected unissued instructions and approvals. If already sent or unknown, preserve the operation ID and request processor reconciliation before issuing an offset or replacement. A newly closed period changes posting authority, not historical service dates. Never cancel an external payment by merely changing a local status.
