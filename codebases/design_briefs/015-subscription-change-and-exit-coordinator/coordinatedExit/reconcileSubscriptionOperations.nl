---
description: Reconcile immutable service mutation receipts before treating a subscription operation as complete or retryable.
args:{request:SubscriptionExitRequest;choices:ChangeOption[];receipts:EvidencePage[]}
returns:SubscriptionOperation|null
---
Match the operation receipt to exact option, subscription IDs, account version, idempotency identity and approval. A confirmed receipt yields the operation; a rejected receipt remains failed. A timeout or missing status remains unknown and requires authoritative readback; never blindly retry an irreversible change.
