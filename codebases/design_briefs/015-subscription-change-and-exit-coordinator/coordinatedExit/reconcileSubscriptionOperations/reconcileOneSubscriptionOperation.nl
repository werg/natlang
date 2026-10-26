---
args:{request:SubscriptionExitRequest;choice:ChangeOption;receipts:EvidencePage[]}
returns:SubscriptionOperation|null
---
Match receipts to this exact selected option, customer account version, subscription set, effective date and idempotency key. Return an operation only when its authenticated authority and service identity match. A rejected operation remains failed; a lost response remains unknown and needs readback. Never infer success from a draft or from a receipt for another option.
