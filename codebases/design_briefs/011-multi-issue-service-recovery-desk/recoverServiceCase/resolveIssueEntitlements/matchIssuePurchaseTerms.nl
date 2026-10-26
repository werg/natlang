---
description: Bind purchase and exception evidence to the exact issue scope.
args:{issues:ServiceIssue[];purchaseRecords:EvidencePage[];exceptions:EvidencePage[]}
returns:EntitlementReview[]
---
Match each issue's order and purchase date to current policy evidence. Verify exception customer, issue, amount, authority and expiry tuple. Preserve conflicting editions and out-of-scope exceptions as unresolved. Do not infer entitlement from a later policy or an unrelated order.
