---
args: { request: SubscriptionExitRequest, entitlements: EvidencePage[] }
returns: ExitGuardrail
---
For each requested product, identify the contract manager and cancellation authority. If a reseller owns the entitlement, specify the separate reseller action and preserve it as pending until a matching acknowledgment is received; do not represent a local request as full cancellation.
