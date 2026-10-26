---
args: { request: SubscriptionExitRequest, entitlements: EvidencePage[] }
returns: ExitGuardrail
---
Check whether the subscription scope includes a shared workspace and identify owners whose access would change. Require authority for the affected workspace and preserve unaffected owners and access until an approved transition is acknowledged.
