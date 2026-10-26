---
args: { request: SubscriptionExitRequest, mutations: EvidencePage[] }
returns: ExitGuardrail
---
Bind consent to the exact option, subscription IDs, effective date, and account version. If the customer changes any of those before execution, mark the prior consent token stale and require a fresh affirmative consent receipt; never apply the old option.
