---
args: { request: SubscriptionExitRequest, records: SubscriptionCase[] }
returns: GoalContract
---
Match the requested case, customer, and exact subscription set. Distinguish cancellation of one product or add-on from closure of the account. Carry requested effective date, expected version, retained data scope, and unrequested linked products explicitly; any identity ambiguity blocks execution.
