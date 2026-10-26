---
args: { request: SubscriptionExitRequest, billing: EvidencePage[] }
returns: ExitGuardrail
---
Match each refund to its original charge and operation ID. Keep pending, reversed, failed, and settled states distinct from subscription state; a closed subscription does not discharge a refund obligation or authorize a second refund.
