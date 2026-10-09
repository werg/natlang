---
description: Choose the next forward step of an order that is moving along normally.
args:
  state: WorkflowState
returns: Decision
---
Choose the next forward step by state.phase:
- new: { action: "reserve", reason: "a new order reserves its stock" }.
- reserved: { action: "charge", reason: "the stock is reserved, so the customer is charged" }.
- charged: { action: "ship", reason: "the payment is done, so the order ships" }.
- any other phase: { action: "wait", reason: "the order is <phase>" } with the phase named.
Return the Decision.
