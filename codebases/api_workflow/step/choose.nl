---
args:
  current: WorkflowState
  item: WorkflowEvent
returns: Decision
---
Choose exactly one next action from reserve, charge, ship, refund, release,
reconcile or wait. When pending is nonempty, choose reconcile for a reconciliation
event and otherwise wait. Never repeat an uncertain charge. For a new order
reserve inventory; after reserve charge; after charge ship. On a definite
shipping failure or cancellation after payment, refund, then release inventory.
On charge failure, release inventory. A completed shipment waits. Use the
history and obligations, and explain the choice briefly. The host checks
prerequisites and durable receipt identity.
