---
description: Plan the compensation of an order that is being undone, in order.
args:
  state: WorkflowState
returns: Compensation
---
List the undos still to do for the order in state, from state.history. An effect stands when the history has an entry
of that action with status "done", and no entry of its undo with status "done". The undo of charge is refund; the undo
of reserve is release.
1. When the history has an entry of action ship with status "done", the goods left: return { steps: [] }.
2. steps starts empty. When charge stands, push { action: "refund", reason: a sentence saying the charge is being
   returned }.
3. When reserve stands, push { action: "release", reason: a sentence saying the reserved stock is being returned }.
   The refund comes before the release.
4. Return { steps }. It is empty when nothing stands.
