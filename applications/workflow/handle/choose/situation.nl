---
description: Decide where an order stands, which decides how the workflow proceeds.
readout: decision
args:
  state: WorkflowState
  event: WorkflowEvent
returns: Situation
---
Decide where the order in state stands, given event. Take the first of these that holds.
uncertain: state.pending is not empty, so an operation was started and its outcome is unknown.
settled: event.kind is "reconcile" (and nothing is pending), or state.phase is shipped or released and no obligation
  is open.
cancelling: event.kind is "cancel".
owed: state.obligations is not empty (a compensation failed and is still owed), or state.phase is refunded (the
  reserved stock is still to be released).
failed: the last entry of state.history has status "failed".
forward: none of the above; state.phase is new, reserved or charged and the order moves on.
settled: otherwise.
