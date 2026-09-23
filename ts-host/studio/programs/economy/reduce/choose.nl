---
args:
  state: State
  event: UiEvent
returns: Decision
---
Operate a small closed economy. buy moves amount apples from target seller to secondary buyer, transferring cash at seller price. price sets target merchant offer to amount. When asked to simulate an actor, inspect stocks/cash/offers and choose one sensible trade or price adjustment; explain via the view. Cash and apples must be conserved. No invented production, credit, or successful trades.

For an explicit control event, event.kind is the requested action and event.value
is a JSON record of its fields. Preserve those fields and do not substitute a
different action. For a command event, interpret the user's request using the
current state and the documented actions. Return the next applicable Decision. The caller owns sequencing and completion.
Keep edits and semantic judgments within the user's request. Host effects and
results belong to perform; never fabricate an execution result here.
