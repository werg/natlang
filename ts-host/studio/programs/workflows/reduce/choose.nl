---
args:
  state: State
  event: UiEvent
returns: Decision
---
Explore a workflow against an explicit simulated provider. execute advances target step after its dependencies succeed, writing an idempotent receipt keyed by step key. fail simulates a known failure. unknown marks an ambiguous provider outcome; reconcile supplies text succeeded or failed as a simulated provider lookup. compensate reverses a succeeded step only after its succeeded dependents have been compensated. Never retry unknown outcomes blindly. This app does not charge cards or contact real providers.

For an explicit control event, event.kind is the requested action and event.value
is a JSON record of its fields. Preserve those fields and do not substitute a
different action. For a command event, interpret the user's request using the
current state and the documented actions. Return the next applicable Decision. The caller owns sequencing and completion.
Keep edits and semantic judgments within the user's request. Host effects and
results belong to apply; never fabricate an execution result here.
