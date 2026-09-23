---
args:
  state: State
  event: UiEvent
returns: Decision
---
Reconcile schemas. input validates JSON records text. map sets source column text to target column secondary, rejecting duplicate targets. remove deletes mapping target ID. transform applies the complete mapping exactly and records missing fields rather than silently discarding them. Use semantic knowledge to propose mappings in response to a command; never silently merge distinct people or invent missing values.

For an explicit control event, event.kind is the requested action and event.value
is a JSON record of its fields. Preserve those fields and do not substitute a
different action. For a command event, interpret the user's request using the
current state and the documented actions. Return the next applicable Decision. The caller owns sequencing and completion.
Keep edits and semantic judgments within the user's request. Host effects and
results belong to perform; never fabricate an execution result here.
