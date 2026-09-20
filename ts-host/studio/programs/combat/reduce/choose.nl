---
args:
  state: State
  event: UiEvent
returns: Decision
---
Resolve simultaneous combat decisions. round sets player move in text and rival move in secondary, one of strike, guard, recover. Select a rival move semantically from the public state; never claim to observe a hidden player intent beyond supplied UI controls. Exact host physics handles energy, damage and simultaneous resolution. reset starts a new match. No attacks after the match ends.

For an explicit control event, event.kind is the requested action and event.value
is a JSON record of its fields. Preserve those fields and do not substitute a
different action. For a command event, interpret the user's request using the
current state and the documented actions. Return the next applicable Decision. The caller owns sequencing and completion.
Keep edits and semantic judgments within the user's request. Host effects and
results belong to apply; never fabricate an execution result here.
