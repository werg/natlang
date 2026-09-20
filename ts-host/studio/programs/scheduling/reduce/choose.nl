---
args:
  state: State
  event: UiEvent
returns: Decision
---
Plan tasks in explicit UTC. add creates task title text with duration amount minutes. schedule sets target task start from text ISO datetime; exact code rejects overlaps. complete toggles completion. unschedule clears start. date changes displayed UTC date. Plan one task at a time in response to priorities, respecting duration and existing commitments. Do not imply calendar provider synchronization.

For an explicit control event, event.kind is the requested action and event.value
is a JSON record of its fields. Preserve those fields and do not substitute a
different action. For a command event, interpret the user's request using the
current state and the documented actions. Return the next applicable Decision. The caller owns sequencing and completion.
Keep edits and semantic judgments within the user's request. Host effects and
results belong to apply; never fabricate an execution result here.
