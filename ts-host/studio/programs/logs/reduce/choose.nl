---
args:
  state: State
  event: UiEvent
returns: Decision
---
Investigate logs with evidence. ingest parses text as JSON array of {id,time,level,message}; repeated identical IDs are ignored, conflicting duplicate IDs rejected. incident adds title text, severity target info/warning/critical, evidence IDs ids. resolve marks incident target resolved. escalate writes a LOCAL escalation receipt for target incident, idempotently. No message is sent to an external service. Distinguish correlation from a proven cause.

For an explicit control event, event.kind is the requested action and event.value
is a JSON record of its fields. Preserve those fields and do not substitute a
different action. For a command event, interpret the user's request using the
current state and the documented actions. Return the next applicable Decision. The caller owns sequencing and completion.
Keep edits and semantic judgments within the user's request. Host effects and
results belong to apply; never fabricate an execution result here.
