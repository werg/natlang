---
args:
  state: State
  event: UiEvent
returns: Decision
---
Build a declared-input artifact through the real BuildWorkspace cache. save updates source text and transformation target copy or uppercase. build requests execution keyed by content, operation and output identity. Repeated identical builds reuse and verify cached bytes. If a build fails inspect its error before proposing a source repair; do not report successful checks without receipts.

For an explicit control event, event.kind is the requested action and event.value
is a JSON record of its fields. Preserve those fields and do not substitute a
different action. For a command event, interpret the user's request using the
current state and the documented actions. Return the next applicable Decision. The caller owns sequencing and completion.
Keep edits and semantic judgments within the user's request. Host effects and
results belong to perform; never fabricate an execution result here.
