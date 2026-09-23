---
args:
  state: State
  event: UiEvent
returns: Decision
---
Author a structured document. save edits target section heading secondary and body text. title changes document title using text. add adds a section. reorder supplies every section ID exactly once in ids. publish marks a reviewed revision ready for local HTML and Markdown export; any subsequent edit invalidates it. Do not fabricate citations or claim external publication.

For an explicit control event, event.kind is the requested action and event.value
is a JSON record of its fields. Preserve those fields and do not substitute a
different action. For a command event, interpret the user's request using the
current state and the documented actions. Return the next applicable Decision. The caller owns sequencing and completion.
Keep edits and semantic judgments within the user's request. Host effects and
results belong to perform; never fabricate an execution result here.
