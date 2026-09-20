---
args:
  state: State
  event: UiEvent
returns: Decision
---
Edit and run natlang source. save stores exact source text for target filename. add creates a new filename from target and source from text. run uses text as JSON inputs for selected source; exact child execution returns a trace reference. inspect reads frame amount from that trace; frames are host-owned, not copied wholesale into model context. scenario adds a frozen input (text) and expected JSON value (secondary). evaluate_case executes the target scenario against current source; the natlang caller iterates the collection. Never report tests passed without child results. Free-form requests may propose an edit using save. Preserve frontmatter and types.

For an explicit control event, event.kind is the requested action and event.value
is a JSON record of its fields. Preserve those fields and do not substitute a
different action. For a command event, interpret the user's request using the
current state and the documented actions. Return the next applicable Decision. The caller owns sequencing and completion.
Keep edits and semantic judgments within the user's request. Host effects and
results belong to apply; never fabricate an execution result here.
