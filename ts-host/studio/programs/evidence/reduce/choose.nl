---
args:
  state: State
  event: UiEvent
returns: Decision
---
Build evidence-backed notes. add stores passage target/title secondary/text. search performs literal ranked retrieval with text query. claim adds an interpretation text, source passage target and literal quote secondary; the host rejects fabricated quotes. Interpret evidence carefully and state unknowns. remove deletes a claim by target. A matching quotation establishes provenance, not semantic entailment.

For an explicit control event, event.kind is the requested action and event.value
is a JSON record of its fields. Preserve those fields and do not substitute a
different action. For a command event, interpret the user's request using the
current state and the documented actions. Return the next applicable Decision. The caller owns sequencing and completion.
Keep edits and semantic judgments within the user's request. Host effects and
results belong to apply; never fabricate an execution result here.
