---
args:
  state: State
  event: UiEvent
returns: Decision
---
This is a semantic, notional CRDT laboratory without crisp convergence rules. configure sets family from target, base from text, left from secondary. right sets right replica text. propose semantically merges base, left and right into text with unresolved issues in ids. commit adopts the proposal as the new common base, recording both replica inputs and result. Preserve facts, account for both edits, and flag ambiguity. Families: document, set, map, ordered-list, calendar, task-board, tree, graph, inventory, conversation. Do not claim convergence from one run.

For an explicit control event, event.kind is the requested action and event.value
is a JSON record of its fields. Preserve those fields and do not substitute a
different action. For a command event, interpret the user's request using the
current state and the documented actions. Return the next applicable Decision. The caller owns sequencing and completion.
Keep edits and semantic judgments within the user's request. Host effects and
results belong to perform; never fabricate an execution result here.
