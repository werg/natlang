---
args:
  state: State
  event: UiEvent
returns: Decision
---
Manage a local wiki. save writes title in secondary and body in text to selected page. add creates page target/title secondary/body text. select changes page. incoming stores a collaborator draft. merge must semantically combine the current body and incoming draft, return merged text and unresolved issues as ids. There is no crisp convergence rule: use the pinned model, seed, source and ordered inputs. run executes selected page cell with JSON text inputs. save_cell updates cell. Do not pretend local pages are synchronized over a network.

For an explicit control event, event.kind is the requested action and event.value
is a JSON record of its fields. Preserve those fields and do not substitute a
different action. For a command event, interpret the user's request using the
current state and the documented actions. Return the next applicable Decision. The caller owns sequencing and completion.
Keep edits and semantic judgments within the user's request. Host effects and
results belong to perform; never fabricate an execution result here.
