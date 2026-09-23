---
args:
  state: State
  event: UiEvent
returns: Decision
---
Prepare a repository migration in an isolated candidate. propose supplies exact old text in text and replacement secondary; reject missing or ambiguous contexts. check writes the candidate into an isolated repository, executes syntax and greeting behavior checks, and returns actual evidence. reset restores the original. Download exports the proposed source; do not claim it was merged into the user repository.

For an explicit control event, event.kind is the requested action and event.value
is a JSON record of its fields. Preserve those fields and do not substitute a
different action. For a command event, interpret the user's request using the
current state and the documented actions. Return the next applicable Decision. The caller owns sequencing and completion.
Keep edits and semantic judgments within the user's request. Host effects and
results belong to perform; never fabricate an execution result here.
