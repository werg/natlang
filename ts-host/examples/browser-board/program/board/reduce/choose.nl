---
args:
  state: Board
  event: UiEvent
returns: Decision
---
Interpret the incoming UI event. A command can add a task, mark an existing
task done, reopen it, or remove it. A toggle event names an exact task ID and
switches its done status. A clear_done event removes completed tasks. Return
one decision with kind add, toggle, remove, clear_done, or ignore; use item_id
for a specific task and text for an added task. Keep the task text concise.
Do not invent an item ID. The exact apply helper checks all decisions.
