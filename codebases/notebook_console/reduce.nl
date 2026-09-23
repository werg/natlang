---
description: Interpret a notebook terminal request, execute its dependency graph, and retain the explained run.
args:
  state: ConsoleState
  event: ConsoleEvent
  files?: Record<string, File>
returns: ConsoleState
---
function reduce(state, event, files) -> ConsoleState
  if event.kind is not "request":
    return state
  cells = catalog()
  goal = choose_goal(event.value, cells)
  result = run_notebook(goal, event.value, files)
  return append(state, event.value, result)
