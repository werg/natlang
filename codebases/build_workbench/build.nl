---
description: Build a declared dependency graph, choosing each ready task by its purpose and retaining exact execution evidence.
args:
  goal: Text
  tasks: Task[]
returns: State
---
function build(goal, tasks) -> State
  initial = prepare(goal, tasks)
  if finished(initial):
    return initial
  else:
    return repeat step(initial), until finished(state), at most one round per declared task
