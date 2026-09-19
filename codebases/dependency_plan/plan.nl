---
args:
  tasks: Task[]
returns: State
types:
  Task: '{ id: Text, needs: Text[], description: Text }'
  State: '{ tasks: Task[], order: Text[], blocked: Text[], finished: Bool }'
description: Plan a dependency graph with a semantic priority choice; expose cycles
  and missing dependencies.
---
function plan(tasks) -> State
  initial = prepare(tasks)
  if initial.finished:
    return initial
  else:
    return repeat step(initial), until finished(state), at most 16 rounds
