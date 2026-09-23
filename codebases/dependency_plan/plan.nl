---
args:
  tasks: Task[]
returns: State
description: Plan a dependency graph with a semantic priority choice; expose
  cycles and missing dependencies.
---
function plan(tasks) -> State
  initial = prepare(tasks)
  if initial.finished:
    return initial
  else:
    return step.iterateOn(initial).withLimit({ maxSteps: 16 }).until(finished)
