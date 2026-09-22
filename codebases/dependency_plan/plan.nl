import { finished } from "./plan/finished";
import { prepare } from "./plan/prepare";
import { step } from "./plan/step";
---
args:
  tasks: Task[]
returns: State
types:
  Task: '{ id: string, needs: string[], description: string }'
  State: '{ tasks: Task[], order: string[], blocked: string[], finished: boolean }'
description: Plan a dependency graph with a semantic priority choice; expose cycles
  and missing dependencies.
---
function plan(tasks) -> State
  initial = prepare(tasks)
  if initial.finished:
    return initial
  else:
    return repeat step(initial), until finished(state), at most 16 rounds
