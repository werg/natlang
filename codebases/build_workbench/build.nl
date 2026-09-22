import { finished } from "./build/finished";
import { prepare } from "./build/prepare";
import { step } from "./build/step";
---
description: Build a declared dependency graph, choosing each ready task by its purpose and retaining exact execution evidence. Use files to inspect declared workspace inputs before choosing a task.
args:
  goal: Text
  tasks: Task[]
  files?: Dict<File>
returns: State
---
function build(goal, tasks, files) -> State
  initial = prepare(goal, tasks)
  if finished(initial):
    return initial
  else:
    return repeat step(initial, files), until finished(state), at most one round per declared task
