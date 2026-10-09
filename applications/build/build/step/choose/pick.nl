---
description: Build scheduling, choice (natural-language implementation). Choose one task ID from the ready tasks - work the goal needs first, prerequisite work before the rest, then the smallest ID.
args:
  ready: Task[]
  goal: string
  files?: Folder
returns: string
---
Choose one ID from ready, the tasks whose dependencies are all finished. The workspace checks readiness again before
it runs the choice.

1. When a task's description names a local input file, read only that file from files before choosing; it can say
   how urgent or how large the task is. Without files, go by the descriptions.
2. Keep the tasks that the goal needs more than the others: a task whose description says it produces something
   goal's description or id asks for.
3. Among the tasks left, keep those whose description says they create a prerequisite (something other tasks wait
   for).
4. Among the tasks left, take the lexicographically smallest ID.

Return that ID alone.
