---
args:
  ready: Task[]
  goal: string
  files?: Folder
returns: string
---
Choose one ID from the supplied ready tasks. If a task description names a
local input, read only that file from files before choosing. Prefer work needed
by the goal over unrelated work; among equally relevant tasks, prefer work
whose description says it creates a prerequisite, then use the
lexicographically smallest ID. Return only the chosen ID. The workspace checks
readiness again before executing it.
