---
args:
  ready: Task[]
  goal: string
  files?: Record<string, File>
returns: string
---
Choose one ID from the supplied ready tasks. If a task description names a local input, inspect only that files leaf before choosing. Prefer work needed by the goal over
unrelated work; among equally relevant tasks, prefer work whose description says
it creates a prerequisite, then use the lexicographically smallest ID. Return
only the chosen ID. The host checks readiness again before executing it.
