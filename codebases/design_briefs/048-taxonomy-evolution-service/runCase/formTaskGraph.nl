---
args: { goal: GoalContract }
returns: WorkGraph
---
Create named work nodes with prerequisites and blocked questions. Each node must produce a distinct artifact and have its required inputs.
Return the typed result, preserving missing facts as missing and citing evidence IDs where a decision depends on evidence.
