---
description: formTaskGraph for Participatory Budgeting Facilitator.
args: { goal: GoalContract; stages: string[] }
returns: WorkGraph
---
Construct a finite dependency graph. Each node must have available inputs, a distinct artifact, an acceptance gate, and a stopping condition; return missing prerequisites as blocked questions.
