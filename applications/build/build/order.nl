---
description: Build planning, order. One valid order of the needed tasks, each after the tasks it depends on, and the tasks that wait on each other, if any.
args:
  graph: Graph
  goal: string
  needed: string[]
returns: Plan
---
Order needed, the closure of goal in graph, exactly in eval, level by level.

1. nodeOf maps each node's id to the node. placed = []. remaining = needed.
2. Repeat at most needed.length times, as a counted loop:
   - available = the ids in remaining whose node's deps that are in needed are all in placed, sorted by string order.
   - When available is empty, stop. Otherwise append available to placed, in that order, and remove it from remaining.
3. cycle = remaining sorted by string order. It is empty exactly when every needed task was placed.

Return { goal, needed, order: placed, cycle }.
