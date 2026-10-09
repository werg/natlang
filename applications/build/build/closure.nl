---
description: Build planning, closure. The tasks a goal needs - the goal and everything it depends on, directly or through others.
args:
  graph: Graph
  goal: string
returns: string[]
---
Compute the closure of goal over graph exactly in eval.

1. nodeOf maps each node's id to the node. reached = [goal]. frontier = [goal].
2. Repeat at most graph.nodes.length times, as a counted loop:
   - next = the ids that appear in the deps of a frontier id's node, are not in reached, and have a node in nodeOf,
     without duplicates.
   - When next is empty, stop. Otherwise add next to reached and set frontier to next.
3. Return reached sorted by string order.
