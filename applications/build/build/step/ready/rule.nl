---
description: Build scheduling, ready set (natural-language implementation). The needed tasks that are not finished and whose dependencies are all finished.
args:
  graph: Graph
  needed: string[]
  finished: string[]
returns: string[]
---
Find the ready tasks exactly in eval.

1. nodeOf maps each node's id to the node of graph.
2. A task id in needed is ready when it is not in finished and every id in its node's deps is in finished.
3. Return the ready ids sorted by string order. When no task is ready, return [].
