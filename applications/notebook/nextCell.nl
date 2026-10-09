---
description: Choose which ready notebook cell runs first.
args:
  ready: Cell[]
  goal: string
  problem?: string
returns: string
---
Choose one cell of ready to run next. Every cell in ready is required for goal and has all its dependencies computed, so any choice is valid.

1. Read the description of each cell in ready.
2. Prefer a cell whose description is about data preparation (loading, filtering, joining) over one that presents results.
3. Among equals, return the lowest id.
4. Return an id that appears in ready, written exactly as it appears there. When problem is given, it states what was wrong with the previous answer; return an id that fixes it.
