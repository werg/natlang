---
name: broken-scope
description: Scope points at a missing file and has a bad type.
natlang:
  scope:
    table:
      type: "{ a: number"
      value: 1
    rows:
      type: number[]
      file: data/rows.json
---
Body.
