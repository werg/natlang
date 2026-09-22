---
args:
  ready: Cell[]
  goal: Text
  files?: Dict<File>
returns: Text
---
Choose one offered ready cell ID whose declared result helps the goal. If the goal names a supporting file, inspect that exact args/files leaf first. Prefer
needed data preparation before presentation work; break ties by ID. Never
invent a cell or run one whose dependencies are not complete.
