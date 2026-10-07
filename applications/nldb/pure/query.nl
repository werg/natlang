---
description: Answer a question from a database folder, reading only.
kind: directory-reducer
args:
  question: string
  catalog: Catalog
returns: Answer
---
Answer question from the database in folder, whose schema is catalog, as a query engine would; do not change any
file. Plan first: which tables, which conditions, which joins (through foreign keys), which grouping, ordering and
limit. Read only what the plan needs, using an index file to find rows by a value when one fits. Compute counts,
sums, averages and comparisons exactly. Answer with the result's columns and rows (each row a list of values in
column order), how you found it, and the assumptions you made where the question allowed more than one reading. When
no rows match, the answer is empty rows, not a failure.
