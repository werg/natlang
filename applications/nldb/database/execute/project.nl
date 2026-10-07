---
description: Projection. Compute each output column of every row; with distinct, drop repeated rows.
args:
  rows: Row[]
  columns: Output[]
  distinct: boolean
returns: Row[]
---
For each row, make a row holding each output's expression evaluated over the row's fields, under the output's name.
Use SQLite's semantics, in eval:
- arithmetic with null is null; integer division of integers truncates;
- || joins text; CASE, COALESCE, IFNULL, ROUND, ABS, LOWER, UPPER, LENGTH, SUBSTR and TRIM are available;
- date(x, modifier) and strftime work on YYYY-MM-DD text.
With distinct, keep only the first of rows whose values are all equal. Keep the rows' order.
