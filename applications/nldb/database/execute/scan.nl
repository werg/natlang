---
description: Sequential scan. Read every row of a table from its pages, keeping those that meet where.
kind: directory-reducer
args:
  table: string
  alias: string
  where: string | null
returns: Row[]
---
Read table's pages in folder in order: tables/<table>/0.jsonl, 1.jsonl and so on, every page file there is. Each
non-empty line is one row as stored. Rename its fields alias.<field>, _id included. Keep the rows that meet where, an
SQL condition over those names, and keep them in page order. Change no file.

Evaluate where exactly in eval, with SQL's rules:
- a comparison with null is not true, and only IS NULL matches null;
- text compares by its characters, case-sensitively unless the condition lowers it;
- dates and timestamps compare as their text;
- LIKE matches with % and _, ignoring ASCII case.
