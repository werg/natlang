---
description: Sort and limit. Order rows by expressions, then skip offset rows and keep at most limit.
args:
  rows: Row[]
  by: "{ expression: string, descending: boolean }[]"
  limit: number | null
  offset: number
returns: Row[]
---
Order rows by the values of the by expressions over each row's fields: the first expression decides, then the next.
Ascending puts null first; descending puts it last, as SQLite does. Compare numbers as numbers and text by its
characters. Rows that tie keep their order. Then skip the first offset rows and keep limit of the rest (all when limit
is null).
