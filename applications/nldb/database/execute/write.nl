---
description: Page writer. Store one table's row changes in its pages, as a storage engine's heap writer does, and keep the catalog's counts current.
kind: directory-reducer
args:
  table: string
  changes: RowChange[]
returns: RowChange[]
---
Apply changes to table's pages in folder (tables/<table>/<n>.jsonl). catalog.json's entry for table has its
columns, nextId, rows and pages. Return the changes as stored.
- Insert (no before). The row gets the table's nextId as _id, and nextId goes up by one. It has a field for every
  column of the table, null where it has no value. It goes at the end of the last page, or on a new page once the
  last holds 256 rows.
- Update (before and after). The after row replaces the line whose _id is before's _id, in place, keeping that _id.
- Delete (no after). The line whose _id is before's _id is removed. A page left without rows stays as an empty file.
A row whose _id no page holds is an error: fail and name it.

Write each changed page once. Every line is one row as compact JSON: _id first, then the columns in the table's
order. Then set the table's rows (how many rows it holds), pages (how many page files) and nextId in catalog.json.
Return changes in the order given, each insert's after row with its _id.
