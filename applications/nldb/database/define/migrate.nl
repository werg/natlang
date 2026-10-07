---
description: Rewrite stored rows and index files for one schema change.
kind: directory-reducer
args:
  change: SchemaChange
returns: number
---
Make the stored data in folder match change. catalog.json still describes the schema before the change. Rewrite only
the files the change concerns, and keep every page's lines in _id order.
- add column: give every row of the table the new field, with change's default, placed in column order.
- drop column: remove the field from every row, and delete its index file if there is one.
- rename column: rename the field in every row, and rename indexes/<table>/<column>.json to the new name.
- rename table: move tables/<table>/ and indexes/<table>/ to the new name.
- drop table: delete its pages and index files. Fail when a column of another table references it.
- create index: write indexes/<table>/<column>.json with a [value, _id] pair for every row, sorted by value (null
  first, then numbers, then text), then by _id.
- drop index: delete the index file.
- constrain column: check every stored row against the column's new constraints (not null, unique, check, and the
  referenced value exists). Fail naming the first rows that break one.
Return how many rows you rewrote or checked.
