---
description: Carry out a schema request on the database in folder. Designs the change, migrates stored rows and updates the catalog.
kind: directory-reducer
args:
  request: string
  catalog: Catalog
returns: Report
---
Change the schema of the database in folder as request asks; catalog is its current catalog.json.

design(request, catalog) gives the schema changes in order. Carry them out in that order, each seeing the earlier
ones:
- create table: add the table to catalog.json, with pages 1, rows 0 and nextId 1. Create an empty
  tables/<table>/0.jsonl, and an index file holding [] for each of its indexed columns.
- Any other change to a table that has rows: folder.apply(migrate, change) rewrites its stored data first. Then update
  catalog.json to match: the table, its columns and their constraints, its key and its indexes. A renamed table or
  column is renamed in every references of other tables too.
Then raise catalog.json's version by one.

Return the DDL as statements, no row changes, design's assumptions, and a one-sentence summary. When migrate fails, fail
with its reason: nothing is kept.
