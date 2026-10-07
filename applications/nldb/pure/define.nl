---
description: Carry out a schema request against a database folder.
kind: directory-reducer
args:
  request: string
  catalog: Catalog
returns: Report
---
Change the schema of the database in folder as request asks; catalog is its current catalog.json. Design what a
careful database designer would: one table per kind of thing, a column per fact with the narrowest fitting type,
keys, foreign keys to the tables things refer to, unique, not-null and check constraints the request implies, and
indexes on keys, foreign keys and columns the request says will be searched. Write the new catalog.json (version
plus one). For a new table create tables/<table>/0.jsonl, empty. When a change affects existing rows (a new
column, a rename, a dropped column or table), rewrite every page and index so they match the new schema; give a
new column its stated default or null, and fail when a new constraint does not hold for rows already stored.

Return the schema statements you carried out (as SQL DDL) and a one-sentence summary.
