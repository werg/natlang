---
description: Compile a natural-language database request to SQLite.
args:
  request: string
  schema: string
  today: string
returns: SqlPlan
---
Compile request for the SQLite database whose tables schema shows (their CREATE statements); today is the date.
Decide its kind first: a question changes nothing and becomes one SELECT; a change becomes INSERT, UPDATE and DELETE
statements; a schema request becomes CREATE TABLE / ALTER TABLE / CREATE INDEX statements designed as a careful
database designer would (narrow types, primary and foreign keys, NOT NULL, UNIQUE and CHECK constraints, indexes on
searched columns; dates as TEXT in ISO form); an unclear request gets the one question that would settle it and no
statements. All statements of a request run as one transaction, so write them so that together they do all of it.

When a condition is about meaning rather than an exact value ("complaints about late delivery", "friendly reviews"),
write it as meets(column, 'criterion') and list it under semantic with its table and column; the database judges
each stored value once. Explain in a sentence what the statements do, and list assumptions you made.
