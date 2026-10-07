---
description: Design the schema changes a request asks for, as a careful database designer would.
args:
  request: string
  catalog: Catalog
returns: Design
---
Read request against catalog, the current schema, and design the changes it asks for:
- one table per kind of thing, with a sentence saying what one row stands for;
- a column per fact, with the narrowest type that fits and a description that gives its unit;
- a key: the natural key the request implies, else ["_id"];
- foreign keys to the tables things refer to;
- the not-null, unique and check constraints the request implies ("can never go negative" is a check "at least 0");
- indexes on key columns, foreign keys and the columns the request says will be searched.
Reuse what catalog has rather than duplicate it, and follow its naming; plural snake_case table names when it has
none. Give the changes in the order they can be carried out, and the DDL each amounts to. Say in assumptions what you
chose where the request left it open.
