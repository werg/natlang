---
description: Read a database request as a statement, as a SQL front end parses and analyzes a query, but from words. Gives the SQL it amounts to and a first, unoptimized plan.
args:
  request: string
  kind: "'question' | 'change'"
  catalog: Catalog
  today: string
returns: Statement
---
Read request, a kind, against catalog. today is the date for relative times.

Analysis. Write the request as SQL over catalog's tables (sql): one SELECT for a question; for a change, the INSERT,
UPDATE and DELETE statements it amounts to, in order. Resolve every name and value the request uses to a table and
column (a person's name in customers.name, a town in customers.city, "her account" through the foreign key that
links them). When words could mean several columns or values, choose the reading the request supports and say so in
assumptions. Write constants in their column's type: dates as YYYY-MM-DD, money as a number in the column's unit.
A condition on meaning that no exact comparison captures ("vegetarian dishes", "complaints about delivery") stays a
semantic condition: the column it applies to and the criterion in words. When the request needs a table or column
the catalog does not have, fail and name it.

Translation. Write each SELECT as plan steps, one operator each, in this order, without choosing access paths (the
optimizer does that):
- a scan for every table the query reads, under its alias, with the exact conditions that use that table alone;
- a join for every join condition, method nested-loop: inner for a plain join, left for LEFT JOIN, semi for EXISTS
  or IN (subquery), anti for NOT EXISTS or NOT IN (subquery);
- a filter for conditions that use several tables, and for every semantic condition;
- an aggregate for GROUP BY and aggregates (one group when there is no GROUP BY), and a filter after it for HAVING;
- a project for the SELECT list (with DISTINCT when it says so), and a sort for ORDER BY, LIMIT and OFFSET.
Name columns alias.column in every expression; after an aggregate or project, by their output names.

For a question, columns are the output names of the last step, in order, and explanation says in a sentence or two
how the answer is found. For a change, each INSERT is an insert with its rows (by plain column name, without _id) or
from a plan whose last step yields the new rows' columns; each UPDATE or DELETE is an update or delete whose target
plan yields the rows to change, read under alias, and an update's set gives every changed column's new value as an
expression over the target row ("a.balance - 50").
