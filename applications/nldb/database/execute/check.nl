---
description: Constraint checker. Check one table's row changes against the schema's constraints, on the database as the transaction left it.
kind: directory-reducer
args:
  table: string
  changes: RowChange[]
returns: string[]
---
Check table's changed rows against catalog.json and return every violation, one sentence each. Name the table, the
row (its _id and key) and the constraint broken. No violations is an empty list. folder holds the database as the
transaction has left it. Change no file.

For each after row:
- types: each value is null or has its column's type (integer, real, text, boolean, date as YYYY-MM-DD, timestamp as
  ISO 8601, neuralese as a block ID);
- not null: a column that is not nullable has a value;
- check: the value meets its column's check, a condition in words such as "at least 0"; null passes;
- unique: no other row of the table has the same value in a unique column, or the same values in all of the key's
  columns when the key is not ["_id"]. Use the column's index when it has one; otherwise read the pages;
- foreign keys: a value in a column that references other.column matches some row's other.column.

For each before row that was deleted, or whose value changed in a column that other tables reference: no row of a
table whose column references this table's column may still hold the old value. Find such tables among
catalog.json's columns.
