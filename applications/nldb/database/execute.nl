---
description: Run a planned statement on the database in folder, as an executor does. Runs the plan's operators in order; for a change, it then writes pages, maintains indexes and checks constraints.
kind: directory-reducer
args:
  statement: Statement
returns: Execution
---
Run statement on the database in folder, with the operators in your folder.

Running a plan. Run its steps in order and keep each step's rows; a step's inputs are earlier steps, by position. Each
op is the function of the same name:
- scan(folder, table, alias, where);
- lookup(folder, table, alias, column, values, range, where);
- filter(rows of input, where, semantic);
- join(rows of left, rows of right, on, kind, method);
- aggregate(rows of input, groupBy, aggregates);
- project(rows of input, columns, distinct);
- sort(rows of input, by, limit, offset).
Steps that do not depend on each other may run at the same time. A plan's result is its last step's rows.

A question. Run its plan. Return rows: each result row as a list of values in the order of statement's columns. A
column named x is the row's field x, or else its only field ending in .x. Return no changes.

A change. Carry out its changes in order, each seeing what the earlier ones did. For each, make the row changes:
- insert: one with no before for each new row: its rows, or the rows of from's plan with their alias. prefixes
  dropped.
- update: run target. For each target row, before is that row's alias. fields with the prefix dropped. after is
  before with every set column replaced by its expression, computed exactly in eval from the target row.
- delete: run target. For each target row, before is its alias. fields with the prefix dropped, and there is no
  after.
Then folder.apply(write, table, those row changes) stores them and returns them as stored, with inserted rows' _id.
Pass those to folder.apply(reindex, table, them). Count the inserted, updated and deleted rows per table.

After the last change, call check(folder, table, row changes) for each table the transaction changed, with all of
its row changes as stored. If any check finds violations, fail with them; then nothing of the transaction is kept.
Return the counts as changes, and no rows.
