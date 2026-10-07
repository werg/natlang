---
description: Choose how to run a statement, as a cost-based query optimizer does. Picks access paths, join order and methods, and where each filter goes.
args:
  statement: Statement
  catalog: Catalog
returns: Statement
---
Rewrite every plan in statement into an equivalent plan that reads and judges less. That is a question's plan, and
each change's target or from plan. Return statement with the new plans and nothing else changed. catalog gives each
table's row count and its indexed columns: these are your statistics.

Rewrites, in this order:
1. Push each exact condition down to the earliest step whose rows have all its columns. A condition on the inner side
   of a left, semi or anti join may move into that side's own steps only when the join's result stays the same.
2. Access paths. A scan whose where compares an indexed column with constants (=, IN), or bounds it (<, <=, >, >=,
   BETWEEN), becomes a lookup on that column with those values or that range. The rest of where stays as the
   lookup's where. When several columns qualify, choose an equality before a range; then choose the column that
   should match the fewest rows.
3. Join methods. A join whose on is one or more equalities between a column of each side uses method hash, with the
   side that should have fewer rows as left. Any other join uses nested-loop.
4. Join order. With three or more inputs joined by inner joins, join first the pair whose result should be smallest,
   by row counts and the selectivity of their conditions. Keep left, semi and anti joins with their sides as written.
5. Semantic conditions cost a model judgment per distinct value. Move every filter with semantic conditions after all
   exact filters and joins that remove rows. Split a filter into its exact part, kept early, and its semantic part,
   placed late.
6. Drop steps that do nothing: a filter with no condition, or a project that keeps every column as it is.

Renumber inputs to match the new order of steps. Each plan's last step must yield the same rows with the same columns
as before.
