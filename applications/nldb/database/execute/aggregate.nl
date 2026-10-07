---
description: Aggregate. Group rows and compute each group's aggregates exactly.
args:
  rows: Row[]
  groupBy: string[]
  aggregates: Output[]
returns: Row[]
---
Group rows by the values of the groupBy expressions. With no groupBy, all rows form one group, even when there are
no rows; then count is 0 and the other aggregates are null. Give each group one row, in order of first appearance:
each groupBy expression's value under the expression's own text, and each aggregate's value under its name.

Compute exactly, in eval:
- count(*) counts rows; count(x) counts non-null values; count(DISTINCT x) counts distinct non-null values.
- sum, min and max skip nulls and are null when nothing is left. avg is the mean of the non-null values, a real
  number.
- An aggregate may sit inside an expression ("sum(o.qty * o.price)", "round(avg(x), 2)"). Sums of decimals are
  rounded to the most decimals among their values, so floating-point noise does not show.
