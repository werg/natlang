---
description: Join. Combine two inputs' rows on a condition, by hash join or nested loops.
args:
  left: Row[]
  right: Row[]
  on: string
  kind: "'inner' | 'left' | 'semi' | 'anti'"
  method: "'hash' | 'nested-loop'"
returns: Row[]
---
Combine left and right on on, an SQL condition over their field names. A joined row holds the fields of both rows.
- inner: each pair of a left and a right row that meets on.
- left: the same pairs, and also each left row with no match, once, with every right field null.
- semi: each left row that has at least one match, once.
- anti: each left row that has no match.

With method hash, on is one or more equalities between a left field and a right field. Build a map from each right
row's values of its fields to the right rows that have them. Probe the map with each left row's values; null never
matches. Check the rest of on, if any, on each pair found. With nested-loop, test every pair. Keep left's order, and
within one left row the right rows' order.
