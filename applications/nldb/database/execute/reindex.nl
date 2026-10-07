---
description: Index maintenance. Keep one table's index files in step with its row changes.
kind: directory-reducer
args:
  table: string
  changes: RowChange[]
returns: number
---
For each indexed column of table (catalog.json's indexes for it), update indexes/<table>/<column>.json in folder. The
file is a JSON array of [value, _id] pairs sorted by value, then _id. For each change:
- remove the pair [before's value, _id] when there is a before;
- add [after's value, _id] when there is an after.
A change that leaves the column's value as it was leaves its pair alone. Null values are indexed too.

Keep the array sorted with null first, then numbers by value, then text by its characters; ties go by _id. Find
where to remove and insert by binary search. Write each file you changed once. Return how many pairs you removed
and added in all.
