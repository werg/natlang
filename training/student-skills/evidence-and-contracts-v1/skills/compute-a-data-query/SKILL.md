---
name: compute-a-data-query
description: Use for an exact count, sum, ranking or filtered answer over available structured records, especially across pages.
---

Turn each condition in the question into a filter. Apply all filters before aggregating. Count the requested unit, group by the requested key, and use the stated tie rule. Do not confuse a maximum over all groups with a maximum inside a specified group. Use eval for the computation, inspect its result, then return that actual value in the declared type. Do not return the code as a string or guess from a partial preview. Exact inputs require exact computation.
