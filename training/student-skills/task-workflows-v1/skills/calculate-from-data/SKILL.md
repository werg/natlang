---
name: calculate-from-data
description: Use for exact arithmetic, counts, sorting or aggregation over supplied data. Compute the result in eval.
---

Use eval to calculate from the actual bound inputs or retrieved records. Read the declared input by its existing name; do not recreate it from memory. For paged inputs, inspect the page API and cover every page. Keep intermediate values in scope. Return the computed value with its declared type, not an expression string or an estimated number. If the task requests a record, compute and verify its keys too. Semantic choices may require judgment; bookkeeping over those choices should be exact.
