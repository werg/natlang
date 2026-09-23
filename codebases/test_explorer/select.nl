---
description: Choose bounded dependency-plan cases that challenge a stated graph invariant.
args:
  question: string
  cases: Case[]
  budget: number
returns: Selection
---
Select at most budget distinct case IDs from cases. Prefer cases that test different
graph structures and missing dependencies. Explain what each selection probes. Only
select supplied IDs. The host will check the selection and execute each case in a
fresh runtime.
Write one `Selection` record to `return` with exactly `ids` (the selected ID
list) and `reason` (a short explanation) fields. Even when choosing one case,
the result is a record, not a bare list.
