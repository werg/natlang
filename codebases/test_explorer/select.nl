---
description: Choose bounded dependency-plan cases that challenge a stated graph invariant.
args:
  question: Text
  cases: Case[]
  budget: Num
returns: Selection
types:
  Case: '{ id: Text, description: Text, group: Text }'
  Selection: '{ ids: Text[], reason: Text }'
---
Select at most budget distinct case IDs from cases. Prefer cases that test different
graph structures and missing dependencies. Explain what each selection probes. Only
select supplied IDs. The host will check the selection and execute each case in a
fresh runtime.
