---
description: Select a feasible workday schedule from exact host-generated alternatives.
args:
  request: Text
returns: ScheduleResult
---
function plan(request) -> ScheduleResult
  snapshot = inspect()
  alternatives = enumerate()
  if alternatives.options is empty:
    return infeasible(snapshot, alternatives)
  chosen = rank(request, snapshot, alternatives)
  return commit(chosen, snapshot.revision)
