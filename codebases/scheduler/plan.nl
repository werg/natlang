import commit from "./plan/commit";
import enumerate from "./plan/enumerate";
import infeasible from "./plan/infeasible";
import inspect from "./plan/inspect";
import rank from "./plan/rank";
---
description: Select a feasible workday schedule from exact host-generated alternatives.
args:
  request: string
returns: ScheduleResult
---
function plan(request) -> ScheduleResult
  snapshot = inspect()
  alternatives = enumerate()
  if alternatives.options is empty:
    return infeasible(snapshot, alternatives)
  chosen = rank(request, snapshot, alternatives)
  return commit(chosen, snapshot.revision)
