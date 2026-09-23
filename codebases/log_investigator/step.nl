import assess from "./step/assess";
import gap from "./step/gap";
import observe from "./step/observe";
import search from "./step/search";
import transition from "./step/transition";
---
description: Investigate one log or source-gap event with exact evidence and semantic judgment. When an event refers to a local log or runbook, inspect its specific files leaf before judging it; do not walk unrelated files.
args:
  acc: IncidentState
  item: LogEvent
  files?: Record<string, File>
returns: IncidentState
---
function step(acc, item, files) -> IncidentState
  if item.kind is "gap":
    return gap(acc, item)
  observation = observe(item)
  evidence = search(observation)
  judgement = assess(item, observation, evidence, files)
  return transition(acc, item, observation, evidence, judgement)
