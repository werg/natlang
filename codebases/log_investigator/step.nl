---
description: Investigate one log or source-gap event with exact evidence and semantic judgment.
args:
  acc: IncidentState
  item: LogEvent
returns: IncidentState
---
function step(acc, item) -> IncidentState
  if item.kind is "gap":
    return gap(acc, item)
  observation = observe(item)
  evidence = search(observation)
  judgement = assess(item, observation, evidence)
  return transition(acc, item, observation, evidence, judgement)
