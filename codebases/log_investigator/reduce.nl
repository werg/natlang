---
description: Reduce one log stream event through the semantic incident investigator.
args:
  state: IncidentState
  event: LogEvent
  files?: Record<string, File>
returns: IncidentState
---
function reduce(state, event, files) -> IncidentState
  return step(state, event, files)
