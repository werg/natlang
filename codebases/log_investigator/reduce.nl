---
description: Reduce one log stream event through the semantic incident investigator.
args:
  state: IncidentState
  event: LogEvent
returns: IncidentState
uses:
  step: ./step.nl
---
function reduce(state, event) -> IncidentState
  return step(state, event)
