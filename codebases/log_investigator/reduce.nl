import { step } from "./step";
---
description: Reduce one log stream event through the semantic incident investigator.
args:
  state: IncidentState
  event: LogEvent
  files?: Dict<File>
returns: IncidentState
uses:
  step: ./step.nl
---
function reduce(state, event, files) -> IncidentState
  return step(state, event, files)
