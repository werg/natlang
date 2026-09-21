---
description: Reduce one interactive terminal event through the semantic terminal program.
args:
  state: Session
  event: Event
returns: Session
uses:
  step: ./step.nl
---
function reduce(state, event) -> Session
  return step(state, event)
