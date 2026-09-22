---
description: Reduce one interactive terminal event through the semantic terminal program.
args:
  state: Session
  event: Event
  files?: Dict<File>
returns: Session
uses:
  step: ./step.nl
---
function reduce(state, event, files) -> Session
  return step(state, event, files)
