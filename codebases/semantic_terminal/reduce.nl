---
description: Reduce one interactive terminal event through the semantic terminal program.
args:
  state: Session
  event: Event
  files?: Record<string, File>
returns: Session
---
function reduce(state, event, files) -> Session
  return step(state, event, files)
