---
args:
  state: State
  files?: Record<string, File>
returns: State
---
function step(state, files) -> State
  ready = ready_tasks(state)
  if ready is empty:
    return stall(state)
  else:
    chosen = choose(ready, state.goal, files)
    return advance(state, chosen)
