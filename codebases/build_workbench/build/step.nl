---
args:
  state: State
returns: State
---
function step(state) -> State
  ready = ready_tasks(state)
  if ready is empty:
    return stall(state)
  else:
    chosen = choose(ready, state.goal)
    return advance(state, chosen)
