---
args:
  state: NotebookState
returns: NotebookState
---
function step(state) -> NotebookState
  ready = ready_cells(state)
  if ready is empty:
    return stall(state)
  chosen = choose(ready, state.goal)
  return advance(state, chosen)
