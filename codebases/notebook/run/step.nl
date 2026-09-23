import advance from "./step/advance";
import choose from "./step/choose";
import ready_cells from "./step/ready_cells";
import stall from "./step/stall";
---
args:
  state: NotebookState
  files?: Record<string, File>
returns: NotebookState
---
function step(state, files) -> NotebookState
  ready = ready_cells(state)
  if ready is empty:
    return stall(state)
  chosen = choose(ready, state.goal, files)
  return advance(state, chosen)
