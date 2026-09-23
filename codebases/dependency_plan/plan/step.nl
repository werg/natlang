import advance from "./step/advance";
import choose from "./step/choose";
import ready_tasks from "./step/ready_tasks";
import stall from "./step/stall";
---
args:
  state: State
returns: State
---
function step(state) -> State
  ready = ready_tasks(state)
  if ready is empty:
    return stall(state)                # expose every unfinished task; do not invent an order
  else:
    chosen = choose(ready)
    return advance(state, chosen)      # checks readiness again before updating state
