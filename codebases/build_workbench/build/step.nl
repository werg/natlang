import { advance } from "./step/advance";
import { choose } from "./step/choose";
import { ready_tasks } from "./step/ready_tasks";
import { stall } from "./step/stall";
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
