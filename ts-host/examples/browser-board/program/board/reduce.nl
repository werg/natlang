---
description: Reduce one command or control event into the task board state.
args:
  state: Board
  event: UiEvent
returns: Board
---
function reduce(state, event) -> Board
  decision = choose(state, event)
  return apply(state, event, decision)
