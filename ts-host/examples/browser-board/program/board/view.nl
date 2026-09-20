---
description: Generate a task board UI structure and render it as safe DOM data.
args:
  state: Board
returns: UiNode
---
function view(state) -> UiNode
  plan = describe(state)
  return layout(state, plan)
