---
description: One round of forward chaining - apply every rule to what is known.
args:
  state: State
  rules: Text[]
returns: State
---
function derive(state, rules) -> State

  found = for each r in rules: conclusions(r, state.known)
  return merge(state, found)
