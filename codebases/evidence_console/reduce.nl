---
description: Answer a terminal question through the semantic evidence-search program.
args:
  state: ConsoleState
  event: ConsoleEvent
returns: ConsoleState
uses:
  answer: ../evidence_atlas/answer.nl
---
function reduce(state, event) -> ConsoleState
  if event.kind is not "question":
    return state
  result = answer(event.value)
  return append(state, event.value, result)
