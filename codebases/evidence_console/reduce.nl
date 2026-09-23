import append from "./reduce/append";
import answer from "../evidence_atlas/answer";
---
description: Answer a terminal question through the semantic evidence-search program.
args:
  state: ConsoleState
  event: ConsoleEvent
returns: ConsoleState
---
function reduce(state, event) -> ConsoleState
  if event.kind is not "question":
    return state
  result = answer(event.value)
  return append(state, event.value, result)
