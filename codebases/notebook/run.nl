---
description: Execute the declared notebook cell graph and interpret bounded results.
args:
  goal: Text
  question: Text
returns: NotebookState
---
function run(goal, question) -> NotebookState
  initial = prepare(goal)
  finished = repeat step(initial), until complete(state), at most one round per declared cell
  answer = explain(question, finished)
  return attach(finished, answer)
