---
description: Execute the declared notebook cell graph and interpret bounded results. Use files for supporting notes or data descriptions when answering the question.
args:
  goal: Text
  question: Text
  files?: Dict<File>
returns: NotebookState
---
function run(goal, question) -> NotebookState
  initial = prepare(goal)
  finished = repeat step(initial), until complete(state), at most one round per declared cell
  answer = explain(question, finished)
  return attach(finished, answer)
