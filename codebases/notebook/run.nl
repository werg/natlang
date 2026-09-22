import { attach } from "./run/attach";
import { complete } from "./run/complete";
import { explain } from "./run/explain";
import { prepare } from "./run/prepare";
import { step } from "./run/step";
---
description: Execute the declared notebook cell graph and interpret bounded results. If the question names a supporting note or schema, inspect that specific args/files leaf before answering.
args:
  goal: Text
  question: Text
  files?: Dict<File>
returns: NotebookState
---
function run(goal, question, files) -> NotebookState
  initial = prepare(goal)
  finished = repeat step(initial, files), until complete(state), at most one round per declared cell
  answer = explain(question, finished, files)
  return attach(finished, answer)
