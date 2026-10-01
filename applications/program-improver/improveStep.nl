---
kind: directory-reducer
args:
  state: SearchState
  policy: ImprovementPolicy
returns: SearchState
---
Improve this source through one evidenced experiment. Return `await lifecycle.step(folder, evaluator, rewriteProgram, state, policy)` in eval with finish:true. Exact authored helpers supply measured source and training feedback to one directory reducer that diagnoses and edits the private folder, then check, measure and select the result. Return a rejected or unchanged experiment honestly. The outer folder.iterateOn owns repetition; do not run additional experiments inside this step or invent scores.
