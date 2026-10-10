---
kind: directory-reducer
args:
  state: SearchState
  policy: ImprovementPolicy
returns: SearchState
---
Improve this source through one evidenced experiment. Return `await lifecycle.step(folder, evaluator, plans, state, policy)` in eval with finish:true. The authored helpers plan the experiment, measure the source and its training feedback, ask the diagnose, hypothesize and edit functions for one change to the private folder, then check, measure and select the result. Return a rejected or unchanged experiment honestly. The outer folder.iterateOn owns repetition; one experiment runs per step, with the scores the host measured.
