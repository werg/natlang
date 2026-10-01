---
kind: directory-reducer
args:
  state: SearchState
  policy: ImprovementPolicy
returns: SearchState
---
Improve this source through one evidenced experiment. Return `await lifecycle.step(folder, evaluator, planExperiment, rewriteProgram, state, policy)` in eval with finish:true to complete in one action. The authored exact lifecycle measures the current source, invokes planExperiment for semantic diagnosis, then executes exactly one rewrite and measurement, independently selects the best measured population member and advances the finite experiment count. Do not duplicate its bookkeeping or invent scores. The outer folder.iterateOn owns repetition; this reducer performs one experiment only. A rejected candidate is returned as an experiment outcome; only the outer iteration can request another hypothesis.
