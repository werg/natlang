---
description: Build report, closing words. What the build did and what a person should do next, from its final state.
args:
  state: BuildState
returns: Summary
---
Write the closing words of a build for state.goal from the final state, in plain language for the person who asked
for the build.

- summary: at most three sentences. Say the status. For a build that is done, say how many tasks ran and how many
  were reused because their recorded outputs were valid (state.judgments, by action). For a failed or unknown task,
  name it and state its cause and fix from state.diagnosis. For a blocked build, name the tasks that wait
  (state.blocked, state.detail). For an invalid one, name the faults in state.detail.
- next: the concrete actions that follow, one short sentence each, from state.diagnosis.fix and state.diagnosis.retry
  (inspect-first means check the task's outputs before running again), or from state.detail. Empty when the build
  is done.

Return { summary, next }.
