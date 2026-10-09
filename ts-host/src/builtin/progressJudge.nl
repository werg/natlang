---
description: Library progress judge for iterateOn.
args:
  trajectory: Live<"IterationTrajectory", "shape", "summary,states,steps,repeats">
  step: string
returns: ProgressVerdict
types:
  ProgressVerdict: { verdict: "continue" | "divergent", reason: string }
---
An iterative process has been running for a while. Decide whether it is still making meaningful progress.
Inspect the trajectory: use trajectory.summary(), trajectory.steps(), trajectory.repeats(), and trajectory.states(start, end) to page through states.
Distinguish real improvement from repetition, oscillation between the same few states, unproductive churn, and a goal that looks impossible.
An unusually long run that is still improving should continue.
Return { verdict: "continue", reason } if further steps are likely to help, or { verdict: "divergent", reason } if the process is stuck or cannot succeed. Give a concrete reason.
