---
description: "The note added to the system prompt of a natural-language stopping predicate. The runtime replaces {progress} with one sentence about the loop's progress."
args:
  state: unknown
returns: boolean
---
This call is the stopping condition of an iterative loop (iterateOn). Answer true to stop the loop with the current state as its result, or false to run another step.
{progress}
Judge whether the criterion is met in substance by the current state. Accept a state that reasonably meets it; do not hold out for perfection or for details the criterion does not ask for, since every further step costs a model call and the loop ends only when you accept.
If the state has stopped changing, further steps are unlikely to change your answer: if it meets the criterion in substance, answer true. Do not answer true for a state that fails the criterion; a loop that cannot succeed is stopped by its progress review, not by this answer.
