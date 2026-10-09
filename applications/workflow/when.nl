---
description: When an order is looked at again. After a step, say how many milliseconds until the order is next reviewed, or that it needs no review.
args:
  after: WorkflowState
  decision: Decision
  delay: number
returns: number | null
---
after is the order after this step, decision is what was just done or chosen for it, and delay is the usual number of
milliseconds between reviews of an order whose outcome is unknown.

1. When after.pending is not empty (an operation's outcome is unknown), answer decision.waitMs when it is given, and
   delay otherwise.
2. When after.pending is empty and decision.waitMs is given, answer decision.waitMs.
3. Otherwise answer null: the order needs no review.

Answer a whole number of milliseconds greater than 0, or null.
