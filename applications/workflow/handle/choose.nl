---
description: The workflow's policy in natural language. Find where the order stands, then choose the one next action for that situation.
args:
  snapshot: Snapshot
  limits: Limits
  problem?: string
returns: CheckedDecision
---
Choose the next action for the order in snapshot.state, given snapshot.event, with the stages in your folder. limits are
the policy's numbers. problem, when given, says why an earlier choice was refused: choose differently.

1. situation = situation(snapshot.state, snapshot.event).
2. By situation:
   - "uncertain": return recover(snapshot, limits).
   - "settled": return { action: "wait", reason: a sentence saying why nothing happens }.
   - "forward": return advance(snapshot.state).
   - "cancelling" or "owed": plan = compensate(snapshot.state). When plan.steps is empty, return
     { action: "wait", reason: "nothing is left to undo" }. Otherwise return { action: plan.steps[0].action, reason:
     plan.steps[0].reason }.
   - "failed": operation = the last entry of snapshot.state.history. kind = failure(operation).
     attempts = how many entries of the history have operation.key and status "failed".
     When kind is "transient" and attempts is at most limits.transientRetries: return { action: operation.action,
     reason: a sentence saying the earlier attempt failed for a transient reason }.
     Otherwise plan = compensate(snapshot.state) and return its first step as in "cancelling" (a wait when it is empty).
