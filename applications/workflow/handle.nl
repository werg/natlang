---
description: Decide what the order workflow does next for one event, and check the decision before it is applied.
args:
  snapshot: Snapshot
  limits: Limits
returns: Decision
---
Choose the next action for the order in snapshot.state, given snapshot.event. snapshot.receipt is the remote's receipt
for snapshot.state.pending, or null. limits are the policy's numbers. The mechanism applies the decision only if
ledger accepts it.

1. choice = choose(snapshot, limits).
2. problem = ledger.validate(snapshot.state, snapshot.event, choice). When problem is null, return choice.
3. Otherwise choose again with the problem in hand: choice = choose(snapshot, limits, problem), and check it with
   ledger.validate again. When the problem is null, return choice.
4. When there is still a problem, the workflow takes no step and the reason says so: return
   { action: "wait", reason: the problem, waitMs: limits.recheckMs }.
