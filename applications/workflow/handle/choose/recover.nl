---
description: Recover an operation whose outcome is unknown. Look for its receipt, look again later, or send it again under the same key.
args:
  snapshot: Snapshot
  limits: Limits
returns: Decision
---
snapshot.state.pending is the key of an operation that was started and not acknowledged. snapshot.receipt is the remote's
receipt for that key, or null. snapshot.state.checks counts the looks that found no receipt. Choose by the first rule
that holds:
1. snapshot.event.kind is not "reconcile": { action: "wait", reason: a sentence saying the outcome of the pending key is
   unknown and will be looked at again, waitMs: limits.recheckMs }.
2. snapshot.receipt is not null: { action: "reconcile", reason: a sentence saying the remote has a receipt }. The
   receipt proves the effect happened once.
3. snapshot.state.checks is below limits.checksBeforeRetry: { action: "reconcile", reason: a sentence saying no receipt
   exists yet and the remote is asked again, waitMs: limits.recheckMs }.
4. Otherwise: { action: "retry", reason: a sentence saying the looks found no receipt, so the same key is sent again
   (the remote carries an effect out once per key) }.
