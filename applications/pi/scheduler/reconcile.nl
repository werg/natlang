---
description: Cascades, joins and finalization (spec §5.4, §5.5; R8). Decides abort marks below cancelled owners, failFast marks, which conversations' queued inputs to withdraw, and which held outcomes become terminal.
args:
  facts: ReconcileFacts
returns: ReconcileDecision
uses: [scheduler/cleanup]
---
Decide what the committed records imply, applying THE OWNERSHIP TREE rules of the facts types exactly. Collect mark
(task IDs, each at most once), withdraw (conversation IDs) and finalize (in order).

1. Cascade. For every task t of facts.tasks with background false and abortRequested false: when t.above is BELOW
   CANCELLED, add t.id to mark.
2. failFast. For each waiter w of facts.failFast: when some member of w.members is FAILED (status "completing" or
   "terminal" with an outcome other than "completed"), add to mark every member that is live, is not failed itself and
   has abortRequested false. The waiter itself is not marked.
3. Withdraw. For each entry q of facts.queued: when q.above is BELOW CANCELLED, add q.conversation to withdraw.
4. Finalize. Repeat, at most as many times as facts.tasks has tasks: compute the owned live work of every task over
   the remaining live tasks (facts.tasks without those finalized so far); every remaining task with status
   "completing" and no owned live work is finalized now: add it to finalize, in facts order, and remove it from the
   remaining tasks (finalizing a task can free its owner). Stop when a round finalizes nothing.
   For a finalized task whose held outcome is "faulted" or "orphaned", give it its cleanup:
   cleanup({ id: t.id, kind: t.kind, conversationId: t.conversationId }, { status: t.outcome, message: t.message,
   reason: t.reason }); other finalized tasks have no cleanup.

Return { mark, withdraw, finalize }, each list possibly empty.
