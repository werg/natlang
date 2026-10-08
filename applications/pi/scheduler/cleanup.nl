---
description: The Harness cleanup of a task the scheduler settles itself, faulted or orphaned (pi-durable live.ts settleSchedulerOutcome). Returns the write operations; the host applies them in the commit that makes the task terminal.
args:
  task: CleanupTask
  outcome: SchedulerOutcome
returns: Op[]
---
Return the operations that clean up after task, which the scheduler settles with outcome without running the task.
The operations apply in task's own scope (task.id is "this task" for slot, compactionStatus and endRun). Read the
conversation's live state first: live = scheduler.live(task.conversationId).

By task.kind:
1. "pi.tool": find the slot in live.tools whose taskId is task.id. With such a slot, return
   [{ op: "slot", callId: slot.callId, status: "done" }]: the slot is done without a result entry, and model context
   then shows a missing result for its call. Without one, return [].
2. "pi.compaction": return [{ op: "compactionStatus", remove: true }].
3. "pi.generation": only when live.run exists and live.run.taskId is task.id (the generation owns the run), return
   [{ op: "convertPartial" }, { op: "endRun", settlement }], where settlement is
   { status: "unanswered", reason: "faulted", detail: outcome.message } when outcome.status is "faulted", and
   { status: "unanswered", reason: outcome.reason } when it is "orphaned". Otherwise return [].
4. Any other kind: return [].
