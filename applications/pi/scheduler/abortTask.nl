---
description: Harness.abortTask (spec §2.2, §5.4; R7). Decides whether to reject, report a terminal task, orphan a task no definition can take, or set the abort mark, and whether to join the running invocation.
args:
  facts: AbortTaskFacts
returns: AbortTaskDecision
uses: [scheduler/cleanup]
---
Decide the abort of task facts.id, applying THE OWNERSHIP TREE rules of the facts types. The first matching rule
decides:

1. facts.task is null: return { kind: "reject", message: "Task <facts.id> does not exist", join: false }.
2. facts.task.status is "terminal": return { kind: "terminal", join: false }. Nothing is written.
3. facts.invocation is null, facts.task.status is not "completing", the task has no owned live work (OWNED LIVE WORK
   over facts.tasks), and facts.blocked is not null: no definition can take it, so it ends now. Return
   { kind: "orphan", reason: facts.blocked, join: false, cleanup }, where cleanup = cleanup({ id: facts.task.id,
   kind: facts.task.kind, conversationId: facts.task.conversationId }, { status: "orphaned", reason: facts.blocked }).
4. Otherwise return { kind: "mark", join: facts.invocation === "run" }. A completing task is only marked: it keeps
   its held outcome, and the mark cascades to its owned work.
