---
description: A scheduling pass (spec §5.1, §5.4; R4, R4a, R9). Decides which live tasks to reserve now and in which mode, which abort-marked tasks to orphan, and whether each awaited scope is idle.
args:
  facts: PassFacts
returns: PassDecision
uses: [scheduler/cleanup]
---
Classify every task of facts.tasks, in order, and collect the decision. Apply THE OWNERSHIP TREE rules of the facts
types exactly. There is no priority and no limit: everything eligible is reserved in this one pass.

First compute, for every live task, its owned live work (an ID list) with the OWNED LIVE WORK rule over facts.tasks.

For each task t, the first matching rule decides:
1. t.invocation is not null: it is running; skip it.
2. t.status is "completing": its outcome is held; skip it.
3. Its waits-on list is not empty: skip it. The waits-on list is t's owned live work when t.abortRequested is true
   (abort runs bottom up); otherwise, when t.status is "waiting", the IDs of t.on that are live (in facts.tasks);
   otherwise empty.
4. The definition fit (R4a), with def = the entry of facts.definitions whose kind is t.kind:
   - no def: blocked "missing_task";
   - def.version equals t.version: fits, no migration;
   - def.version is lower than t.version: blocked "task_too_old";
   - t.id is in facts.migrationFailed: blocked "migration_failed";
   - otherwise: fits, with migration.
   Blocked and t.abortRequested is true: add { id: t.id, reason, cleanup } to orphan, where cleanup =
   cleanup({ id: t.id, kind: t.kind, conversationId: t.conversationId }, { status: "orphaned", reason }).
   Blocked and not abort-marked: skip it; it stays pending and is reconsidered when the registry changes.
5. Otherwise add { id: t.id, mode: t.abortRequested ? "abort" : "run", migrate } to reserve, where migrate is true
   when it fits with migration.

Idle (R9): for each scope s in facts.idleScopes, s is idle when no task of facts.tasks with background false is in
scope of s, where a task is in scope when its walk `above` is IN SCOPE of s without crossing background owners, and a
walk that reaches "unknown" counts as in scope. s is a conversation ID, or null for every ownerless conversation. Add
{ scope: s, idle } for every s.

Return { reserve, orphan, idle }, each list possibly empty.
