---
description: Step precedence (spec §5.1, R5). Decides, before each phase of a run invocation and once after an abort handler, whether the invocation continues, stops, faults its task, or hands it over to a new definition.
args:
  facts: StepFacts
returns: StepDecision
uses: [scheduler/cleanup]
---
Decide what happens to the invocation from facts. Apply these rules in order; the first that matches is the decision.

1. facts.task is null, or facts.task.status is not "running" (it is waiting, pending or completing): return
   { kind: "stop" }. A waiting task ends its invocation even when its checkpoint is unchanged.
2. facts.closing is true: return { kind: "stop" }. The checkpoint and abort mark stay for reopening.
3. facts.mode is "abort" (the abort handler just returned without settling the task): fault with message
   facts.previous.error when facts.previous has an error, otherwise
   "Abort handler of task <facts.task.id> returned without a terminal outcome".
4. facts.task.abortRequested is true: return { kind: "stop" }. The abort invocation starts later.
5. facts.previous is null (no phase has run yet): return { kind: "continue" }.
6. facts.previous.error is set (the phase threw): fault with message facts.previous.error.
7. facts.previous.unchanged is true: fault with message
   "Task <facts.task.kind> phase <facts.previous.phase> returned without durable progress".
8. The phase made progress. If facts.definition.same is false:
   - facts.definition.exists and facts.definition.canReserve are both true: return { kind: "handover" };
   - otherwise, when facts.reported is false, return { kind: "continue", report:
     "Task <facts.task.id> keeps running under its old <facts.task.kind> definition" }; when it is true, go on.
9. Return { kind: "continue" }.

To fault with a message: cleanup = cleanup({ id: facts.task.id, kind: facts.task.kind, conversationId:
facts.task.conversationId }, { status: "faulted", message }), then return { kind: "fault", message, cleanup }.
