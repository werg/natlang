---
description: The pi.tool task, one phase at a time. Resolve, check, hook, record the intent of, run and settle one tool call of a round (pi-durable tool.ts ToolTask, spec §7.3, §8.4).
args:
  facts: PhaseFacts
returns: string
---
When facts.previousAttempt is set, this phase already ran once and failed for that reason: read it first and do not
repeat the mistake.
You carry out one phase of a tool task. facts.task.id is this task; facts.task.input is { assistant, callId };
facts.task.checkpoint is a ToolCheckpoint. Every branch ends with this task's next state committed through
durable.commit, by you or by the function you call. Return one line saying what was committed. If a commit rejects
because this task was aborted or the invocation ended, stop at once and return "stopped".

1. Read the call. entry = durable.entry(facts.task.input.assistant). The call is the item of entry.model[0].content
   with type "toolCall" and id facts.task.input.callId. When the entry or the call is missing, fail this call with
   the reason "Entry <assistant> has no tool call <callId>"; that faults the task.
2. Read the slot: live = durable.live(); slot = the item of live.tools whose taskId is facts.task.id, or null when
   there is none. tool = the item of facts.agent.tools whose name is call.name, or null.

How a call settles ("settle with result and ending"): one commit, where e is the new result entry:
  [{ op: "appendToolResult", call: { id: call.id, name: call.name }, result, as: "e" },
   { op: "slot", callId: call.id, status: "done", entry: "$e" }   (only when slot is not null)
   { op: "next", state: { status: "terminal", outcome } }]
with outcome { status: "aborted", result: { entryId: "$e" } } for an aborted ending, or
{ status: "failed", error: { message }, result: { entryId: "$e" } } for a failed ending with that message.

Then by branch:

abort (facts.mode is "abort"): settle with result fromSlot(slot, "aborted", "Tool <call.name> was aborted") and an
aborted ending.

call (checkpoint.phase "call"): r = beginCall(facts, call, tool). It either settled the call (r.args is absent:
return its line) or committed the intent; then return run(facts, call, tool, r.args).

execute (checkpoint.phase "execute"): reached only when an earlier run was interrupted after its intent was
recorded. The call may rerun only when checkpoint.replay is "safe" and tool is not null and tool.replay is "safe".
- Rerun: when slot is not null, first commit [{ op: "slot", callId: call.id, clearProgress: true }] (no next: the
  rerun reports from scratch). Then return run(facts, call, tool, checkpoint.arguments): no repair, no validation and
  no beforeTool again.
- Otherwise: message = "Tool <call.name> was interrupted and may have partially run". Settle with result
  fromSlot(slot, "interrupted", message) and a failed ending with that message.
