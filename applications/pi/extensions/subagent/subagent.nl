---
description: pi's subagent tool. Run a self-contained task in a child conversation and return its answer.
args:
  args: "{ task: string }"
returns: ToolExecutionResult
---
Delegate args.task to a subagent through the delegation service. When a step says "fail with", finish with status
failed and that text exactly as the reason.

1. child = delegation.child(): this call's child conversation (found again on a rerun, else created).
2. delegation.details({ conversationId: child }).
3. settled = delegation.ask(child, args.task, "subagent:<delegation.taskId()>"). The request ID makes a rerun reuse
   the same submission instead of asking twice.
4. Unless settled.status is "done" and settled.type is "input", fail with "Subagent <child> failed: <settled.status>".
5. text = delegation.answerText(settled.answer).
6. Return { content: [{ type: "text", text }], details: { conversationId: child } }.
