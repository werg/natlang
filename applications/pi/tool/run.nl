---
description: Run one tool call with its final arguments and settle it - limits, execution, the error rule, assembly, afterTool, bounding, the result entry (pi-durable tool.ts run, finalResult and settle, spec §7.3).
args:
  facts: PhaseFacts
  call: ToolCall
  tool: AgentTool
  args: JsonObject
returns: string
---
Run call with args and commit its settlement. Return one line saying how it ended.

1. limits = { maxBytes: tool.outputLimits?.maxBytes ?? 51200, maxLines: tool.outputLimits?.maxLines ?? 2000,
   retain: tool.outputLimits?.retain ?? "head" }.
2. ex = tools.execute(call.name, args, limits).
3. The ending:
   - ex.error is set (the tool or its environment threw): result = { isError: true, diagnostics: [{ severity:
     "error", code: "tool_error", message: ex.error }] } with no content, and the ending is failed with message
     "Tool <call.name> threw".
   - Otherwise result = ex.result and the ending is completed. A result with isError true still completes.
4. a = format.assemble(result, ex). final = a.result; keep a.retained for step 6.
5. afterTool. handlers = durable.hooks("afterTool"). For each index i in order:
   h = durable.hook("afterTool", i, [call, final]). h.error: durable.report("afterTool: " + h.error) and go on.
   h.value is an object: final = h.value (it replaces the result). Otherwise final stays.
6. final = format.bound(final, limits, a.retained).
7. Settle: live = durable.live(); slot = the item of live.tools whose taskId is facts.task.id, or null. One commit:
   [{ op: "appendToolResult", call: { id: call.id, name: call.name }, result: final, durationMs: ex.durationMs, as: "e" },
    (leave durationMs out when ex.durationMs is absent)
    { op: "slot", callId: call.id, status: "done", entry: "$e" }   (only when slot is not null)
    { op: "next", state: { status: "terminal", outcome } }]
   outcome for completed: { status: "completed", result: { entryId: "$e", control: final.control } }, leaving
   control out when final.control is absent; for failed: { status: "failed", error: { message: "Tool <call.name>
   threw" }, result: { entryId: "$e" } }.
