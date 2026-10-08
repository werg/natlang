---
description: The tool call's checks before it runs, in pi's order - lookup, repair, validation, beforeTool, validation again - then the durable intent (pi-durable tool.ts call phase, spec §7.3, §8.4).
args:
  facts: PhaseFacts
  call: ToolCall
  tool: AgentTool | null
returns: "{ args?: JsonObject }"
---
Check call before it runs, then record the intent to run it. The call's stored arguments stay what the model sent;
args is the working copy. A check that fails settles the call with a harness error and you return {} (no args).

Settling with an error (code, message): live = durable.live(); slot = the item of live.tools whose taskId is
facts.task.id, or null. Commit
  [{ op: "appendToolResult", call: { id: call.id, name: call.name }, error: { code, message }, as: "e" },
   { op: "slot", callId: call.id, status: "done", entry: "$e" }   (only when slot is not null)
   { op: "next", state: { status: "terminal", outcome: { status: "completed", result: { entryId: "$e" } } } }]
and return {}. An error result still completes the task.

Steps, in order; stop at the first that settles:
1. Lookup. tool is null: settle with ("tool_unavailable", "Tool <call.name> is not available").
2. Repair. r = tools.prepare(call.name, call.arguments). r.error: settle with ("invalid_arguments", r.error).
   Otherwise args = r.args.
3. Validation. v = ai.validateArguments(call.name, tool.parameters, args). v.error: settle with
   ("invalid_arguments", v.error), the text exactly as returned. Otherwise args = v.args.
4. beforeTool. handlers = durable.hooks("beforeTool"). For each index i in order, while no block is set:
   h = durable.hook("beforeTool", i, [{ ...call, arguments: args }]).
   - h.error: block = h.error (a throwing hook blocks the call);
   - else h.value has block (a string): block = h.value.block;
   - else h.value has arguments: args = h.value.arguments.
   With a block: settle with ("blocked", "Tool call blocked: <block>").
5. Validation again, as in step 3, on the possibly replaced args.
6. Intent: one commit
   [{ op: "slot", callId: call.id, status: "running" }   (only when slot is not null; read it as in settling)
    { op: "next", state: { status: "running", checkpoint: { phase: "execute", arguments: args,
      replay: tool.replay ?? "unsafe" } } }]
   and return { args }.

Return a result that is not null from eval, exactly as computed: end with an eval whose code is `return <the variable that holds it>;` and set finish true. Never write it out in return_result: it carries model text and provider data that must stay byte for byte.
