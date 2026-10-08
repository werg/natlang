---
description: Tool round setup (pi-durable generation.ts startToolRound and createToolTask, spec §8.3, §8.5). Record the tool-calling answer and start its tool tasks, one commit.
args:
  facts: PhaseFacts
  checkpoint: GenerationCheckpoint
  message: AssistantMessage
uses: [harness/context]
returns: string
---
message is an answer with tool calls; checkpoint is the request or poll checkpoint it answers (it has cutoff). Start
its tool round in one commit; the generation keeps the run and waits.

1. calls = the items of message.content with type "toolCall", in order.
2. offered = the names of (await context(checkpoint.cutoff)).tools: the tools the request offered, from the committed
   context (never a beforeRequest replacement).
3. sequential = facts.settings.toolExecution is "sequential", or some call names an offered tool whose entry in
   facts.agent.tools has executionMode "sequential".
4. Build the ops. ops = [{ op: "appendAssistant", message, as: "a" }]; slots = []; tasks = []; pending = []. For each
   call, at index i, in call order:
   - call.name not in offered: it is answered now. Add { op: "appendToolResult", call: { id: call.id, name:
     call.name }, error: { code: "tool_unavailable", message: "Tool " + call.name + " is not available" }, as: "r" + i }
     and push the slot { callId: call.id, name: call.name, status: "done", entry: "$r" + i }.
   - sequential and tasks is not empty: push call.id to pending and the slot { callId, name, status: "pending" } (no
     task yet).
   - otherwise create its task: add { op: "createTask", kind: "pi.tool", input: { assistant: "$a", callId: call.id },
     owner: "self", as: "t" + i }, push "$t" + i to tasks and the slot { callId, name, taskId: "$t" + i, status:
     "pending" }.
   So in a sequential round only the first offered call starts now; an unavailable call never counts as started.
5. Add { op: "liveGeneration", value: null }, { op: "setTools", slots }, and
   { op: "next", state: { status: "waiting", on: tasks, policy: "allSettled", checkpoint: { phase: "tools",
     assistant: "$a", tools: tasks, pending } } }. Commit.
Return how many tasks started, how many calls wait, and how many were unavailable.
