---
description: The end of a turn (pi-durable generation.ts answer, spec §6, §8.3, appendix 2 R3). Record the final answer, place the final boundary, then continue the run (onYield), or settle it and start the next run for placed input.
args:
  facts: PhaseFacts
  message: AssistantMessage
uses: [harness/context, harness/boundary]
returns: string
---
message is this generation's final answer. Decide who runs next and commit it in one commit.

1. The onYield chain. names = await durable.hooks("onYield"). For each index i in order, until one continues:
   r = await durable.hook("onYield", i, [message]); when r.value has a continue field, continuation = r.value.continue
   and stop asking. An error result does not continue.
2. Read the boundary facts: live = await durable.live(); items = await durable.inbox(); view = await context();
   activeStart = view.head ? view.head.head : null.
3. selection = await boundary(items, facts.settings.steeringMode, facts.settings.followUpMode, "final", activeStart).
4. ops = [{ op: "appendAssistant", message, as: "a" }, { op: "place", selection }].
   - Continue: when continuation is set, selection.users is empty and selection.reset is false: add
     { op: "appendUser", content: continuation }, { op: "createTask", kind: "pi.generation", input: {}, owner:
     "conversation", as: "g" }, { op: "handOver", to: "$g" }, { op: "liveGeneration", value: null }. The run's
     inputs stay open with the successor.
   - Otherwise (a continuation is dropped, never retried): add { op: "endRun", settlement: { status: "done", answer:
     "$a" } }, and when selection.users is not empty, { op: "startRun", inputs: selection.users }.
   Either way end with { op: "next", state: { status: "terminal", outcome: { status: "completed", result: { entryId: "$a" } } } }.
5. Commit with expect { run: live.run ? live.run.taskId : null, inbox: items.map(item => item.id) }. If it rejects with
   "state changed" (new input arrived), go back to step 2.
Return what happened: continued, or answered and how many queued inputs started a new run.
