---
description: Tool round control (pi-durable generation.ts finishToolRound, spec §6, §8.5, appendix 2 R3). After every tool task of the round is terminal - terminate, handoff, added tools, the boundary - end the run or hand it to the next generation.
args:
  facts: PhaseFacts
  checkpoint: GenerationCheckpoint
uses: [harness/context, harness/boundary]
returns: string
---
checkpoint is { phase: "tools", assistant, tools, pending: [] }: assistant is the tool-calling answer's entry and
tools the round's tool tasks in creation order, all terminal now. Decide the round's outcome and commit it in one
commit. Nothing limits the number of rounds.

1. Controls. outcomes = await durable.outcomes(checkpoint.tools). control of task tools[k] = outcomes[k].result.control
   when outcomes[k].status is "completed", else none.
2. Slots. live = await durable.live(); slots = live.tools ?? []. results = the entry of every slot that has one, in
   slot order.
3. Observers. names = await durable.hooks("afterTools"); for each index i: await durable.hook("afterTools", i,
   [checkpoint.assistant, results]).
4. terminate = slots is not empty and every slot has a taskId whose control has terminate: true (a call answered
   without a task, such as an unavailable tool, prevents it). added = every name of every control's addTools, in task
   order. handoff = the handoff text of the last control (in task order) that has one, or none.
5. Boundary facts: items = await durable.inbox(); view = await context(); activeStart = view.head ? view.head.head :
   null. modes: facts.settings.steeringMode and facts.settings.followUpMode.
6. ops = []; when added is not empty, first { op: "addTools", names: added }.
   a. terminate or handoff: the run ends with the tool-calling answer.
      - With handoff, first add { op: "appendEntry", entry: { kind: "pi.reset", head: "self", model: [{ role: "user",
        content: handoff, timestamp: facts.now }] } }. That entry starts a new context, newer than every queued head:
        pass activeStart = (the largest id in view.entries) + 1 to boundary so every queued write with a numeric head
        is judged stale.
      - selection = await boundary(items, steeringMode, followUpMode, "final", activeStart). Add { op: "place",
        selection }, { op: "endRun", settlement: { status: "done", answer: checkpoint.assistant } }, and when
        selection.users is not empty { op: "startRun", inputs: selection.users }.
   b. Otherwise: selection = await boundary(items, steeringMode, followUpMode, "postTools", activeStart). Add
      { op: "place", selection }, then:
      - selection.reset (a queued reset cut the context before an answer): { op: "endRun", settlement: { status:
        "unanswered", reason: "reset" } }, and when selection.users is not empty { op: "startRun", inputs:
        selection.users };
      - otherwise the run goes on: { op: "clearTools" }, { op: "runInputs", append: selection.users } (the placed
        steers join the run), { op: "createTask", kind: "pi.generation", input: {}, owner: "conversation", as: "g" },
        { op: "handOver", to: "$g" }. Generation continues even with nothing queued.
   End with { op: "next", state: { status: "terminal", outcome: { status: "completed", result: { entryId:
   checkpoint.assistant } } } }.
7. Commit with expect { run: live.run ? live.run.taskId : null, inbox: items.map(item => item.id) }. If it rejects with
   "state changed", go back to step 5.
Return which way the round ended.
