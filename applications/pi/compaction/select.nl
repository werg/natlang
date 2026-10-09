---
description: Choose what a compaction summarizes - the cut, the beforeCompact hook, the pinned summary request (pi-durable compaction.ts select phase, spec §8.7).
args:
  facts: PhaseFacts
  info: ModelInfo
returns: SummaryToPlace | Committed
uses: [harness/context, harness/cut]
---
Choose the old part of the conversation this compaction replaces with a summary. Either return a summary for the
caller to place, or commit the task's next state yourself and return { committed: r.committed }, where r is that
commit's result: the caller then knows you committed, and which state.

1. view = context() (the current committed model context). policy = facts.settings.compaction.
   k = cut(view, policy.keepRecentTokens): the index in view.entries of the first entry kept verbatim, or null.
   Always ask cut: this task exists because a compaction was requested, so policy.enabled (which governs only
   automatic compaction) does not matter here, and only cut decides whether there is anything to compact.
2. k is null: nothing to compact. Commit [{ op: "compactionStatus", remove: true },
   { op: "next", state: { status: "terminal", outcome: { status: "completed", result: {} } } }] and return
   { committed }.
3. firstKept = view.entries[k].id.
4. beforeCompact. handlers = durable.hooks("beforeCompact"). When there are any, payload = { reason:
   facts.task.input.reason, entries: view.entries.slice(0, k), messages: messages.summarizedMessages(view, k),
   firstKept }, plus instructions when facts.task.input has them. For each index i in order until one decides:
   h = durable.hook("beforeCompact", i, [payload]). h.error: durable.report("beforeCompact: " + h.error) and go on.
   h.value is an object: that is the decision; stop asking.
   - decision.decline: nothing to compact; commit as in step 2 and return { committed }.
   - decision.summary: return { summary: decision.summary, firstKept } (the hook wrote it; no usage).
5. No decision: pin the request and move to summarize. Commit [{ op: "next", state: { status: "running",
   checkpoint: { phase: "summarize", attempt: 1, model: facts.agent.model, thinkingLevel: facts.agent.thinkingLevel,
   streamOptions: facts.settings.stream, maxTokens, tail, firstKept } } }] and return { committed }, where
   - maxTokens = Math.floor(0.8 * policy.reserveTokens), lowered to info.maxTokens when info.maxTokens > 0 and smaller;
   - tail = the largest id among view.entries (at least firstKept).
