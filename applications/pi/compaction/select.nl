---
description: Choose what a compaction summarizes - the cut, the beforeCompact hook, the pinned summary request (pi-durable compaction.ts select phase, spec §8.7).
args:
  facts: PhaseFacts
  info: ModelInfo
returns: SummaryToPlace | null
uses: [harness/context, harness/cut]
---
Choose the old part of the conversation this compaction replaces with a summary. Commit and return null, or return
a summary for the caller to place.

1. view = context() (the current committed model context). policy = facts.settings.compaction.
   k = cut(view, policy.keepRecentTokens): the index in view.entries of the first entry kept verbatim, or null.
2. k is null: nothing to compact. Commit [{ op: "compactionStatus", remove: true },
   { op: "next", state: { status: "terminal", outcome: { status: "completed", result: {} } } }] and return null.
3. firstKept = view.entries[k].id.
4. beforeCompact. handlers = durable.hooks("beforeCompact"). When there are any, payload = { reason:
   facts.task.input.reason, entries: view.entries.slice(0, k), messages: messages.summarizedMessages(view, k),
   firstKept }, plus instructions when facts.task.input has them. For each index i in order until one decides:
   h = durable.hook("beforeCompact", i, [payload]). h.error: durable.report("beforeCompact: " + h.error) and go on.
   h.value is an object: that is the decision; stop asking.
   - decision.decline: nothing to compact; commit as in step 2 and return null.
   - decision.summary: return { summary: decision.summary, firstKept } (the hook wrote it; no usage).
5. No decision: pin the request and move to summarize. Commit [{ op: "next", state: { status: "running",
   checkpoint: { phase: "summarize", attempt: 1, model: facts.agent.model, thinkingLevel: facts.agent.thinkingLevel,
   streamOptions: facts.settings.stream, maxTokens, tail, firstKept } } }] and return null, where
   - maxTokens = Math.floor(0.8 * policy.reserveTokens), lowered to info.maxTokens when info.maxTokens > 0 and smaller;
   - tail = the largest id among view.entries (at least firstKept).

Return a result that is not null from eval, exactly as computed: end with an eval whose code is `return <the variable that holds it>;` and set finish true. Never write it out in return_result: it carries model text and provider data that must stay byte for byte.
