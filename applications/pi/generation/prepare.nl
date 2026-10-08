---
description: The generation's prepare phase (pi-durable generation.ts prepare and thresholdCompaction, prompt.ts renderSections, spec §7.4, §8.3). Render the prompt, plan the system entries, decide compaction, and commit either a wait on a blocking compaction or the request checkpoint.
args:
  facts: PhaseFacts
  checkpoint: GenerationCheckpoint
  info: ModelInfo
uses: [harness/context, harness/cut, harness/estimate, harness/planSystem]
returns: string
---
Prepare one request of this generation. checkpoint is { phase: "prepare", attempt, compacted?, overflow? }; info is
the agent's model (facts.agent.model) as the catalog describes it. End with exactly one commit that sets the next
state, and return one line saying what it was.

1. After a blocking compaction for an overflow. When checkpoint has both compacted and overflow: outcome =
   (await durable.outcomes([checkpoint.compacted]))[0]. Unless outcome.status is "completed" and outcome.result.entryId
   is a number (a summary was placed), fail the run with the overflow text and return:
   durable.commit([{ op: "endRun", settlement: { status: "unanswered", reason: "model_error", detail: checkpoint.overflow } },
   { op: "next", state: { status: "terminal", outcome: { status: "failed", error: { message: checkpoint.overflow,
   detail: { reason: "model_error" } } } } }]).
2. The context. view = await context() (the newest committed context). shown = an object mapping each
   view.sections[i].key to its text (the sections the transcript shows now, already wrapped).
3. Render the sections. desired = []. For each section of facts.agent.sections, in order:
   r = await durable.renderSection(section.key, shown).
   - r.omit: the section shows nothing; skip it.
   - r.error: keep what the transcript shows: if shown[section.key] exists, push { key, text: shown[section.key] };
     otherwise skip it. (The host already reported the error.)
   - r.text: push { key, text: section.tag ? "<" + key + ">\n" + r.text + "\n</" + key + ">" : r.text }.
4. Plan. planned = await planSystem(view, desired, facts.agent.tools, facts.now): a list of { message, edits? }
   system entries to append (0, 1 or 2 of them).
5. Threshold. Only when checkpoint.compacted is absent; otherwise over = none. policy = facts.settings.compaction.
   - over = none when policy.enabled is false or info.contextWindow <= 0.
   - Otherwise tokens = await estimate(view, planned.map(p => p.message)); blocking = info.contextWindow -
     policy.reserveTokens; background = blocking - policy.backgroundTokens. over = "blocking" when tokens > blocking;
     else "background" when policy.backgroundTokens > 0 and tokens > background; else none.
   - When over is set but (await cut(view, policy.keepRecentTokens)) is null, over = none (nothing to compact).
6. Blocking: compact first, then prepare again. Commit and return:
   [{ op: "createCompaction", reason: "threshold", owner: "self", as: "c" },
    { op: "next", state: { status: "waiting", on: ["$c"], policy: "allSettled",
      checkpoint: { phase: "prepare", attempt: checkpoint.attempt, compacted: "$c" } } }]
   Whatever the compaction's outcome, prepare runs again and does not compact again.
7. Otherwise the request commit. tail = the largest id among view.entries (the newest entry). When view.entries is
   empty, fail this call: "Conversation <facts.task.conversationId> has no entries to send". ops = [], and for each
   planned[i], in order: { op: "appendSystem", message: planned[i].message, edits: planned[i].edits, as: "s" + i }
   (leave edits out when there are none). cutoff = "$s<last i>" when planned is not empty, else tail. When over is
   "background", add { op: "createCompaction", reason: "threshold", ifNone: true }. Then add
   { op: "next", state: { status: "running", checkpoint: { phase: "request", attempt: checkpoint.attempt,
     compacted: checkpoint.compacted (only when set), model: facts.agent.model, thinkingLevel: facts.agent.thinkingLevel,
     streamOptions: facts.settings.stream, cutoff } } }.
   Commit with expect { tail }: the planned entries are a diff against exactly that transcript. If the commit rejects
   with "state changed", the transcript grew: start again at step 2.
