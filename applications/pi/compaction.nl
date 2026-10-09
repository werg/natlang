---
description: The pi.compaction task, one phase at a time. Pick an old prefix of the model context, have it summarized, and place a summary entry whose head cuts the context (pi-durable compaction.ts CompactionTask, placeSummary, spec §8.7).
args:
  facts: PhaseFacts
returns: string
---
When facts.previousAttempt is set, this phase already ran once and failed for that reason: read it first and do not
repeat the mistake.
You carry out one phase of a compaction task. facts.task.id is this task; facts.task.input is a CompactionInput
({ reason, instructions? }); facts.task.checkpoint is a CompactionPhase. Every branch ends with this task's next
state committed through durable.commit, by you or by the function you call. Return one line saying what was
committed. If a commit rejects because this task was aborted or the invocation ended, stop at once and return
"stopped".

abort (facts.mode is "abort"): commit [{ op: "compactionStatus", remove: true },
{ op: "next", state: { status: "terminal", outcome: { status: "aborted" } } }]. No entry is written.

Otherwise, by checkpoint.phase:

select and summarize first resolve the model: ref = facts.agent.model for select, checkpoint.model for summarize.
info = ai.model(ref) when ref is set. With no ref the message is "No model is configured"; with a ref ai.model does
not know, "Model <provider>/<modelId> is not available". Then commit [{ op: "compactionStatus", remove: true },
{ op: "next", state: { status: "terminal", outcome: { status: "failed", error: { message, detail: { reason:
"no_model" } } } } }] and return. A manual compaction runs even when automatic compaction is disabled.

select: p = select(facts, info). When p has committed, select already committed the task's next state (p.committed
says which: done, or the summarize checkpoint): commit nothing more and return p.committed. Otherwise place p (below).

summarize: p = summarize(facts, checkpoint). When p has committed, summarize already committed the task's next
state (a retry or a failure): commit nothing more and return p.committed. Otherwise place p.

retry: durable.sleep(checkpoint.until). Then commit [{ op: "compactionStatus", attempt: checkpoint.attempt + 1,
retry: null }, { op: "next", state: { status: "running", checkpoint } }] where the new checkpoint is the old one with
phase "summarize", attempt + 1, and no until.

Placing a summary p (a SummaryToPlace):
1. entry = summaryEntry(p.summary, p.firstKept, facts.task.input.reason, durable.now()).
2. usage = [{ op: "usage", bucket: "models", key: p.usageKey, usage: p.usage }] when p.usage is set, else [].
3. Who owns this task decides how the entry is placed: task = durable.task(facts.task.id).
   - task.owner is set: a generation owns it and waits for it while holding the run, so append directly:
     commit [...usage, { op: "compactionStatus", remove: true }, { op: "appendEntry", entry, as: "s" },
     { op: "next", state: { status: "terminal", outcome: { status: "completed", result: { entryId: "$s" } } } }].
   - Otherwise the conversation owns it: only the run's task may append to a busy conversation, so admit the entry
     as a write: submissionId = durable.submit({ type: "write", requestId: "compaction:" + facts.task.id, entry }).
     It is placed at once when the conversation is idle, at its next turn boundary when busy, or settled stale.
     Admitting the same requestId again returns the same ID, so this step is safe to repeat. Then commit
     [...usage, { op: "compactionStatus", remove: true },
     { op: "next", state: { status: "terminal", outcome: { status: "completed", result: { submissionId } } } }].
