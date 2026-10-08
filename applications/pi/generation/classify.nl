---
description: Response classification (pi-durable generation.ts classify, spec §8.3). What a terminal provider message means for the run - poll, tool round, answer, overflow compaction, retry or failure - and the commit for the cases that end here.
args:
  facts: PhaseFacts
  checkpoint: GenerationCheckpoint
  message: AssistantMessage
  pollAt: number | null
uses: [harness/context, harness/cut]
returns: '"answer" | "tools" | "committed"'
---
Decide what message means. checkpoint is the request or poll checkpoint it answers ({ attempt, compacted?, model,
cutoff, … }); pollAt is set when message came from polling a deferred response, null after a request. Apply the
first rule that matches. Return "answer" or "tools" without committing for those two; every other rule commits the
next state and returns "committed". keep = { compacted: checkpoint.compacted } when checkpoint has compacted, else {}.

1. Deferred. message.stopReason is "deferred" and message.deferred is set: handle = message.deferred. next =
   durable.now() + (handle.pollAfterMs ?? 5000); when pollAt is not null, next = max(next, pollAt + 1). Commit
   [{ op: "liveGeneration", value: { attempt: checkpoint.attempt, deferred: { pollAt: next } } },
    { op: "next", state: { status: "running", checkpoint: { phase: "poll", attempt: checkpoint.attempt, ...keep,
      model: checkpoint.model, cutoff: checkpoint.cutoff, handle, pollAt: next } } }]
   and return "committed". No hook runs: the message is not terminal.
2. Observers. names = await durable.hooks("afterResponse"); for each index i: await durable.hook("afterResponse", i,
   [message]). Their results do not matter.
3. Tool calls. message.stopReason is "toolUse" and message.content has at least one item of type "toolCall": return
   "tools".
4. Answer. message.stopReason is "stop", "length" or "toolUse" (toolUse without calls): return "answer".
5. Overflow. f = ai.failure(message). When f.overflow, checkpoint has no compacted, and
   facts.settings.compaction.enabled: view = await context(checkpoint.cutoff); when (await cut(view,
   facts.settings.compaction.keepRecentTokens)) is not null, compact and prepare again (this is not a retry). Commit
   [{ op: "appendAssistant", message }, { op: "liveGeneration", value: null },
    { op: "createCompaction", reason: "overflow", owner: "self", as: "c" },
    { op: "next", state: { status: "waiting", on: ["$c"], policy: "allSettled", checkpoint: { phase: "prepare",
      attempt: checkpoint.attempt, compacted: "$c", overflow: message.errorMessage ?? "Context overflow" } } }]
   and return "committed".
6. Retry or fail. policy = facts.settings.retry. retry = message.stopReason is "error" and not f.overflow and
   f.retryable and policy.enabled and checkpoint.attempt <= policy.maxRetries.
   - retry: until = durable.now() + ai.retryDelayMs(policy, checkpoint.attempt). Commit
     [{ op: "appendAssistant", message },
      { op: "liveGeneration", value: { attempt: checkpoint.attempt, retry: { at: until, error: message.errorMessage ?? "" } } },
      { op: "next", state: { status: "running", checkpoint: { phase: "retry", attempt: checkpoint.attempt, ...keep, until } } }].
   - otherwise fail the run: text = message.errorMessage ?? "Model response ended with stop reason " +
     message.stopReason. Commit
     [{ op: "appendAssistant", message },
      { op: "endRun", settlement: { status: "unanswered", reason: "model_error", detail: text } },
      { op: "next", state: { status: "terminal", outcome: { status: "failed", error: { message: text, detail: { reason: "model_error" } } } } }].
     This covers every other error, a second overflow, "aborted" without an abort, and "deferred" without a handle.
   Return "committed".
