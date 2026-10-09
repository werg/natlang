---
description: Ask the conversation's model for the summary with the pinned request, then classify the response - a summary, a retry, or a failure (pi-durable compaction.ts summarize phase, summaryText, summaryFailure, spec §8.7).
args:
  facts: PhaseFacts
  checkpoint: CompactionPhase
returns: SummaryToPlace | null
uses: [harness/context]
---
checkpoint is the summarize checkpoint: the pinned request (attempt, model, thinkingLevel, streamOptions,
maxTokens, tail, firstKept). Return the summary for the caller to place, or commit a retry or a failure and return
null.

1. view = context(checkpoint.tail): the context as of the tail, which no longer changes, so this is the range select
   chose. k = the index in view.entries of the entry whose id is checkpoint.firstKept.
2. messages = summaryRequest.summaryMessages(view, k, facts.task.input.instructions ?? null, durable.now()): pi's
   summarization system prompt and one user message with the serialized conversation.
3. options = checkpoint.streamOptions without its deferred field, plus cacheRetention "none", maxTokens
   checkpoint.maxTokens, thinkingLevel checkpoint.thinkingLevel, sessionId facts.sessionId.
4. message = ai.turn(checkpoint.model, messages, options). No live publishing and no tools.
5. usageKey = message.provider + "/" + message.model. Every attempt's usage counts, failed ones too.
6. The summary: when message.stopReason is "stop" and its content has no item of type "toolCall", join the text of
   its "text" items with "\n" and trim; a non-empty result is the summary. Then return { summary, firstKept:
   checkpoint.firstKept, usageKey, usage: message.usage }.
7. Retry, when message.stopReason is "error" and ai.failure(message).retryable and facts.settings.retry.enabled and
   checkpoint.attempt <= facts.settings.retry.maxRetries: until = durable.now() +
   ai.retryDelayMs(facts.settings.retry, checkpoint.attempt). Commit
   [{ op: "usage", bucket: "models", key: usageKey, usage: message.usage },
    { op: "compactionStatus", retry: { at: until, error: message.errorMessage ?? "" } },
    { op: "next", state: { status: "running", checkpoint: { ...checkpoint, phase: "retry", until } } }]
   and return null.
8. Otherwise it failed. The message, first match:
   - stopReason "error" or "aborted": "Summarization failed: <message.errorMessage, else the stopReason>";
   - "length": "Summarization hit the token limit; the summary is incomplete";
   - its content has a "toolCall" item: "Summarization attempted to call a tool";
   - anything else: "Summarization produced no text".
   Commit [{ op: "usage", bucket: "models", key: usageKey, usage: message.usage },
    { op: "compactionStatus", remove: true },
    { op: "next", state: { status: "terminal", outcome: { status: "failed", error: { message: text, detail:
      { reason: "model_error" } } } } }]
   and return null.
