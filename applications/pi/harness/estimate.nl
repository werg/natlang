---
description: Context size estimate (pi-durable compaction.ts estimateContext, spec §8.3). The tokens a request over view followed by extra would hold.
args:
  view: ContextView
  extra: Message[]
returns: number
---
Estimate how many tokens a request over view.messages followed by extra holds. Compute it exactly in eval.

1. Measured response. Find the newest assistant message whose usage reports a context size, among the entries that
   come after the head marker: walk view.entries from the last index down; skip an entry whose id is at most
   view.head.id (when view.head is not null). In each remaining entry k, take the LAST assistant message of
   view.contributions[k] whose size(usage) > 0. Stop at the first entry that has one. size(usage) = usage.totalTokens
   when it is non-zero, else usage.input + usage.output + usage.cacheRead + usage.cacheWrite.
2. tokens = size(measured.usage), plus the estimates of every message of view.messages after measured (find measured
   in view.messages by its last position: the same message object, equal by its JSON), plus the estimates of every
   message of extra. Without a measured message, tokens = the estimates of every message of view.messages plus those
   of extra. ai.estimateTokens(messages) returns one estimate per message; sum them.

Return tokens.
