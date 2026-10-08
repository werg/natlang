---
description: Compaction range selection (pi-durable compaction.ts selectCut and isCandidate, spec §8.7). The index in view.entries of the first entry a summary keeps, or null when there is nothing to compact.
args:
  view: ContextView
  keepRecentTokens: number
returns: number | null
---
Choose where a summary would cut view's context: the index (into view.entries and view.contributions) of the first
entry kept verbatim, or null. It is a pure function of view; compute it exactly in eval.

Notation: C = view.contributions (C[k] is entry k's messages). start = 1 when view.head is not null (index 0 is the
head marker, which is never kept), else 0.

1. Candidates. An index k >= start is a candidate when C[k]'s first message decides so:
   - C[k] is empty, or its first message is a toolResult or system message: not a candidate;
   - its first message is an assistant message: a candidate;
   - its first message is a user message: find the nearest k2 < k (walking down from k - 1) whose C[k2] contains an
     assistant message, and take the toolCall ids of the LAST assistant message in C[k2]. No such entry, or no ids: a
     candidate. Otherwise scan forward over the messages of C[k], C[k+1], …, in order: meeting an assistant message
     that is not C[k][0] first means a candidate; meeting a toolResult whose toolCallId is one of those ids first means
     not a candidate (a result of those calls still belongs before this user message); reaching the end means a
     candidate.
   candidates = the candidate indexes in increasing order.
2. Walk back. kept = 0. For k from C.length - 1 down to start: add the token estimates of C[k]'s messages
   (ai.estimateTokens(C[k]) gives one number per message; sum them) to kept. If kept < keepRecentTokens, continue
   down. At the first k where kept >= keepRecentTokens: cut = the first candidate >= k, or, when there is none, the
   last candidate; stop walking. If the walk ends without reaching keepRecentTokens, or there are no candidates,
   return null.
3. Something to summarize. Return cut only if some index j with start <= j < cut has a non-empty C[j]; otherwise
   return null.

Example: entries 1 user, 2 assistant (calls read), 3 its 30k-token result, 4 assistant, 5 user, 6 assistant, no head,
keepRecentTokens 20000: the walk reaches 20000 at index 2 (entry 3); the first candidate at or after it is index 3
(entry 4), so the result is 3.
