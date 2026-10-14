---
description: Answer every declared query from its actual dimension and historical view.
args:
  source: DialogueSource
  scoped: ScopedConversation
  episodePlan: EpisodePlan
  evolution: PerspectiveEvolution
  request: DialogueRequest
returns: QueryAnswers
---
1. For each original query call resolveQueryFocus(query, source, scoped, evolution). Consume its actual
   participant/time/dimension focus and the last source-established view at or before the requested time.
2. Call joinProof(focus, source, scoped, episodePlan, evolution). Consume the actual support/defeat graph
   and its actual returned challenge; a convenient globally true proposition cannot substitute for this proof.
3. Derive QueryObservable from the actual focus and joined evidence: statement speaker/due from its
   attributed act, belief from current endorsed or defeated atoms, knowledge from endorsement plus available
   corroboration, commitment from its actual condition/status, verified truth from bounded physical facts.
4. Copy actualGraph.challenge into QueryAnswer.challenge; a missing challenge remains unresolved.
   Keep irrelevant observable fields null. Known absence of endorsement is not false endorsement;
   ambiguous referents retain their exact eligible set and unresolved illocution keeps actual alternatives.
5. A query that asks whether a reporter created a promise must inspect the reporter's own act scope,
   not merely find any embedded original promise. Explain absence with report-only source evidence.
6. Set supported/refuted/ambiguous/unresolved status from actual proof and challenge. A failed challenge
   leaves unresolved and does not silently replace the answer with an independently reconstructed preferred one.
7. Return actual QueryAnswers in original query order, with generated proof identity consistently joined
   and free explanatory prose. Exact invariants are declared fields, not serialized semantic string codes.
