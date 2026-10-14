---
description: Consume one original hypothesis-evidence pair through distinct actual linguistic judgments.
args:
  state: GrammarFrontier
returns: GrammarFrontier
uses: [analyzeCorpus/interpretSegmentation, analyzeCorpus/interpretRoles]
---
1. If state.pending is empty, return state. Otherwise remove exactly its first original pair and bind actualHypothesis and actualEvidence from the retained original domain; preserve all other pending pairs and completed results.
2. If this actual hypothesis is already rejected, append SkippedPair naming its actual first contradiction attempt. Do not call linguistic children or invent a support result for this skipped evidence.
3. Otherwise call interpretSegmentation(actualHypothesis, actualEvidence.utterance, state.source, state.graph). Bind actualSegments with exact original surface spans and the actual boundary decisions.
4. Call interpretRoles(actualHypothesis, actualSegments, actualEvidence, state.source). Bind actualRoles; it predicts the current hypothesis meaning, not the scene's preferred answer.
5. Call compareJudgment(actualHypothesis, actualSegments, actualRoles, actualEvidence, state.source). Retain the actual returned compatibility and the incoming hypothesis in a new HypothesisAttempt.
6. For contradiction mark only that actual hypothesis rejected with the returned original evidence support. For support retain it for the next pending evidence. For unknown preserve the blocker and allow later actual evidence tests without relabeling the unknown as support or contradiction.
7. A transport/type failure preserves prior attempts and an explicit unresolved current pair outcome; never discard a complete earlier evidence ledger or rebuild predictions directly from source tables.
8. Bind next:GrammarFrontier to the actual shorter pending list, actual updated hypothesis rows, attempts/skips and unresolved reasons. Finish immediately with eval({code:"return next;",finish:true}); the parent owns the next pair.
