---
description: Consume one original target and every actual survivor interpretation.
args:
  state: TranslationFrontier
returns: TranslationFrontier
---
1. If pending is empty, return state. Otherwise remove only the first original target; bind its actual utterance and context from state.source.
2. For each actual state.grammar.survivorIds member resolve its actual hypothesis and call interpretPhrase(target, hypothesis, state.source, state.graph). Preserve each actual returned MeaningReading.
3. A child with missing contextual/linguistic premises returns an unresolved reading, not a replacement generated from a different survivor. Retain source identity and hypothesis ID in that actual row.
4. Call compareReadings(target, actualReadings, state.source.rules). Bind actualAlternatives with groups and consequential ambiguity; do not retranslate while aggregating.
5. Require one actual reading per actual completed survivor and correct target/utterance/hypothesis joins. Failed returns remain explicit blockers; no completed prior target is discarded.
6. Append actualAlternatives to completed. Bind next to the shorter actual typed frontier preserving all prior targets and original source/grammar objects.
7. Finish immediately with eval({code:"return next;",finish:true}); the parent owns the next original target.
