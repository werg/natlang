---
description: Join one actual response to its prompted field and consultant judgment.
args:
  utterance: UtteranceReading
  source: CorpusSource
  effective: EffectiveNoteSet
returns: ElicitationAlignment
---
1. Select the utterance's actual session, prompt clauses and original situation by original IDs. Preserve a missing link instead of substituting a visually similar elicitation.
2. Follow effective entries for notes referring to this utterance. Retain disputed authority and unadopted corrections separately from the actual judgment note set.
3. Call judgeResponseScope(utterance, session, situation, notes, source.rules) with these actual typed values. Bind actualScope without re-judging which fields would discriminate candidates best.
4. A accepted answer to who is being discussed constrains topic, while a role-identification response may constrain agent/patient. Consume actualScope.assertedFields/excludedFields independently of all true scene facts.
5. A consultant's rejection of a proposed gloss constrains that proposed meaning, not necessarily the grammar's segmentation or surface acceptability. Retain this difference in scope and effective-note IDs.
6. Validate original IDs and exact quoted support, then retain actualScope in ElicitationAlignment. If a needed prompt/authority premise is unresolved, preserve the blocker in the alignment.
7. Return the actual joined alignment ready for one candidate/evidence attempt; it is not a predicted linguistic interpretation.
