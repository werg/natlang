---
description: Interpret original correction scope, exclusions and adopted replacement.
args:
  unit: SourceClause
  clauses: SourceClause[]
  notes: NoteReading[]
  sessions: SessionIndex
  rules: SourceClause[]
returns: CorrectionMeaning
---
1. Bind primary to the actual selected unit. Directly invoke a typed scoped NL CorrectionMeaning literal capturing primary, actual original context clauses, notes, sessions and governing rules. Follow referenced target, replacement and explicit adoption dependencies from that selected primary; sibling context supplies evidence rather than changing the selected correction. Exclude survivor hypotheses and repaired gloss text.
2. Keep correctionId equal to primary.clauseId. Resolve deictic references such as this gloss or the copied second line against the original note IDs and source writing sequence. Preserve alternatives as unresolved if the source leaves several genuine targets.
3. Decide whether surface, gloss, judgment or context is being revised. Explicitly retain excluded slots and a still-valid transcription even when its adjoining translation is wrong.
4. Follow actual original correction/adoption references to the actual replacement note and whether a governing source convention or explicit researcher's action adopts it. Consultant authority and notebook adoption are distinct facts.
5. If only part of a compound gloss changes, return its actual targetSlotIds, not all semantic fields of the utterance. Keep untargeted role/temporal uncertainty outside the correction.
6. Return original IDs, scope, adopted status, explanatory reason and exact source quotes. Parent validates identity/substring and later version resolution consumes the actual judgment.
7. Missing correction attribution, adoption or replacement linkage stays null/unresolved. A plausible translation improvement alone is not a source-authorized correction.
