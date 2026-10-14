---
description: Interpret original correction scope, exclusions and adopted replacement.
args:
  clauses: SourceClause[]
  notes: NoteReading[]
  sessions: SessionIndex
returns: CorrectionMeaning
---
1. Capture the selected original correction clauses and actual note/session inventory in a typed scoped NL CorrectionMeaning literal. Exclude survivor hypotheses and repaired gloss text.
2. Resolve deictic references such as this gloss or the copied second line against the original note IDs and source writing sequence. Preserve alternatives as unresolved if the source leaves several genuine targets.
3. Decide whether surface, gloss, judgment or context is being revised. Explicitly retain excluded slots and a still-valid transcription even when its adjoining translation is wrong.
4. Resolve the actual replacement note and whether a governing source convention or explicit researcher's action adopts it. Consultant authority and notebook adoption are distinct facts.
5. If only part of a compound gloss changes, return its actual targetSlotIds, not all semantic fields of the utterance. Keep untargeted role/temporal uncertainty outside the correction.
6. Return original IDs, scope, adopted status, explanatory reason and exact source quotes. Parent validates identity/substring and later version resolution consumes the actual judgment.
7. Missing correction attribution, adoption or replacement linkage stays null/unresolved. A plausible translation improvement alone is not a source-authorized correction.
