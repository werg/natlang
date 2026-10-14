---
description: Resolve authoritative note slots without erasing original alternatives.
args:
  notes: NoteReading[]
  corrections: CorrectionIndex
  rules: SourceClause[]
returns: EffectiveNoteSet
---
1. Read actual source rules for consultant judgment authority, notebook adoption and targeted version precedence. Interpret those rules in NL before selecting any effective slot.
2. For each original note/slot inventory actual corrections that target it, retaining their actual scope/adopted verdict and explicit exclusions. Group a correction and its explicit adoption when they support the same original replacement note; preserve all original support IDs but count distinct effective versions, not repeated mentions, when judging competition.
3. If no adopted applicable correction governs the slot, retain the original note. An unadopted suggestion remains visible context, not an automatic replacement.
4. If one source-authorized adopted replacement governs, return originalNoteId→effectiveNoteId with status replaced and its actual correction IDs. Preserve any untargeted transcription or temporal slot unchanged.
5. If multiple adopted replacements have an explicit source order/override relation, follow that relation. Otherwise retain disputed status and null effectiveNoteId rather than choose a linguistically appealing gloss.
6. Preserve source-grounded original/replacement quotations in every decided entry; validate exact note/slot identities. Do not overwrite original NoteReading objects.
7. Return EffectiveNoteSet so evidence applicability, grammar tests and later patch authorization can each consume the same actual version decision.
