---
args: {"changes": "EvidenceItem[]", "segments": "ContextSegment[]", "prior": "LocaleEditionLedger"}
returns: "LocaleInvalidation"
---
For each changed source or glossary item call compareChangedMeaning(change, segments, prior), collecting affected candidateIds. Derive their segmentKeys from actual candidate dependencies, and include every new segment absent from the prior ledger even when no previous candidate exists. Invalidate only candidates depending on the affected source meaning, concept/sense or authority edition. Preserve punctuation-only candidates only when the comparison establishes unchanged meaning and exact structural requirements remain satisfied. A glossary change for one sense cannot invalidate every string sharing that spelling. Return distinct segmentKeys and candidateIds as LocaleInvalidation.
