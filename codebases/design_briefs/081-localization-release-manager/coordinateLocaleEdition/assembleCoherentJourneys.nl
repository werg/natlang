---
args: {"segments": "ContextSegment[]", "candidates": "LocaleCandidate[]", "reviews": "LocaleFidelityReview[]", "prior": "LocaleBundle[]", "change": "LocaleSourceChange"}
returns: "LocaleBundle[]"
---
For each locale and user journey collect exactly its source/context/glossary-compatible candidates and independent reviews. Reject a bundle mixing old and new critical instructions or containing an unresolved material meaning/structure defect. When a locale is delayed, apply only an explicit approved fallback and test that the entire affected journey remains coherent. Preserve actual prior released editions until a new exact bundle is approved; bind release to concrete artifact IDs and record actual deployment receipts separately.
