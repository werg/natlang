---
args: {"segment": "ContextSegment", "candidate": "LocaleCandidate", "receipts": "LocaleStructureReceipt[]", "evidence": "EvidenceItem[]"}
returns: "LocaleFidelityReview"
---
Call challengeSourceMeaning(segment, candidate, evidence), then call compareProtectedTokens(segment.sourceText, candidate.text, segment.protectedTokens). Preserve the challenge's material defects independently of exact token success. Accept structural receipts only for this actual candidate/source and rendered artifact, with an identified validator version and every required check passing. Missing or mismatched receipts remain defects. Distinguish harmless natural variation from a changed obligation, product behavior or terminology sense.
