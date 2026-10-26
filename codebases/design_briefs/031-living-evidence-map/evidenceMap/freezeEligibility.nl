---
args: { request: EvidenceMapRequest, policies: PolicyCandidate[] }
returns: EligibilitySnapshot
---
Bind inclusion and exclusion rules to the review's named policy version and cutoff date before screening. Make population, intervention/exposure, comparator, outcome, setting and acceptable design explicit. Keep ambiguous or unreviewed criteria in unresolvedRules rather than widening the corpus silently.
