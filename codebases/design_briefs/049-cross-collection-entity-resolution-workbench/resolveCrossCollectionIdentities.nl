---
args: { resolution: EntityResolutionRequest }
returns: EntityResolutionProposal
---
1. Bind candidates = generateAccessBoundIdentityCandidates(resolution). Preserve blocking provenance and audit sampled pairs outside the candidate blocks for missed recall.
2. Bind judgments = weighIdentityPairEvidence(candidates, resolution). Separate same-entity support from affiliation, succession, and address coincidence; missing attributes remain unknown.
3. Bind clusters = mergeOnlyConsistentIdentityComponents(judgments, resolution). Test every proposed component merge against cannot-link pairs and effective-time chronology before joining it.
4. Bind canonical = attributeFieldClaims(clusters, resolution). Choose values per field authority and effective time while preserving aliases and meaningful disagreements.
5. Bind revision = stageReversibleIdentityCorrection(canonical, clusters, resolution). A reversal propagates a split to downstream links; a revoked source is removed and all dependent matches are reassessed.
6. Return { resolutionId: resolution.resolutionId, candidates, judgments, clusters, canonical, revision, commitAllowed: false }.
