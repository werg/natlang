---
args: { candidates: IdentityPairCandidate[], resolution: EntityResolutionRequest }
returns: PairIdentityJudgment[]
---
For each pair compare stable identifiers, name, chronology, source reliability, and context. Shared name or employer alone cannot establish identity. A shared address between parent and subsidiary supports a relationship edge, not a merge. Bind recycled identifiers to disjoint validity intervals. Preserve positive, negative, and missing evidence separately and return same, different, related, or unresolved.
For each candidate bind strength = qualifyIdentityEvidenceStrength(candidate, resolution.sourceEvidence). Do not turn a weak shared attribute into a same-entity judgment.
