---
args: { judgments: PairIdentityJudgment[], resolution: EntityResolutionRequest }
returns: ConstrainedIdentityClusters
---
Consider proposed same-entity edges one at a time. Before accepting an edge, inspect all members already in both components for established cannot-link pairs and impossible overlapping identity histories. If any conflict appears, keep components separate and route the triangle or chronology conflict to review instead of applying transitive closure.
